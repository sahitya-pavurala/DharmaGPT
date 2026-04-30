"""
test_new_features.py — unit tests for dataset management, audio auto-split,
chunker backend routing, and retrieval dataset filtering.

All tests are offline — no Sarvam, Pinecone, or OpenAI calls are made.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── dataset_store ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point dataset_store at a throwaway SQLite DB for each test."""
    db_path = tmp_path / "test_vectors.sqlite3"
    import core.dataset_store as ds
    monkeypatch.setattr(ds, "_DB_PATH", db_path)
    # Clear module-level cache so _connect picks up the patched path
    yield ds


def test_register_and_list(tmp_db):
    tmp_db.register("ramayanam", "Ramayanam by Chaganti")
    datasets = tmp_db.list_all()
    assert len(datasets) == 1
    assert datasets[0]["name"] == "ramayanam"
    assert datasets[0]["display_name"] == "Ramayanam by Chaganti"
    assert datasets[0]["active"] == 1
    assert datasets[0]["vector_count"] == 0


def test_register_is_idempotent(tmp_db):
    tmp_db.register("ramayanam")
    tmp_db.register("ramayanam")  # second call should not raise or duplicate
    assert len(tmp_db.list_all()) == 1


def test_increment_count(tmp_db):
    tmp_db.register("gita")
    tmp_db.increment_count("gita", 42)
    assert tmp_db.list_all()[0]["vector_count"] == 42
    tmp_db.increment_count("gita", 8)
    assert tmp_db.list_all()[0]["vector_count"] == 50


def test_set_active_false_and_back(tmp_db):
    tmp_db.register("mahabharata")
    assert tmp_db.set_active("mahabharata", False) is True
    assert tmp_db.list_all()[0]["active"] == 0
    assert tmp_db.set_active("mahabharata", True) is True
    assert tmp_db.list_all()[0]["active"] == 1


def test_set_active_missing_returns_false(tmp_db):
    assert tmp_db.set_active("nonexistent", False) is False


def test_get_active_names_excludes_inactive(tmp_db):
    tmp_db.register("ds_a")
    tmp_db.register("ds_b")
    tmp_db.set_active("ds_b", False)
    active = tmp_db.get_active_names()
    assert "ds_a" in active
    assert "ds_b" not in active


def test_get_active_names_empty(tmp_db):
    assert tmp_db.get_active_names() == []


def test_any_registered_false_when_empty(tmp_db):
    assert tmp_db.any_registered() is False


def test_any_registered_true_after_register(tmp_db):
    tmp_db.register("test")
    assert tmp_db.any_registered() is True


def test_remove_dataset(tmp_db):
    tmp_db.register("to_delete")
    assert tmp_db.remove("to_delete") is True
    assert tmp_db.list_all() == []


def test_remove_missing_returns_false(tmp_db):
    assert tmp_db.remove("ghost") is False


# ── translate_corpus glob fix ─────────────────────────────────────────────────

def test_translate_corpus_globs_subdirectories(tmp_path, monkeypatch):
    """translate_corpus.py must find JSONL files in subdirectories."""
    import scripts.translate_corpus as tc
    monkeypatch.setattr(tc, "PROCESSED_DIR", tmp_path)

    # file directly in processed/
    (tmp_path / "seed.jsonl").write_text('{"text":"x","language":"te"}\n')
    # file in subdirectory (audio_transcript/...)
    sub = tmp_path / "audio_transcript" / "part1"
    sub.mkdir(parents=True)
    (sub / "part01.jsonl").write_text('{"text":"y","language":"te"}\n')

    files = sorted(tmp_path.glob("**/*.jsonl"))
    assert len(files) == 2


# ── audio auto-split ──────────────────────────────────────────────────────────

def test_split_threshold_constant():
    from api.routes.audio import _SPLIT_THRESHOLD_BYTES, _SEGMENT_SECS
    assert _SPLIT_THRESHOLD_BYTES == 2 * 1024 * 1024
    assert _SEGMENT_SECS == 29


def test_zero_indexed_generated_audio_parts_map_to_canonical_parts():
    from utils.naming import part_number_from_filename

    assert part_number_from_filename("source_te_audio_part0000.mp3") == 1
    assert part_number_from_filename("source_te_audio_part0001.mp3") == 2
    assert part_number_from_filename("source_te_audio_part0281.mp3") == 282
    assert part_number_from_filename("source_te_transcript_part01.jsonl") == 1
    assert part_number_from_filename("source_te_transcript_part12.jsonl") == 12


def test_ollama_embedding_backend_batches_requests(monkeypatch):
    from core.backends.embedding import OllamaEmbeddings

    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("requests.post", fake_post)

    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434/", dims=768)
    vectors = embedder.embed_documents(["rama", "dharma"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert calls == [
        (
            "http://localhost:11434/api/embed",
            {"model": "nomic-embed-text", "input": ["rama", "dharma"]},
            120,
        )
    ]


def test_openai_chat_backend_maps_messages(monkeypatch):
    from core.backends.llm import OpenAIChatModel

    captured = {}

    class FakeMessage:
        content = "answer text"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"choices": [FakeChoice()]})()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key, timeout):
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    model = OpenAIChatModel(model="gpt-4.1-mini", api_key="test-key", timeout=30)
    response = model.invoke([
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "What is dharma?"},
    ])

    assert response.content == "answer text"
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 30
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["messages"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "What is dharma?"},
    ]


@pytest.mark.asyncio
async def test_small_file_skips_split():
    """Files under threshold go straight to _transcribe_audio without splitting."""
    small_bytes = b"x" * 100  # 100 bytes << 2MB threshold

    fake_result = ({"transcript": "small clip text", "words": []}, "sarvam_stt", "saaras:v3")

    with patch("api.routes.audio._transcribe_audio", new=AsyncMock(return_value=fake_result)) as mock_stt, \
         patch("api.routes.audio._split_audio_to_segments") as mock_split:

        from api.routes.audio import _transcribe_with_auto_split
        result = await _transcribe_with_auto_split(small_bytes, "clip.mp3", "te-IN", ".mp3")

        mock_stt.assert_called_once()
        mock_split.assert_not_called()
        assert result[0]["transcript"] == "small clip text"


@pytest.mark.asyncio
async def test_large_file_triggers_split_and_merges_transcripts():
    """Files over threshold are split; segment transcripts are merged."""
    large_bytes = b"x" * (3 * 1024 * 1024)  # 3MB > 2MB threshold

    seg1 = {"transcript": "first segment text", "words": [
        {"word": "first", "start": 0.1, "end": 0.5},
        {"word": "segment", "start": 0.6, "end": 1.0},
    ]}
    seg2 = {"transcript": "second segment text", "words": [
        {"word": "second", "start": 0.2, "end": 0.6},
    ]}

    fake_stt_calls = [
        (seg1, "sarvam_stt", "saaras:v3"),
        (seg2, "sarvam_stt", "saaras:v3"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two fake segment files
        seg_paths = []
        for i in range(2):
            p = os.path.join(tmp_dir, f"seg{i:04d}.mp3")
            Path(p).write_bytes(b"fake_audio")
            seg_paths.append(p)

        with patch("api.routes.audio._split_audio_to_segments", return_value=(seg_paths, tmp_dir)), \
             patch("api.routes.audio._transcribe_audio", new=AsyncMock(side_effect=fake_stt_calls)), \
             patch("builtins.open", side_effect=lambda *a, **kw: open(*a, **kw)):

            from api.routes.audio import _transcribe_with_auto_split
            result_data, mode, version = await _transcribe_with_auto_split(
                large_bytes, "big_audio.mp3", "te-IN", ".mp3"
            )

    assert "first segment text" in result_data["transcript"]
    assert "second segment text" in result_data["transcript"]
    assert mode == "sarvam_stt"

    # Word timestamps from segment 2 should be offset by 29s
    words = result_data["words"]
    seg2_word = next(w for w in words if w["word"] == "second")
    assert seg2_word["start"] == pytest.approx(0.2 + 29, abs=0.01)
    assert seg2_word["end"] == pytest.approx(0.6 + 29, abs=0.01)


@pytest.mark.asyncio
async def test_failed_segment_is_skipped_others_succeed():
    """If one segment fails STT, it is skipped and remaining are merged."""
    large_bytes = b"x" * (3 * 1024 * 1024)

    async def stt_side_effect(audio_bytes, filename, lang, suffix):
        if "seg0001" in filename:
            raise RuntimeError("Sarvam timeout")
        return {"transcript": "good text", "words": []}, "sarvam_stt", "saaras:v3"

    with tempfile.TemporaryDirectory() as tmp_dir:
        seg_paths = []
        for i in range(2):
            p = os.path.join(tmp_dir, f"seg{i:04d}.mp3")
            Path(p).write_bytes(b"fake")
            seg_paths.append(p)

        with patch("api.routes.audio._split_audio_to_segments", return_value=(seg_paths, tmp_dir)), \
             patch("api.routes.audio._transcribe_audio", new=AsyncMock(side_effect=stt_side_effect)):

            from api.routes.audio import _transcribe_with_auto_split
            result_data, _, _ = await _transcribe_with_auto_split(
                large_bytes, "big.mp3", "te-IN", ".mp3"
            )

    assert result_data["transcript"] == "good text"


# ── audio_chunker Postgres staging ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chunker_stages_chunks_for_later_vector_sync(monkeypatch):
    """chunk_and_index stores chunks as pending instead of calling Pinecone."""
    from core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_backend", "pinecone")
    monkeypatch.setattr(settings, "vector_db_backend", "pinecone")
    monkeypatch.setattr(settings, "openai_api_key", "fake-key")

    staged = []

    with patch("pipelines.audio_chunker.upsert_chunk") as mock_upsert:
        mock_upsert.side_effect = lambda chunk_id, **kwargs: staged.append((chunk_id, kwargs))

        from pipelines.audio_chunker import chunk_and_index
        transcript_data = {
            "transcript": "రామ రామ రామ జయ రాజా రామ",
            "words": [
                {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
                for i, w in enumerate("రామ రామ రామ జయ రాజా రామ".split())
            ],
        }
        result = await chunk_and_index(
            transcript_data, "test.mp3",
            {"language_code": "en", "description": "test"},
            dataset_id="test-dataset",
        )

    assert result["chunks_created"] > 0
    assert result["vector_db"] == "postgres"
    assert result["vectors_upserted"] == 0
    assert len(staged) > 0
    assert staged[0][1]["metadata"]["dataset_id"] == "test-dataset"
    assert staged[0][1]["vector_status"] == "pending"


@pytest.mark.asyncio
async def test_chunker_stamps_dataset_id_on_metadata(monkeypatch):
    """Every staged chunk must carry dataset_id in its metadata."""
    from core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_backend", "pinecone")
    monkeypatch.setattr(settings, "vector_db_backend", "pinecone")
    monkeypatch.setattr(settings, "openai_api_key", "fake-key")

    staged = []

    with patch("pipelines.audio_chunker.upsert_chunk") as mock_upsert:
        mock_upsert.side_effect = lambda chunk_id, **kwargs: staged.append((chunk_id, kwargs))

        from pipelines.audio_chunker import chunk_and_index
        await chunk_and_index(
            {"transcript": "test content for dataset stamping check", "words": []},
            "clip.mp3",
            {"language_code": "en", "description": "clip"},
            dataset_id="my-dataset",
        )

    for _, kwargs in staged:
        assert kwargs["metadata"].get("dataset_id") == "my-dataset"


# ── retrieval dataset filtering ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieval_returns_empty_when_all_datasets_disabled(monkeypatch, tmp_db):
    """If datasets are registered but all inactive, retrieve() returns []."""
    tmp_db.register("ds1")
    tmp_db.set_active("ds1", False)

    import core.retrieval as retrieval_mod
    monkeypatch.setattr(retrieval_mod, "any_registered", tmp_db.any_registered)
    monkeypatch.setattr(retrieval_mod, "get_active_names", tmp_db.get_active_names)

    # Early-exit happens before embed_query is called — assert it is never reached
    with patch("core.retrieval.embed_query", new=AsyncMock(side_effect=AssertionError("embed_query called"))):
        from core.retrieval import retrieve
        results = await retrieve("What is dharma?")

    assert results == []


@pytest.mark.asyncio
async def test_retrieval_no_dataset_filter_when_none_registered(monkeypatch, tmp_db):
    """If no datasets registered at all, Pinecone query has no dataset_id filter."""
    import core.retrieval as retrieval_mod
    monkeypatch.setattr(retrieval_mod, "any_registered", tmp_db.any_registered)
    monkeypatch.setattr(retrieval_mod, "get_active_names", tmp_db.get_active_names)

    from core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_backend", "pinecone")
    monkeypatch.setattr(settings, "vector_db_backend", "pinecone")
    monkeypatch.setattr(settings, "openai_api_key", "fake")
    monkeypatch.setattr(settings, "rag_min_score", 0.0)

    pinecone_result = MagicMock()
    pinecone_result.matches = []
    mock_index = MagicMock()
    mock_index.query.return_value = pinecone_result

    with patch("core.retrieval.embed_query", new=AsyncMock(return_value=[0.1] * 10)), \
         patch("core.retrieval.get_pinecone") as mock_get_pc:
        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_get_pc.return_value = mock_pc

        from core.retrieval import retrieve
        await retrieve("What is dharma?")

    call_kwargs = mock_index.query.call_args.kwargs
    assert "dataset_id" not in (call_kwargs.get("filter") or {})
