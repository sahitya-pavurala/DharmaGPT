"""
tests/backend/test_rag_engine.py
Run: pytest tests/backend/ -v
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from models.schemas import QueryRequest, QueryMode, SourceChunk


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_source(citation="Valmiki Ramayana, Sundara Kanda, Sarga 15", score=0.9):
    return SourceChunk(
        text="Hanuman found Sita in the Ashoka grove.",
        citation=citation,
        section="Sundara Kanda",
        chapter=15,
        score=score,
        source_type="text",
        audio_timestamp=None,
        url="https://www.valmikiramayan.net/utf8/sundara/sarga15/",
    )


# ── Retrieval tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_filtered_results():
    """Low-score results should be filtered out."""
    from core.retrieval import retrieve
    high = make_source(score=0.9)
    low  = make_source(score=0.1)

    mock_results = MagicMock()
    mock_results.matches = [
        MagicMock(score=high.score, metadata={"text_preview": high.text, "citation": high.citation,
                  "kanda": "Sundara Kanda", "sarga": 15, "source_type": "text", "url": ""}),
        MagicMock(score=low.score,  metadata={"text_preview": low.text,  "citation": low.citation,
                  "kanda": "Sundara Kanda", "sarga": 15, "source_type": "text", "url": ""}),
    ]

    from core import retrieval

    original_rag_backend = retrieval.settings.rag_backend
    original_backend = retrieval.settings.vector_db_backend
    original_min_score = retrieval.settings.rag_min_score
    retrieval.settings.rag_backend = "pinecone"
    retrieval.settings.vector_db_backend = "pinecone"
    retrieval.settings.rag_min_score = 0.35
    try:
        with patch("core.retrieval.embed_query", new=AsyncMock(return_value=[0.1] * 3072)), \
             patch("core.retrieval.get_pinecone") as mock_pc:
            mock_pc.return_value.Index.return_value.query.return_value = mock_results
            results = await retrieve("Where did Hanuman find Sita?")
    finally:
        retrieval.settings.rag_backend = original_rag_backend
        retrieval.settings.vector_db_backend = original_backend
        retrieval.settings.rag_min_score = original_min_score

    assert len(results) == 1
    assert results[0].score == 0.9


# ── Prompts tests ──────────────────────────────────────────────────────────────

def test_guidance_prompt_contains_context():
    from core.prompts import get_system_prompt
    ctx = "[PASSAGE 1 — Sundara Kanda, Sarga 15]\nHanuman found Sita."
    prompt = get_system_prompt("guidance", ctx)
    assert "Sundara Kanda" in prompt
    assert "Hanuman found Sita" in prompt


def test_children_prompt_no_context_block():
    from core.prompts import get_system_prompt
    prompt = get_system_prompt("children", "")
    assert "5–12" in prompt


def test_scholar_prompt_requires_citations():
    from core.prompts import get_system_prompt
    prompt = get_system_prompt("scholar", "")
    assert "citation" in prompt.lower() or "cite" in prompt.lower()


# ── Schema validation ──────────────────────────────────────────────────────────

def test_query_request_validates_mode():
    req = QueryRequest(query="What is dharma?", mode=QueryMode.guidance)
    assert req.mode == QueryMode.guidance


def test_query_request_rejects_short_query():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(query="?", mode=QueryMode.guidance)


def test_query_request_caps_history():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(query="Tell me about Hanuman", mode=QueryMode.guidance, history=history)


# ── RAG engine integration (mocked) ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_answer_returns_response():
    from core.rag_engine import answer
    req = QueryRequest(query="How did Hanuman find Sita?", mode=QueryMode.story)

    mock_chunks = [make_source()]
    mock_answer = "Hanuman found Sita in the Ashoka grove after a long search. [Source: Sundara Kanda, Sarga 15]"

    with patch("core.rag_engine.retrieve", new=AsyncMock(return_value=mock_chunks)), \
         patch("core.rag_engine._call_llm", new=AsyncMock(return_value=mock_answer)):
        response = await answer(req)

    assert response.answer == mock_answer
    assert len(response.sources) == 1
    assert response.mode == QueryMode.story
    assert response.query_id  # non-empty UUID
