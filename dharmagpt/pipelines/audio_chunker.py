"""
audio_chunker.py
Receives Sarvam STT transcript (with word timestamps + diarization),
applies pause-boundary chunking, translates chunks in parallel when needed,
and stores enriched chunks in Postgres for later incremental vector sync.

All backends pluggable via .env:
  TRANSLATION_BACKEND = sarvam | anthropic | ollama | indictrans2 | skip
  EMBEDDING_BACKEND   = openai | local_hash
  RAG_BACKEND         = local | pinecone
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import uuid

import structlog

from core.config import get_settings
from core.chunk_store import upsert_chunk
from core.backends.translation import get_translator

log = structlog.get_logger()
settings = get_settings()

SACRED_MARKERS = [
    "shri ram", "jai ram", "jai hanuman", "namah shivaya",
    "om namo", "sita ram", "jai siya ram", "pavan putra",
    "anjaneya", "bajrangbali",
]
SHLOKA_PATTERN = re.compile(r"[।॥|]+")


def _detect_speaker(text: str) -> str:
    has_danda = bool(SHLOKA_PATTERN.search(text))
    en_ratio = len(re.findall(r"\b[a-zA-Z]{3,}\b", text)) / max(len(text.split()), 1)
    if has_danda or any(m in text.lower() for m in SACRED_MARKERS):
        return "chanting"
    return "commentary_english" if en_ratio > 0.5 else "commentary_hindi"


def _chunk_by_pause(words: list[dict], min_words: int = 12, max_words: int = 70) -> list[dict]:
    chunks, buf, start = [], [], 0.0
    for i, w in enumerate(words):
        buf.append(w)
        is_last = i == len(words) - 1
        gap = (words[i + 1].get("start", 0) - w.get("end", 0)) if not is_last else 999
        text_so_far = " ".join(x.get("word", "") for x in buf)
        should_cut = (
            (gap > 0.8 and len(buf) >= min_words)
            or bool(SHLOKA_PATTERN.search(text_so_far) and len(buf) >= min_words)
            or len(buf) >= max_words
            or is_last
        )
        if should_cut and buf:
            text = re.sub(r"\s+", " ", text_so_far).strip()
            chunks.append({
                "text": text,
                "start": start,
                "end": w.get("end", 0),
                "speaker": _detect_speaker(text),
                "has_shloka": bool(SHLOKA_PATTERN.search(text)),
            })
            buf = []
            start = words[i + 1].get("start", 0) if not is_last else 0
    return chunks


def _fallback_chunk(text: str) -> list[dict]:
    """When timestamps are unavailable, chunk by sentence boundaries."""
    segs = re.split(r"[।॥|]{1,2}|\.(?=\s)", text)
    chunks, buf = [], []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        buf.append(seg)
        if len(" ".join(buf).split()) >= 20:
            t = " ".join(buf)
            chunks.append({
                "text": t, "start": None, "end": None,
                "speaker": _detect_speaker(t),
                "has_shloka": bool(SHLOKA_PATTERN.search(t)),
            })
            buf = []
    if buf:
        t = " ".join(buf)
        chunks.append({
            "text": t, "start": None, "end": None,
            "speaker": _detect_speaker(t),
            "has_shloka": bool(SHLOKA_PATTERN.search(t)),
        })
    return chunks


def _normalize_language_code(language_code: str) -> str:
    lang = (language_code or "").strip().lower()
    if not lang or lang.startswith("en"):
        return "en"
    if "-" in lang:
        return lang.split("-", 1)[0]
    return lang


def _translate_chunks_parallel(
    chunks: list[dict],
    *,
    source_lang: str,
    target_lang: str = "en",
) -> list[str | None]:
    """Translate chunks in parallel using the configured TRANSLATION_BACKEND."""
    if not chunks:
        return []

    translator = get_translator()
    results: list[str | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
        futures = {
            executor.submit(translator.translate, chunk["text"], source_lang, target_lang): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result.text if not result.skipped else None
            except Exception as exc:
                log.warning("audio_translation_failed", chunk_index=idx, error=str(exc))
                results[idx] = None

    return results


def _summarize_provenance(backend_name: str, translated_count: int) -> dict:
    if not translated_count:
        return {
            "translation_mode": None,
            "translation_backend": None,
            "translation_version": backend_name,
            "translation_fallback_reason": None,
            "translation_attempted_backends": None,
        }
    return {
        "translation_mode": backend_name,
        "translation_backend": backend_name,
        "translation_version": backend_name,
        "translation_fallback_reason": None,
        "translation_attempted_backends": [backend_name],
    }


async def chunk_and_index(
    transcript_data: dict,
    filename: str,
    file_metadata: dict,
    dataset_id: str = "",
) -> dict:
    """Main entry: chunk -> translate -> store chunks for later vector sync."""
    words = transcript_data.get("words", [])
    raw_text = transcript_data.get("transcript", "")

    raw_chunks = _chunk_by_pause(words) if words else _fallback_chunk(raw_text)
    if not raw_chunks:
        return {
            "chunks_created": 0,
            "translated_transcript": None,
            "translation_mode": None,
            "translation_backend": None,
            "translation_version": None,
            "translation_fallback_reason": None,
            "translation_attempted_backends": None,
        }

    source_lang = _normalize_language_code(file_metadata.get("language_code", "en"))
    needs_translation = source_lang != "en"
    translated_chunks: list[str] = []

    # Fastest path: Sarvam STT already returned clip-level English — reuse it directly.
    sarvam_en = (transcript_data.get("text_en_sarvam") or "").strip()

    if needs_translation and sarvam_en:
        translated_chunks = [sarvam_en] * len(raw_chunks)
        translator_backend = "sarvam_stt_translate"
        log.info("using_sarvam_clip_translation", file=filename, chunks=len(raw_chunks))
    elif needs_translation:
        per_chunk = _translate_chunks_parallel(raw_chunks, source_lang=source_lang, target_lang="en")
        translated_chunks = [t or "" for t in per_chunk]
        translator_backend = get_translator().backend_name
    else:
        translated_chunks = ["" for _ in raw_chunks]
        translator_backend = "none"

    provenance = _summarize_provenance(
        translator_backend,
        len([t for t in translated_chunks if t]),
    )
    stem = filename.rsplit(".", 1)[0]

    stored = 0
    for i, (chunk, translated) in enumerate(zip(raw_chunks, translated_chunks)):
        chunk_id = f"audio_{stem}_{uuid.uuid4().hex[:8]}_{i:04d}"
        record_metadata = {
            "source_type": "audio",
            "source_file": filename,
            "source": file_metadata.get("source") or stem,
            "source_title": file_metadata.get("source_title") or file_metadata.get("description", stem),
            "text": chunk["text"],
            "text_preview": chunk["text"][:300],
            "start_time_sec": chunk.get("start") or "",
            "end_time_sec": chunk.get("end") or "",
            "speaker_type": chunk["speaker"],
            "has_shloka": chunk["has_shloka"],
            "section": file_metadata.get("section") or "",
            "language": file_metadata.get("language_code", "hi-IN"),
            "description": file_metadata.get("description", stem),
            "citation": f"Audio: {file_metadata.get('description', stem)}",
            "word_count": len(chunk["text"].split()),
            "transcription_mode": file_metadata.get("transcription_mode", "sarvam_stt"),
            "transcription_version": file_metadata.get("transcription_version", "saaras:v3"),
            "translation_mode": provenance["translation_mode"] or "",
            "translation_backend": provenance["translation_backend"] or "",
            "translation_version": provenance["translation_version"] or "",
            "translation_fallback_reason": provenance["translation_fallback_reason"] or "",
            "translation_attempted_backends": provenance["translation_attempted_backends"] or [],
            "embedding_backend": "",
        }
        if dataset_id:
            record_metadata["dataset_id"] = dataset_id
        if translated.strip():
            record_metadata["translated_text"] = translated.strip()
            record_metadata["translated_text_preview"] = translated[:300]

        upsert_chunk(
            chunk_id,
            text=chunk["text"],
            translated_text=translated.strip(),
            metadata=record_metadata,
            vector_status="pending",
        )
        stored += 1

    translated_transcript = (
        "\n".join(t for t in translated_chunks if t.strip()) if needs_translation else None
    )
    log.info(
        "audio_indexed",
        file=filename,
        chunks=len(raw_chunks),
        vector_db="postgres",
        vectors=0,
        translation_backend=provenance["translation_backend"],
        chunks_stored=stored,
    )
    return {
        "chunks_created": len(raw_chunks),
        "translated_transcript": translated_transcript,
        "translation_mode": provenance["translation_mode"],
        "translation_backend": provenance["translation_backend"],
        "translation_version": provenance["translation_version"],
        "translation_fallback_reason": provenance["translation_fallback_reason"],
        "translation_attempted_backends": provenance["translation_attempted_backends"],
        "vector_db": "postgres",
        "vectors_upserted": 0,
        "vector_status": "pending",
        "embedding_backend": "",
    }
