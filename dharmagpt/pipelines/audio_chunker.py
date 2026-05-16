"""
audio_chunker.py
Receives Sarvam STT transcript (with word timestamps + diarization),
applies pause-boundary chunking, and stores enriched chunks in Postgres for
later incremental vector sync.


Translation is NOT done automatically — translations are provided manually via
the discourse_translations table in Postgres.
"""
from __future__ import annotations

import re
import uuid

import structlog

from core.chunk_store import upsert_chunk


log = structlog.get_logger()

SACRED_MARKERS = [
    "shri ram", "jai ram", "jai hanuman", "namah shivaya",
    "om namo", "sita ram", "jai siya ram", "pavan putra",
    "anjaneya", "bajrangbali",
]
SHLOKA_PATTERN = re.compile(r"[।॥|]+")


def _detect_speaker(text: str) -> str:
    has_danda = bool(SHLOKA_PATTERN.search(text))
    en_ratio = len(re.findall(r"\b[a-zA-Z]{3,}\b", text)) / max(len(text.split()), 1)
    if has_danda or any(marker in text.lower() for marker in SACRED_MARKERS):
        return "chanting"
    return "commentary_english" if en_ratio > 0.5 else "commentary_hindi"


def _chunk_by_pause(words: list[dict], min_words: int = 12, max_words: int = 70) -> list[dict]:
    chunks, buf, start = [], [], 0.0
    for i, word in enumerate(words):
        buf.append(word)
        is_last = i == len(words) - 1
        gap = (words[i + 1].get("start", 0) - word.get("end", 0)) if not is_last else 999
        text_so_far = " ".join(item.get("word", "") for item in buf)
        should_cut = (
            (gap > 0.8 and len(buf) >= min_words)
            or bool(SHLOKA_PATTERN.search(text_so_far) and len(buf) >= min_words)
            or len(buf) >= max_words
            or is_last
        )
        if should_cut and buf:
            text = re.sub(r"\s+", " ", text_so_far).strip()
            chunks.append(
                {
                    "text": text,
                    "start": start,
                    "end": word.get("end", 0),
                    "speaker": _detect_speaker(text),
                    "has_shloka": bool(SHLOKA_PATTERN.search(text)),
                }
            )
            buf = []
            start = words[i + 1].get("start", 0) if not is_last else 0
    return chunks


def _fallback_chunk(text: str) -> list[dict]:
    """When timestamps are unavailable, chunk by sentence boundaries."""
    segments = re.split(r"[।॥|]{1,2}|\.(?=\s)", text)
    chunks, buf = [], []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        buf.append(segment)
        if len(" ".join(buf).split()) >= 20:
            chunk_text = " ".join(buf)
            chunks.append(
                {
                    "text": chunk_text,
                    "start": None,
                    "end": None,
                    "speaker": _detect_speaker(chunk_text),
                    "has_shloka": bool(SHLOKA_PATTERN.search(chunk_text)),
                }
            )
            buf = []
    if buf:
        chunk_text = " ".join(buf)
        chunks.append(
            {
                "text": chunk_text,
                "start": None,
                "end": None,
                "speaker": _detect_speaker(chunk_text),
                "has_shloka": bool(SHLOKA_PATTERN.search(chunk_text)),
            }
        )
    return chunks


async def chunk_and_index(
    transcript_data: dict,
    filename: str,
    file_metadata: dict,
    dataset_id: str = "",
) -> dict:
    """Main entry: chunk transcript and stage rows for later vector sync."""

    words = transcript_data.get("words", [])
    raw_text = transcript_data.get("transcript", "")

    raw_chunks = _chunk_by_pause(words) if words else _fallback_chunk(raw_text)
    if not raw_chunks:
        return {
            "chunks_created": 0,
            "vector_db": None,
            "vectors_upserted": 0,
            "vector_status": None,
            "embedding_backend": "",
        }

    stem = filename.rsplit(".", 1)[0]
    stored = 0

    for index, chunk in enumerate(raw_chunks):
        chunk_id = f"audio_{stem}_{uuid.uuid4().hex[:8]}_{index:04d}"
        metadata = {

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
            "embedding_backend": "",
            "chunk_index": index,

        }
        if dataset_id:
            metadata["dataset_id"] = dataset_id

        upsert_chunk(
            chunk_id,
            text=chunk["text"],
            translated_text="",
            metadata=metadata,

            vector_status="pending",
        )
        stored += 1


    log.info(
        "audio_indexed",
        file=filename,
        chunks=len(raw_chunks),
        vector_db="postgres",
        vectors=0,

        chunks_stored=stored,
    )
    return {
        "chunks_created": len(raw_chunks),

        "vector_db": "postgres",
        "vectors_upserted": 0,
        "vector_status": "pending",
        "embedding_backend": "",
    }
