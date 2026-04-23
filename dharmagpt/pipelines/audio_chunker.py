"""
audio_chunker.py
Receives Sarvam STT transcript (with word timestamps + diarization),
applies pause-boundary chunking, and upserts chunks to Pinecone.
"""
import re
import uuid
import structlog
from openai import AsyncOpenAI
from pinecone import Pinecone
from core.config import get_settings

log = structlog.get_logger()
settings = get_settings()

SACRED_MARKERS = [
    "shri ram", "jai ram", "jai hanuman", "namah shivaya", "om namo",
    "sita ram", "jai siya ram", "pavan putra", "anjaneya", "bajrangbali",
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
    """When timestamps unavailable, chunk by sentence boundaries."""
    segs = re.split(r"[।॥|]{1,2}|\.(?=\s)", text)
    chunks, buf = [], []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        buf.append(seg)
        if len(" ".join(buf).split()) >= 20:
            t = " ".join(buf)
            chunks.append({"text": t, "start": None, "end": None,
                           "speaker": _detect_speaker(t), "has_shloka": bool(SHLOKA_PATTERN.search(t))})
            buf = []
    if buf:
        t = " ".join(buf)
        chunks.append({"text": t, "start": None, "end": None,
                       "speaker": _detect_speaker(t), "has_shloka": bool(SHLOKA_PATTERN.search(t))})
    return chunks


async def chunk_and_index(transcript_data: dict, filename: str, file_metadata: dict) -> int:
    """Main entry: chunk transcript → embed → upsert to Pinecone. Returns chunk count."""
    words = transcript_data.get("words", [])
    raw_text = transcript_data.get("transcript", "")

    raw_chunks = _chunk_by_pause(words) if words else _fallback_chunk(raw_text)
    if not raw_chunks:
        return 0

    # Embed all chunks in one batch
    openai = AsyncOpenAI(api_key=settings.openai_api_key)
    texts = [c["text"] for c in raw_chunks]
    embed_response = await openai.embeddings.create(model=settings.embedding_model, input=texts)
    vectors = [r.embedding for r in embed_response.data]

    # Upsert to Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    stem = filename.rsplit(".", 1)[0]
    records = []
    for i, (chunk, vec) in enumerate(zip(raw_chunks, vectors)):
        records.append({
            "id": f"audio_{stem}_{uuid.uuid4().hex[:8]}_{i:04d}",
            "values": vec,
            "metadata": {
                "source_type": "audio",
                "source_file": filename,
                "text_preview": chunk["text"][:300],
                "start_time_sec": chunk.get("start") or "",
                "end_time_sec": chunk.get("end") or "",
                "speaker_type": chunk["speaker"],
                "has_shloka": chunk["has_shloka"],
                "kanda": file_metadata.get("kanda") or "",
                "language": file_metadata.get("language_code", "hi-IN"),
                "description": file_metadata.get("description", stem),
                "citation": f"Audio: {file_metadata.get('description', stem)}",
                "word_count": len(chunk["text"].split()),
            },
        })

    # Batch upsert
    BATCH = 100
    for i in range(0, len(records), BATCH):
        index.upsert(vectors=records[i:i + BATCH])

    log.info("audio_indexed", file=filename, chunks=len(records))
    return len(records)
