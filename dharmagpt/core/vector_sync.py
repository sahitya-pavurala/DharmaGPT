from __future__ import annotations

from core.chunk_store import (
    count_chunks_by_vector_status,
    list_pending_chunks,
)
from core.config import get_settings
from core.retrieval import embed_texts
from core.postgres_db import connect


def _embedding_text(chunk: dict) -> str:
    translated = (chunk.get("translated_text") or "").strip()
    text = (chunk.get("text") or "").strip()
    return f"{text} | {translated}" if translated else text


def _metadata_for_pinecone(chunk: dict, embedding_backend: str) -> dict:
    meta = dict(chunk.get("metadata") or {})
    meta.update(
        {
            "text": chunk.get("text") or "",
            "text_preview": chunk.get("preview") or (chunk.get("text") or "")[:500],
            "source": chunk.get("source") or "",
            "source_title": chunk.get("source_title") or "",
            "source_type": chunk.get("source_type") or "text",
            "citation": chunk.get("citation") or "",
            "section": chunk.get("section") or "",
            "language": chunk.get("language") or "",
            "dataset_id": chunk.get("dataset_id") or "",
            "embedding_backend": embedding_backend,
        }
    )
    if chunk.get("translated_text"):
        meta["translated_text"] = chunk["translated_text"]
        meta["translated_text_preview"] = chunk.get("translated_preview") or chunk["translated_text"][:500]
    if chunk.get("start_time_sec") is not None:
        meta["start_time_sec"] = chunk["start_time_sec"]
    if chunk.get("end_time_sec") is not None:
        meta["end_time_sec"] = chunk["end_time_sec"]
    if chunk.get("speaker_type"):
        meta["speaker_type"] = chunk["speaker_type"]
    if chunk.get("word_count") is not None:
        meta["word_count"] = chunk["word_count"]
    return {k: v for k, v in meta.items() if v is not None}


async def sync_pending_chunks_to_pgvector(
    *,
    limit: int = 100,
    source: str = "",
    dataset_id: str = "",
) -> dict:
    chunks = list_pending_chunks(limit=limit, source=source.strip(), dataset_id=dataset_id.strip())
    if not chunks:
        return {
            "status": "ok",
            "index_name": "pgvector",
            "namespace": "",
            "selected": 0,
            "vectors_upserted": 0,
            "embedding_backend": "",
            "vector_status": count_chunks_by_vector_status(),
        }

    chunk_ids = [chunk["id"] for chunk in chunks]
    try:
        vectors, embedding_backend = await embed_texts([_embedding_text(chunk) for chunk in chunks])
        
        with connect() as conn:
            for chunk_id, vector in zip(chunk_ids, vectors):
                conn.execute(
                    """
                    UPDATE chunk_store 
                    SET embedding = %s::vector, 
                        vector_status = 'indexed', 
                        vector_updated_at = NOW(),
                        vector_error = ''
                    WHERE id = %s
                    """,
                    (vector, chunk_id)
                )
        upserted = len(chunks)
    except Exception as exc:
        with connect() as conn:
            for chunk_id in chunk_ids:
                conn.execute(
                    "UPDATE chunk_store SET vector_error = %s, vector_status = 'error' WHERE id = %s",
                    (str(exc), chunk_id)
                )
        raise

    return {
        "status": "ok",
        "index_name": "pgvector",
        "namespace": "",
        "selected": len(chunks),
        "vectors_upserted": upserted,
        "embedding_backend": embedding_backend,
        "chunk_ids": chunk_ids,
        "vector_status": count_chunks_by_vector_status(),
    }

