from __future__ import annotations

from pinecone import ServerlessSpec

from core.chunk_store import (
    count_chunks_by_vector_status,
    list_pending_chunks,
    mark_chunks_indexed,
    mark_chunks_vector_error,
)
from core.config import get_settings
from core.retrieval import embed_texts, get_pinecone


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


def _ensure_pinecone_index(index_name: str, *, create_index: bool) -> None:
    if not create_index:
        return
    settings = get_settings()
    pc = get_pinecone()
    names = {idx.name for idx in pc.list_indexes()}
    if index_name in names:
        return
    pc.create_index(
        name=index_name,
        dimension=settings.embedding_dims,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
    )


async def sync_pending_chunks_to_pinecone(
    *,
    limit: int = 100,
    index_name: str = "",
    namespace: str = "",
    source: str = "",
    dataset_id: str = "",
    create_index: bool = False,
) -> dict:
    settings = get_settings()
    target_index = (index_name or settings.pinecone_index_name).strip()
    target_namespace = namespace.strip()
    chunks = list_pending_chunks(limit=limit, source=source.strip(), dataset_id=dataset_id.strip())
    if not chunks:
        return {
            "status": "ok",
            "index_name": target_index,
            "namespace": target_namespace,
            "selected": 0,
            "vectors_upserted": 0,
            "embedding_backend": "",
            "vector_status": count_chunks_by_vector_status(),
        }

    chunk_ids = [chunk["id"] for chunk in chunks]
    try:
        vectors, embedding_backend = await embed_texts([_embedding_text(chunk) for chunk in chunks])
        _ensure_pinecone_index(target_index, create_index=create_index)
        index = get_pinecone().Index(target_index)
        records = [
            {
                "id": chunk["id"],
                "values": vector,
                "metadata": _metadata_for_pinecone(chunk, embedding_backend),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        batch_size = 50
        upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            if target_namespace:
                index.upsert(vectors=batch, namespace=target_namespace)
            else:
                index.upsert(vectors=batch)
            upserted += len(batch)
    except Exception as exc:
        mark_chunks_vector_error(chunk_ids, str(exc))
        raise

    mark_chunks_indexed(chunk_ids, index_name=target_index, namespace=target_namespace)
    return {
        "status": "ok",
        "index_name": target_index,
        "namespace": target_namespace,
        "selected": len(chunks),
        "vectors_upserted": upserted,
        "embedding_backend": embedding_backend,
        "chunk_ids": chunk_ids,
        "vector_status": count_chunks_by_vector_status(),
    }
