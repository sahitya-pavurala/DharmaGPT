"""
retrieval.py — vector retrieval, delegating all embedding to the backend registry.

Embedding backend: EMBEDDING_BACKEND in .env (default: openai)
Vector DB backend: RAG_BACKEND / VECTOR_DB_BACKEND in .env (default: local)
"""
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.dataset_store import any_registered, get_active_names
from core.local_vector_store import query_vectors
from core.backends.embedding import get_embedder
from models.schemas import SourceChunk

log = structlog.get_logger()
settings = get_settings()


def get_pinecone():
    from pinecone import Pinecone
    return Pinecone(api_key=settings.pinecone_api_key)


def embed_text_local(text: str, dims: int | None = None) -> list[float]:
    """Backward compat — delegates to registry embedder."""
    return get_embedder().embed_query(text)


def embed_texts_local(texts: list[str], dims: int | None = None) -> list[list[float]]:
    """Backward compat — delegates to registry embedder."""
    return get_embedder().embed_documents(texts)


def use_local_hash_embeddings() -> bool:
    return settings.embedding_backend.lower() == "local_hash"


async def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    """Embed a batch of texts. Returns (vectors, backend_name)."""
    import asyncio
    embedder = get_embedder()
    vectors = await asyncio.to_thread(embedder.embed_documents, texts)
    backend_name = type(embedder).__name__.lower().replace("embeddings", "")
    return vectors, backend_name


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def embed_query(text: str) -> list[float]:
    import asyncio
    return await asyncio.to_thread(get_embedder().embed_query, text)


def _source_text_from_metadata(meta: dict) -> str:
    base_text = (meta.get("text") or "").strip()
    translated_text = (
        meta.get("translated_text") or meta.get("text_en") or meta.get("text_en_model") or ""
    ).strip()
    preview = (meta.get("text_preview") or "").strip()
    language = (meta.get("language") or "").strip().lower()

    if base_text and translated_text and translated_text != base_text:
        if (language and language != "en") or meta.get("source_type") == "audio":
            return f"{base_text}\n\nEnglish translation:\n{translated_text}"
        return base_text

    return base_text or translated_text or preview


async def retrieve(
    query: str,
    top_k: int | None = None,
    filter_section: str | None = None,
    filter_source_type: str | None = None,
) -> list[SourceChunk]:
    top_k = top_k or settings.rag_top_k

    if any_registered() and not get_active_names():
        return []

    vector = await embed_query(query)
    backend = (settings.rag_backend or settings.vector_db_backend or "local").lower()
    matches: list[dict]

    if backend == "local":
        matches = query_vectors(
            vector=vector,
            top_k=top_k,
            min_score=settings.rag_min_score,
            index_name=settings.local_vector_index_name,
            namespace=settings.local_vector_namespace,
            filter_section=filter_section,
            filter_source_type=filter_source_type,
        )
    elif backend == "pgvector":
        from core.postgres_db import query_similar_chunks
        rows = query_similar_chunks(
            vector=vector,
            top_k=top_k,
            filter_section=filter_section,
            filter_source_type=filter_source_type,
        )
        matches = []
        for r in rows:
            meta = dict(r.get("metadata_json") or {})
            meta.update({
                "text_preview": r.get("preview") or r.get("text"),
                "text": r.get("text"),
                "text_en": r.get("text_en"),
                "text_en_model": r.get("text_en_model"),
                "citation": r.get("citation"),
                "section": r.get("section"),
                "kanda": r.get("section"),
                "chapter": r.get("chapter"),
                "sarga": r.get("chapter"),
                "verse": r.get("verse"),
                "source": r.get("source"),
                "source_type": r.get("source_type"),
                "start_time_sec": r.get("start_time_sec"),
                "end_time_sec": r.get("end_time_sec"),
                "url": r.get("url"),
            })
            matches.append({
                "score": r.get("score", 0.0),
                "metadata": meta
            })
    else:
        pc = get_pinecone()
        index = pc.Index(settings.pinecone_index_name)
        pf: dict = {}
        if filter_section:
            pf["kanda"] = {"$eq": filter_section}
        if filter_source_type:
            pf["source_type"] = {"$eq": filter_source_type}
        if any_registered():
            pf["dataset_id"] = {"$in": get_active_names()}
        results = index.query(
            vector=vector, top_k=top_k, include_metadata=True, filter=pf if pf else None,
        )
        matches = [{"score": m.score, "metadata": m.metadata or {}} for m in results.matches]

    chunks: list[SourceChunk] = []
    for match in matches:
        score = float(match.get("score") or 0.0)
        if score < settings.rag_min_score:
            continue
        meta = match.get("metadata") or {}
        section = meta.get("section") or meta.get("kanda") or None
        chapter_raw = meta.get("chapter") or meta.get("sarga")
        verse_raw = meta.get("verse")
        chunks.append(SourceChunk(
            text=_source_text_from_metadata(meta),
            citation=meta.get("citation", ""),
            section=section,
            chapter=int(chapter_raw) if chapter_raw is not None else None,
            verse=int(verse_raw) if verse_raw is not None else None,
            score=round(score, 4),
            source_type=meta.get("source_type", "text"),
            audio_timestamp=(
                f"{meta.get('start_time_sec', '')}s-{meta.get('end_time_sec', '')}s"
                if meta.get("source_type") == "audio" else None
            ),
            url=meta.get("url"),
        ))

    log.info("retrieval_done", backend=backend, query=query[:60], results=len(chunks))
    return chunks


def _full_citation(chunk: SourceChunk) -> str:
    base = chunk.citation or ""
    extras: list[str] = []
    if chunk.chapter is not None and str(chunk.chapter) not in base:
        extras.append(f"Ch. {chunk.chapter}")
    if chunk.verse is not None and str(chunk.verse) not in base:
        extras.append(f"V. {chunk.verse}")
    if extras:
        return f"{base}, {', '.join(extras)}".strip(", ")
    return base


def format_context(chunks: list[SourceChunk], max_chars: int | None = None) -> str:
    max_chars = max_chars or settings.max_context_chars
    parts, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        src = f"[PASSAGE {i} - {_full_citation(chunk)}]"
        if chunk.source_type == "audio" and chunk.audio_timestamp:
            src += f" [Audio @ {chunk.audio_timestamp}]"
        block = f"{src}\n{chunk.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
