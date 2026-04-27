import structlog
import hashlib
import math
import re
from pinecone import Pinecone
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import get_settings
from core.dataset_store import any_registered, get_active_names
from core.local_vector_store import query_vectors
from models.schemas import SourceChunk

log = structlog.get_logger()
settings = get_settings()

_pc: Pinecone | None = None
_openai: AsyncOpenAI | None = None
_TOKEN_RE = re.compile(r"[\w\u0900-\u0d7f]+", re.UNICODE)


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai


def embed_text_local(text: str, dims: int | None = None) -> list[float]:
    """Deterministic local embedding used when cloud embeddings are unavailable."""
    dims = dims or settings.embedding_dims
    vector = [0.0] * dims
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = [text.lower()]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def embed_texts_local(texts: list[str], dims: int | None = None) -> list[list[float]]:
    return [embed_text_local(text, dims=dims) for text in texts]


def _source_text_from_metadata(meta: dict) -> str:
    """
    Choose the richest text available for retrieval context.

    Prefer the full stored chunk text, then any translated text, and only fall
    back to the preview when necessary.
    """
    base_text = (meta.get("text") or "").strip()
    translated_text = (meta.get("translated_text") or meta.get("text_en") or meta.get("text_en_model") or "").strip()
    preview = (meta.get("text_preview") or "").strip()
    language = (meta.get("language") or "").strip().lower()

    if base_text and translated_text and translated_text != base_text:
        if language and language != "en":
            return f"{base_text}\n\nEnglish translation:\n{translated_text}"
        if meta.get("source_type") == "audio":
            return f"{base_text}\n\nEnglish translation:\n{translated_text}"
        return base_text

    if base_text:
        return base_text
    if translated_text:
        return translated_text
    return preview


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def embed_query(text: str) -> list[float]:
    try:
        client = get_openai()
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=[text],
        )
        return response.data[0].embedding
    except Exception as exc:
        if settings.vector_db_backend.lower() == "local":
            log.warning("embedding_fallback_local", error=str(exc))
            return embed_text_local(text)
        raise


async def retrieve(
    query: str,
    top_k: int | None = None,
    filter_section: str | None = None,
    filter_source_type: str | None = None,
) -> list[SourceChunk]:
    """
    Embed query and retrieve top-k chunks from configured vector DB.
    Returns SourceChunk list sorted by relevance score.
    """
    top_k = top_k or settings.rag_top_k

    # Early-exit before any embedding or DB calls if all datasets are disabled
    if any_registered() and not get_active_names():
        return []

    vector = await embed_query(query)

    matches: list[dict]
    if settings.vector_db_backend.lower() == "local":
        matches = query_vectors(
            vector=vector,
            top_k=top_k,
            min_score=settings.rag_min_score,
            index_name=settings.local_vector_index_name,
            namespace=settings.local_vector_namespace,
            filter_section=filter_section,
            filter_source_type=filter_source_type,
        )
    else:
        pc = get_pinecone()
        index = pc.Index(settings.pinecone_index_name)

        # Build metadata filter — "kanda" is the legacy Pinecone field name
        pf: dict = {}
        if filter_section:
            pf["kanda"] = {"$eq": filter_section}
        if filter_source_type:
            pf["source_type"] = {"$eq": filter_source_type}

        # Dataset filter — only apply when datasets have been registered
        if any_registered():
            pf["dataset_id"] = {"$in": get_active_names()}

        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=pf if pf else None,
        )
        matches = [
            {
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in results.matches
        ]

    chunks = []
    for match in matches:
        score = float(match.get("score") or 0.0)
        if score < settings.rag_min_score:
            continue
        meta = match.get("metadata") or {}
        # "section"/"chapter"/"verse" are new names; fall back to legacy "kanda"/"sarga" keys
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
                f"{meta.get('start_time_sec', '')}s–{meta.get('end_time_sec', '')}s"
                if meta.get("source_type") == "audio" else None
            ),
            url=meta.get("url"),
        ))

    log.info("retrieval_done", backend="pinecone", query=query[:60], results=len(chunks))
    return chunks


def _full_citation(chunk: SourceChunk) -> str:
    """
    Return the richest possible citation string for a passage header.
    Appends chapter/verse to the stored citation if they aren't already present.
    """
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
    """Format retrieved chunks into a context block for the LLM prompt."""
    max_chars = max_chars or settings.max_context_chars
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        src = f"[PASSAGE {i} — {_full_citation(chunk)}]"
        if chunk.source_type == "audio" and chunk.audio_timestamp:
            src += f" [Audio @ {chunk.audio_timestamp}]"
        block = f"{src}\n{chunk.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
