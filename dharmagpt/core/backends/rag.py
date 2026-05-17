"""
RAG chain registry — LangChain LCEL pipeline.

Default: local  (SQLite vector store, local_hash embeddings — no external API)
Pluggable via RAG_BACKEND env var:
  local        — SQLite cosine store + configured embedder  (default)
  pinecone     — Pinecone serverless + configured embedder

The chain built here is an LCEL Runnable:
    chain.invoke({"query": "...", "mode": "guidance", "context_override": ""})
    -> {"answer": str, "source_documents": List[Document]}

audio_chunker.py and rag_engine.py both use this chain, so swapping the
vector store or LLM is a single .env change.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog

log = structlog.get_logger()


# ── LangChain VectorStore wrapper for our SQLite local store ─────────────────

class DharmaLocalVectorStore:
    """
    Thin LangChain-compatible wrapper around core.local_vector_store.
    Implements the retriever interface used by the LCEL chain.
    """

    def __init__(self, embedder, settings):
        self._embedder = embedder
        self._settings = settings

    def as_retriever(self, search_kwargs: dict | None = None):
        return _DharmaRetriever(
            store=self,
            top_k=(search_kwargs or {}).get("k", self._settings.rag_top_k),
            min_score=self._settings.rag_min_score,
            settings=self._settings,
        )


class _DharmaRetriever:
    def __init__(self, store: DharmaLocalVectorStore, top_k: int, min_score: float, settings):
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._settings = settings

    def get_relevant_documents(self, query: str, *, filter_section: str | None = None,
                               filter_source_type: str | None = None):
        from langchain_core.documents import Document
        from core.local_vector_store import query_vectors

        vector = self._store._embedder.embed_query(query)
        matches = query_vectors(
            vector=vector,
            top_k=self._top_k,
            min_score=self._min_score,
            index_name=self._settings.local_vector_index_name,
            namespace=self._settings.local_vector_namespace,
            filter_section=filter_section,
            filter_source_type=filter_source_type,
        )
        docs = []
        for m in matches:
            meta = m.get("metadata") or {}
            text = (meta.get("text") or meta.get("text_preview") or "").strip()
            docs.append(Document(page_content=text, metadata={**meta, "score": m.get("score", 0.0)}))
        return docs

    async def aget_relevant_documents(self, query: str, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.get_relevant_documents, query, **kwargs)


# ── Pinecone retriever ────────────────────────────────────────────────────────

class _PineconeRetriever:
    def __init__(self, embedder, settings, top_k: int):
        self._embedder = embedder
        self._settings = settings
        self._top_k = top_k

    def get_relevant_documents(self, query: str, *, filter_section: str | None = None,
                               filter_source_type: str | None = None):
        from langchain_core.documents import Document
        from pinecone import Pinecone

        vector = self._embedder.embed_query(query)
        pc = Pinecone(api_key=self._settings.pinecone_api_key)
        index = pc.Index(self._settings.pinecone_index_name)

        pf: dict = {}
        if filter_section:
            pf["kanda"] = {"$eq": filter_section}
        if filter_source_type:
            pf["source_type"] = {"$eq": filter_source_type}

        results = index.query(
            vector=vector,
            top_k=self._top_k,
            include_metadata=True,
            filter=pf if pf else None,
        )
        docs = []
        for match in results.matches:
            meta = match.metadata or {}
            text = (meta.get("text") or meta.get("text_preview") or "").strip()
            docs.append(Document(page_content=text, metadata={**meta, "score": match.score}))
        return docs

    async def aget_relevant_documents(self, query: str, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.get_relevant_documents, query, **kwargs)


# ── PgVector retriever ────────────────────────────────────────────────────────

class _PgVectorRetriever:
    def __init__(self, embedder, top_k: int):
        self._embedder = embedder
        self._top_k = top_k

    def get_relevant_documents(self, query: str, *, filter_section: str | None = None,
                               filter_source_type: str | None = None):
        from langchain_core.documents import Document
        from core.postgres_db import query_similar_chunks

        vector = self._embedder.embed_query(query)
        matches = query_similar_chunks(
            vector=vector,
            top_k=self._top_k,
            filter_section=filter_section,
            filter_source_type=filter_source_type,
        )
        docs = []
        for match in matches:
            text = (match.get("text") or "").strip()
            # Combine dicts carefully, avoid passing the huge raw embedding back
            meta = match.get("metadata_json") or {}
            meta.update({
                "score": match.get("score", 0.0),
                "source": match.get("source"),
                "source_type": match.get("source_type"),
                "citation": match.get("citation"),
                "section": match.get("section"),
                "chapter": match.get("chapter"),
                "verse": match.get("verse"),
                "start_time_sec": match.get("start_time_sec"),
                "end_time_sec": match.get("end_time_sec")
            })
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    async def aget_relevant_documents(self, query: str, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.get_relevant_documents, query, **kwargs)



# ── LCEL RAG chain ────────────────────────────────────────────────────────────

def _build_chain(retriever, llm, prompts_module):
    """
    Build a simple LCEL chain:
        retrieve → format context → inject into system prompt → call LLM → extract text
    """
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.messages import SystemMessage, HumanMessage

    def retrieve_and_format(inputs: dict) -> dict:
        query = inputs["query"]
        mode = inputs.get("mode", "guidance")
        filter_section = inputs.get("filter_section")

        docs = retriever.get_relevant_documents(query, filter_section=filter_section)

        # Build context string the same way existing format_context() does
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            citation = meta.get("citation", "")
            chapter = meta.get("chapter") or meta.get("sarga")
            verse = meta.get("verse")
            extras = []
            if chapter and str(chapter) not in citation:
                extras.append(f"Ch. {chapter}")
            if verse and str(verse) not in citation:
                extras.append(f"V. {verse}")
            full_citation = f"{citation}, {', '.join(extras)}".strip(", ") if extras else citation
            src_header = f"[PASSAGE {i} — {full_citation}]"
            if meta.get("source_type") == "audio" and meta.get("start_time_sec"):
                src_header += f" [Audio @ {meta['start_time_sec']}s–{meta.get('end_time_sec', '')}s]"
            parts.append(f"{src_header}\n{doc.page_content}")

        context = "\n\n".join(parts)
        system_prompt = prompts_module.get_system_prompt(mode, context)
        return {"system": system_prompt, "query": query, "docs": docs}

    def call_llm(inputs: dict) -> dict:
        messages = [
            SystemMessage(content=inputs["system"]),
            HumanMessage(content=inputs["query"]),
        ]
        response = llm.invoke(messages)
        return {
            "answer": response.content if hasattr(response, "content") else str(response),
            "source_documents": inputs["docs"],
        }

    chain = RunnableLambda(retrieve_and_format) | RunnableLambda(call_llm)
    return chain


# ── Registry ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_chain():
    """
    Returns the configured RAG chain, cached for the process lifetime.
    Respects RAG_BACKEND env var (default: local).

    Usage:
        chain = get_rag_chain()
        result = chain.invoke({"query": "...", "mode": "guidance"})
        result["answer"]            # str
        result["source_documents"]  # List[Document]
    """
    from core.config import get_settings
    from core.backends.embedding import get_embedder
    from core.backends.llm import get_llm
    import core.prompts as prompts_module

    s = get_settings()
    backend = (s.rag_backend or "local").lower()
    embedder = get_embedder()
    llm = get_llm()

    if backend == "local":
        store = DharmaLocalVectorStore(embedder=embedder, settings=s)
        retriever = store.as_retriever(search_kwargs={"k": s.rag_top_k})
        log.info("rag_chain_built", backend="local", embedder=type(embedder).__name__)

    elif backend == "pinecone":
        retriever = _PineconeRetriever(embedder=embedder, settings=s, top_k=s.rag_top_k)
        log.info("rag_chain_built", backend="pinecone", embedder=type(embedder).__name__)

    elif backend == "pgvector":
        retriever = _PgVectorRetriever(embedder=embedder, top_k=s.rag_top_k)
        log.info("rag_chain_built", backend="pgvector", embedder=type(embedder).__name__)

    else:
        raise ValueError(
            f"Unknown RAG_BACKEND: {backend!r}. Valid values: local | pinecone | pgvector"
        )

    return _build_chain(retriever, llm, prompts_module)
