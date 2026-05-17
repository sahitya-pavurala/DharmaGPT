"""
RAG engine.

The query path is intentionally explicit:
    retrieve sources -> format cited context -> build prompt -> call chat model.

LangChain is used only inside core.backends.llm as a provider adapter. Retrieval,
prompting, citation enrichment, and response shaping stay in DharmaGPT code.
"""
from __future__ import annotations

import uuid

import structlog

from core.backends.llm import ainvoke_chat_model
from core.retrieval import retrieve as _retrieve_sources, format_context
from models.schemas import QueryRequest, QueryResponse, SourceChunk

log = structlog.get_logger()


async def retrieve(
    query: str,
    top_k: int | None = None,
    filter_section: str | None = None,
    filter_source_type: str | None = None,
) -> list[SourceChunk]:
    return await _retrieve_sources(
        query,
        top_k=top_k,
        filter_section=filter_section,
        filter_source_type=filter_source_type,
    )


async def _call_llm(system: str, messages: list[dict]) -> str:
    query = messages[-1]["content"] if messages else ""
    return await ainvoke_chat_model(system, query)


async def answer(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline using DharmaGPT retrieval and a LangChain-backed chat model.
    Falls back gracefully to LLM-only if retrieval fails.
    """
    log.info("rag_query", mode=request.mode, query=request.query[:80])

    try:
        chunks = await retrieve(
            request.query,
            filter_section=request.filter_section,
        )
        from core.prompts import get_system_prompt

        system_prompt = get_system_prompt(request.mode.value, format_context(chunks))
        answer_text = await _call_llm(system_prompt, [{"role": "user", "content": request.query}])

    except Exception as exc:
        log.warning("rag_pipeline_failed_fallback", error=str(exc))
        from core.prompts import get_system_prompt

        system = get_system_prompt(request.mode.value, "")
        try:
            answer_text = await ainvoke_chat_model(system, request.query)
        except Exception as llm_exc:
            log.error("llm_fallback_also_failed", error=str(llm_exc))
            answer_text = "I encountered an error while processing your query. Please try again."
        chunks = []

    log.info("rag_answer_done", chars=len(answer_text), sources=len(chunks))

    return QueryResponse(
        answer=answer_text,
        sources=chunks,
        mode=request.mode,
        language=request.language,
        query_id=str(uuid.uuid4()),
    )
