"""
LLM backend registry.

DharmaGPT keeps the product RAG flow in core.rag_engine and core.retrieval.
LangChain is used here as a narrow provider adapter for chat models, not as
the owner of orchestration or citation logic.

Default: anthropic  (LLM_BACKEND in .env)
No fallback — if the configured backend fails, the exception propagates immediately.

Supported values:
  anthropic  — Claude via Anthropic API (default)
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import structlog

log = structlog.get_logger()


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a LangChain BaseChatModel configured from env settings.
    Cached for the process lifetime.
    """
    from core.config import get_settings
    s = get_settings()
    backend = (s.llm_backend or "anthropic").lower()

    if backend != "anthropic":
        raise ValueError(
            f"Unknown LLM_BACKEND: {backend!r}. Valid value: anthropic"
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise RuntimeError(
            "langchain_anthropic is not installed — run: pip install langchain-anthropic"
        )

    model = s.resolved_llm_model
    log.info("llm_backend_loaded", backend="anthropic", model=model)
    return ChatAnthropic(
        model=model,
        anthropic_api_key=s.anthropic_api_key,
        max_tokens=1024,
        timeout=s.llm_timeout_sec,
    )


def invoke_chat_model(system: str, user: str) -> str:
    """Call the configured LangChain chat model and return plain text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content if hasattr(response, "content") else str(response)


async def ainvoke_chat_model(system: str, user: str) -> str:
    return await asyncio.to_thread(invoke_chat_model, system, user)
