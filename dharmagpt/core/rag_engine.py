import uuid
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.llm import LLMBackend, LLMConfig, generate_text_async
from core.retrieval import retrieve, format_context
from core.prompts import get_system_prompt
from models.schemas import QueryRequest, QueryResponse, SourceChunk

log = structlog.get_logger()
settings = get_settings()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm(system: str, messages: list[dict]) -> str:
    backend = LLMBackend((settings.llm_backend or "anthropic").lower())
    llm_config = LLMConfig(
        backend=backend,
        model=settings.resolved_llm_model,
        api_key=settings.llm_api_key
        or (settings.anthropic_api_key if backend == LLMBackend.anthropic else settings.openai_api_key),
        base_url=settings.llm_base_url,
        timeout_sec=settings.llm_timeout_sec,
    )
    return await generate_text_async(system, messages, llm_config)


async def answer(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from Pinecone
    2. Format context
    3. Build system prompt with context injected
    4. Call Claude with conversation history
    5. Return structured response with sources
    """
    log.info("rag_query", mode=request.mode, query=request.query[:80])

    # 1. Retrieve. If embeddings/vector retrieval are unavailable, keep the
    # beta answer path alive and let the LLM respond without retrieved sources.
    try:
        chunks: list[SourceChunk] = await retrieve(
            query=request.query,
            filter_section=request.filter_section,
        )
    except Exception as exc:
        log.warning("retrieval_unavailable", error=str(exc))
        chunks = []

    # 2. Format context
    context = format_context(chunks)

    # 3. System prompt
    system = get_system_prompt(request.mode.value, context)

    # 4. Build messages (include history for multi-turn)
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.history[-6:]  # last 6 turns max
    ]
    messages.append({"role": "user", "content": request.query})

    # 5. Call LLM
    answer_text = await _call_llm(system, messages)

    log.info("rag_answer_done", chars=len(answer_text), sources=len(chunks))

    return QueryResponse(
        answer=answer_text,
        sources=chunks,
        mode=request.mode,
        language=request.language,
        query_id=str(uuid.uuid4()),
    )
