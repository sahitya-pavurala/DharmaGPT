import uuid
import structlog
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.retrieval import retrieve, format_context
from core.prompts import get_system_prompt
from models.schemas import QueryRequest, QueryResponse, SourceChunk

log = structlog.get_logger()
settings = get_settings()

_anthropic: anthropic.AsyncAnthropic | None = None


def get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm(system: str, messages: list[dict]) -> str:
    client = get_anthropic()
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


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

    # 1. Retrieve
    chunks: list[SourceChunk] = await retrieve(
        query=request.query,
        filter_kanda=request.filter_kanda,
    )

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
