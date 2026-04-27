from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import get_settings
from core.retrieval import get_pinecone
from core.local_vector_store import healthcheck as local_vector_healthcheck
import anthropic
import httpx

router = APIRouter()
settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    pinecone_ok = False
    local_vector_ok = False
    anthropic_ok = False
    ollama_ok = False
    sarvam_ok = False
    active_backend = (settings.vector_db_backend or "pinecone").strip().lower()
    llm_backend = (settings.llm_backend or "anthropic").strip().lower()

    local_vector_ok = local_vector_healthcheck()

    try:
        pc = get_pinecone()
        pc.list_indexes()
        pinecone_ok = True
    except Exception:
        pass

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        client.models.list()
        anthropic_ok = True
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(settings.ollama_url.rstrip("/") + "/api/tags")
            r.raise_for_status()
            models = r.json().get("models") or []
            expected = settings.ollama_model or settings.llm_model
            ollama_ok = any((m.get("name") or "").split(":", 1)[0] == expected.split(":", 1)[0] for m in models)
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.sarvam.ai/health",
                headers={"api-subscription-key": settings.sarvam_api_key},
            )
            sarvam_ok = r.status_code == 200
    except Exception:
        pass

    vector_ok = local_vector_ok if active_backend == "local" else pinecone_ok
    vector_name = settings.local_vector_index_name if active_backend == "local" else settings.pinecone_index_name
    llm_ok = ollama_ok if llm_backend == "ollama" else anthropic_ok

    return HealthResponse(
        status="ok" if all([vector_ok, llm_ok]) else "degraded",
        pinecone=pinecone_ok,
        vector_backend=active_backend,
        vector_store=vector_ok,
        anthropic=anthropic_ok,
        sarvam=sarvam_ok,
        vector_name=vector_name,
        llm_backend=llm_backend,
        llm_local=ollama_ok,
    )
