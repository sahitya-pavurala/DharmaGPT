from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import get_settings
from core.backends.embedding import get_embedder
from core.retrieval import get_pinecone
from core.local_vector_store import healthcheck as local_vector_healthcheck
import anthropic
import httpx

router = APIRouter()
settings = get_settings()


def _has_real_key(value: str) -> bool:
    key = (value or "").strip()
    return bool(key and "..." not in key and key not in {"changeme", "placeholder"})


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    pinecone_ok = False
    anthropic_ok = False
    sarvam_ok = False
    llm_ok = False
    embedding_ok = False

    try:
        pc = get_pinecone()
        pc.list_indexes()
        pinecone_ok = True
    except Exception:
        pass

    if _has_real_key(settings.anthropic_api_key):
        try:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            client.models.list()
            anthropic_ok = True
        except Exception:
            pass

    if _has_real_key(settings.sarvam_api_key):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.sarvam.ai/health",
                    headers={"api-subscription-key": settings.sarvam_api_key},
                )
                sarvam_ok = r.status_code == 200
        except Exception:
            pass

    llm_backend = settings.llm_backend.lower()
    if llm_backend == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
                r.raise_for_status()
                models = [item.get("name") or item.get("model") for item in r.json().get("models", [])]
                llm_ok = settings.resolved_llm_model in models
        except Exception:
            llm_ok = False
    elif llm_backend == "anthropic":
        llm_ok = anthropic_ok
    elif llm_backend == "openai":
        if _has_real_key(settings.openai_api_key):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key, timeout=5)
                client.models.retrieve(settings.resolved_llm_model)
                llm_ok = True
            except Exception:
                llm_ok = False
    else:
        llm_ok = False

    try:
        vector = get_embedder().embed_query("health check")
        embedding_ok = len(vector) == settings.embedding_dims
    except Exception:
        embedding_ok = False

    vector_ok = pinecone_ok
    vector_name = settings.pinecone_index_name

    return HealthResponse(
        status="ok" if all([vector_ok, llm_ok, embedding_ok]) else "degraded",
        pinecone=pinecone_ok,
        vector_backend=settings.vector_db_backend,
        vector_store=vector_ok,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
        embedding=embedding_ok,
        anthropic=anthropic_ok,
        sarvam=sarvam_ok,
        vector_name=vector_name,
        llm_backend=settings.llm_backend,
        llm_local=settings.llm_backend.lower() == "ollama",
    )
