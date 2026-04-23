from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import get_settings
from core.retrieval import get_pinecone
import anthropic
import httpx

router = APIRouter()
settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    pinecone_ok = False
    anthropic_ok = False
    sarvam_ok = False

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
            r = await client.get(
                "https://api.sarvam.ai/health",
                headers={"api-subscription-key": settings.sarvam_api_key},
            )
            sarvam_ok = r.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if all([pinecone_ok, anthropic_ok]) else "degraded",
        pinecone=pinecone_ok,
        anthropic=anthropic_ok,
        sarvam=sarvam_ok,
    )
