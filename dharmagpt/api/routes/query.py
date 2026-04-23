from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from core.rag_engine import answer
import structlog

router = APIRouter()
log = structlog.get_logger()


@router.post("/query", response_model=QueryResponse)
async def query_dharma(request: QueryRequest) -> QueryResponse:
    """
    Main RAG endpoint. Accepts a query + mode + optional conversation history.
    Returns an answer grounded in retrieved sacred text passages, with citations.
    """
    try:
        return await answer(request)
    except Exception as e:
        log.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail="The oracle encountered an error. Please try again.")
