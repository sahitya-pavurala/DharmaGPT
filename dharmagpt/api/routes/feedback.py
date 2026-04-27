"""
feedback.py — capture and review model response ratings.

POST   /api/v1/feedback                  — submit a rating (open, no auth)
GET    /api/v1/feedback/pending          — upvoted responses awaiting review
PATCH  /api/v1/feedback/{query_id}       — approve or reject a response
GET    /api/v1/feedback/gold             — list approved gold responses
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from api.auth import require_admin_api_key
from evaluation.gold_store import (
    load_gold_entries,
    list_pending_feedback,
    review_feedback_response,
    save_feedback_response,
)
from models.schemas import FeedbackRequest

router = APIRouter()
log = structlog.get_logger()


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    record = {
        "query_id": request.query_id,
        "query": request.query,
        "answer": request.answer,
        "mode": request.mode.value,
        "sources": [s.model_dump() for s in request.sources],
        "rating": request.rating.value,
        "note": request.note,
        "review_status": "pending",
    }
    save_feedback_response(record)
    log.info("feedback_saved", query_id=request.query_id, rating=request.rating.value)
    return {"status": "saved", "query_id": request.query_id}


@router.get("/feedback/pending")
async def list_pending(_: None = Depends(require_admin_api_key)) -> dict:
    pending = list_pending_feedback()
    return {"pending": pending, "total": len(pending)}


@router.patch("/feedback/{query_id}")
async def review_response(
    query_id: str,
    body: dict,
    _: None = Depends(require_admin_api_key),
) -> dict:
    status = body.get("review_status")
    if status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="review_status must be 'approved' or 'rejected'")

    reviewer = body.get("reviewer")
    review_note = body.get("review_note")
    gold_answer = body.get("gold_answer")
    if gold_answer is not None and not isinstance(gold_answer, str):
        raise HTTPException(status_code=400, detail="gold_answer must be a string when provided")
    if status == "approved" and gold_answer is not None and not gold_answer.strip():
        raise HTTPException(status_code=400, detail="gold_answer cannot be empty when approving")

    try:
        review_feedback_response(
            query_id,
            status,
            reviewer=reviewer,
            review_note=review_note,
            gold_answer_override=gold_answer,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if status == "approved":
        log.info("gold_response_added", query_id=query_id)

    return {"status": "updated", "query_id": query_id, "review_status": status}


@router.get("/feedback/gold")
async def list_gold(_: None = Depends(require_admin_api_key)) -> dict:
    records = load_gold_entries()
    return {"gold": records, "total": len(records)}
