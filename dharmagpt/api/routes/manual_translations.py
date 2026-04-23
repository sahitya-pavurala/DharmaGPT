from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Header, HTTPException
from filelock import FileLock

from core.config import get_settings
from models.schemas import (
    ManualTranslationApplyResponse,
    ManualTranslationBulkRequest,
    ManualTranslationPendingResponse,
    ManualTranslationRecord,
    ManualTranslationReviewRequest,
    ManualTranslationSingleRequest,
)

router = APIRouter()
log = structlog.get_logger()
settings = get_settings()

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (REPO_ROOT / settings.manual_translation_dataset_root).resolve()
MANUAL_TRANSLATION_AUDIT_LOG = (REPO_ROOT / settings.manual_translation_audit_log).resolve()
MANUAL_TRANSLATION_AUDIT_LOCK = FileLock(str(MANUAL_TRANSLATION_AUDIT_LOG) + ".lock")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_api_key(x_api_key: str | None) -> None:
    expected = (settings.manual_translation_api_key or "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _allowed_dataset_ids() -> set[str]:
    raw = (settings.manual_translation_allowed_datasets or "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _resolve_dataset_path(dataset_id: str) -> Path:
    if not dataset_id or not dataset_id.strip():
        raise HTTPException(status_code=400, detail="dataset_id is required")

    allowed = _allowed_dataset_ids()
    if allowed and dataset_id not in allowed:
        raise HTTPException(status_code=403, detail="dataset_id is not allowed")

    candidate = (DATASET_ROOT / f"{dataset_id}.jsonl").resolve()
    try:
        candidate.relative_to(DATASET_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="dataset_id resolved outside dataset root") from exc

    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return candidate


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"Invalid JSONL at line {i}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _append_audit(event: dict) -> None:
    MANUAL_TRANSLATION_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MANUAL_TRANSLATION_AUDIT_LOCK:
        with open(MANUAL_TRANSLATION_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _chunk_index(record: dict) -> int | None:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        idx = metadata.get("chunk_index")
        if isinstance(idx, int):
            return idx

    idx = record.get("chunk_index")
    if isinstance(idx, int):
        return idx
    return None


def _apply_manual_text(
    record: dict,
    manual_text: str,
    *,
    review_status: str,
    reviewer: str | None,
    review_note: str | None,
) -> dict:
    previous = {
        "text_en_manual": record.get("text_en_manual"),
        "review_status": record.get("metadata", {}).get("review_status") if isinstance(record.get("metadata"), dict) else None,
        "reviewer": record.get("metadata", {}).get("reviewer") if isinstance(record.get("metadata"), dict) else None,
    }

    record["text_en_manual"] = manual_text
    if "text_en_model" not in record and isinstance(record.get("text_en"), str):
        record["text_en_model"] = record["text_en"]

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata

    metadata["has_manual_translation"] = True
    metadata["word_count_en_manual"] = len(manual_text.split())
    metadata["review_status"] = review_status
    metadata["reviewer"] = reviewer
    metadata["review_note"] = review_note
    metadata["reviewed_at"] = _timestamp()

    return previous


def _set_review_state(
    record: dict,
    *,
    review_status: str,
    reviewer: str | None,
    review_note: str | None,
) -> dict:
    previous = {
        "review_status": record.get("metadata", {}).get("review_status") if isinstance(record.get("metadata"), dict) else None,
        "reviewer": record.get("metadata", {}).get("reviewer") if isinstance(record.get("metadata"), dict) else None,
        "review_note": record.get("metadata", {}).get("review_note") if isinstance(record.get("metadata"), dict) else None,
    }

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata

    metadata["review_status"] = review_status
    metadata["reviewer"] = reviewer
    metadata["review_note"] = review_note
    metadata["reviewed_at"] = _timestamp()
    return previous


def _apply_updates(
    records: list[dict],
    updates: dict[int, dict],
) -> int:
    updated = 0
    for record in records:
        idx = _chunk_index(record)
        if idx is None or idx not in updates:
            continue
        payload = updates[idx]
        if payload.get("kind") == "review":
            previous = _set_review_state(
                record,
                review_status=payload["review_status"],
                reviewer=payload.get("reviewer"),
                review_note=payload.get("review_note"),
            )
        else:
            previous = _apply_manual_text(
                record,
                payload["text_en_manual"].strip(),
                review_status=payload.get("review_status", "pending"),
                reviewer=payload.get("reviewer"),
                review_note=payload.get("review_note"),
            )

        _append_audit(
            {
                "timestamp": _timestamp(),
                "dataset_id": payload["dataset_id"],
                "chunk_index": idx,
                "kind": payload.get("kind", "submit"),
                "actor": payload.get("reviewer"),
                "before": previous,
                "after": {
                    "text_en_manual": record.get("text_en_manual"),
                    "review_status": record.get("metadata", {}).get("review_status") if isinstance(record.get("metadata"), dict) else None,
                    "reviewer": record.get("metadata", {}).get("reviewer") if isinstance(record.get("metadata"), dict) else None,
                    "review_note": record.get("metadata", {}).get("review_note") if isinstance(record.get("metadata"), dict) else None,
                },
            }
        )
        updated += 1
    return updated


def _pending_records(records: list[dict]) -> list[ManualTranslationRecord]:
    pending: list[ManualTranslationRecord] = []
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if metadata.get("review_status") in {None, "", "pending", "needs_work"}:
            idx = _chunk_index(record)
            if idx is None:
                continue
            pending.append(
                ManualTranslationRecord(
                    chunk_index=idx,
                    text_te=record.get("text_te") or record.get("text") or None,
                    text_en_model=record.get("text_en_model") or record.get("text_en"),
                    text_en_manual=record.get("text_en_manual"),
                    review_status=metadata.get("review_status"),
                    reviewer=metadata.get("reviewer"),
                    review_note=metadata.get("review_note"),
                    reviewed_at=metadata.get("reviewed_at"),
                )
            )
    return pending


@router.post("/manual-translations/chunk", response_model=ManualTranslationApplyResponse)
async def apply_manual_translation_chunk(
    request: ManualTranslationSingleRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ManualTranslationApplyResponse:
    _require_api_key(x_api_key)
    path = _resolve_dataset_path(request.dataset_id)
    lock = FileLock(str(path) + ".lock")

    with lock:
        records = _read_records(path)
        updated = _apply_updates(
            records,
            {
                request.chunk_index: {
                    "kind": "submit",
                    "dataset_id": request.dataset_id,
                    "text_en_manual": request.text_en_manual,
                    "review_status": request.review_status,
                    "reviewer": request.reviewer,
                    "review_note": request.review_note,
                }
            },
        )
        if updated == 0:
            raise HTTPException(status_code=404, detail=f"No record found for chunk_index={request.chunk_index}")

        _write_records(path, records)
    log.info("manual_translation_chunk_applied", dataset_id=request.dataset_id, file=str(path), chunk_index=request.chunk_index)
    return ManualTranslationApplyResponse(
        status="ok",
        dataset_id=request.dataset_id,
        file_path=str(path),
        updated_chunks=updated,
        total_chunks=len(records),
    )


@router.post("/manual-translations/bulk", response_model=ManualTranslationApplyResponse)
async def apply_manual_translation_bulk(
    request: ManualTranslationBulkRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ManualTranslationApplyResponse:
    _require_api_key(x_api_key)
    path = _resolve_dataset_path(request.dataset_id)
    if not request.translations:
        raise HTTPException(status_code=400, detail="translations list is empty")
    lock = FileLock(str(path) + ".lock")

    with lock:
        records = _read_records(path)
        update_map = {
            item.chunk_index: {
                "kind": "submit",
                "dataset_id": request.dataset_id,
                "text_en_manual": item.text_en_manual,
                "review_status": item.review_status,
                "reviewer": None,
                "review_note": None,
            }
            for item in request.translations
        }
        updated = _apply_updates(records, update_map)

        _write_records(path, records)
    log.info("manual_translation_bulk_applied", dataset_id=request.dataset_id, file=str(path), updated=updated)
    return ManualTranslationApplyResponse(
        status="ok",
        dataset_id=request.dataset_id,
        file_path=str(path),
        updated_chunks=updated,
        total_chunks=len(records),
    )


@router.post("/manual-translations/review", response_model=ManualTranslationApplyResponse)
async def review_manual_translation(
    request: ManualTranslationReviewRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ManualTranslationApplyResponse:
    _require_api_key(x_api_key)
    path = _resolve_dataset_path(request.dataset_id)
    lock = FileLock(str(path) + ".lock")

    with lock:
        records = _read_records(path)
        payload = {
            request.chunk_index: {
                "kind": "review",
                "dataset_id": request.dataset_id,
                "review_status": request.review_status,
                "reviewer": request.reviewer,
                "review_note": request.review_note,
            }
        }
        updated = _apply_updates(records, payload)
        if updated == 0:
            raise HTTPException(status_code=404, detail=f"No record found for chunk_index={request.chunk_index}")

        _write_records(path, records)
    log.info(
        "manual_translation_reviewed",
        dataset_id=request.dataset_id,
        file=str(path),
        chunk_index=request.chunk_index,
        review_status=request.review_status,
    )
    return ManualTranslationApplyResponse(
        status="ok",
        dataset_id=request.dataset_id,
        file_path=str(path),
        updated_chunks=updated,
        total_chunks=len(records),
    )


@router.get("/manual-translations/datasets/{dataset_id}/pending", response_model=ManualTranslationPendingResponse)
async def list_pending_manual_translations(
    dataset_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ManualTranslationPendingResponse:
    _require_api_key(x_api_key)
    path = _resolve_dataset_path(dataset_id)
    lock = FileLock(str(path) + ".lock")
    with lock:
        records = _read_records(path)
    pending = _pending_records(records)
    return ManualTranslationPendingResponse(
        dataset_id=dataset_id,
        total_chunks=len(records),
        pending_chunks=pending,
    )
