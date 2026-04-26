"""
gold_store.py - SQLite-backed curated gold-answer storage for DharmaGPT.

This module keeps the gold store separate from live serving:

* feedback responses are persisted in a local SQLite datastore
* approved gold answers are versioned in the same datastore
* review actions are audit-logged in a dedicated audit table

The live RAG path should not depend on this module. It is used for:

* review workflow persistence
* gold set evaluation and regression tests
* optional offline prompt experiments
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_DB_PATH = REPO_ROOT / "knowledge" / "stores" / "dharmagpt.sqlite3"

_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "should",
    "tell", "the", "to", "what", "when", "where", "which", "with",
    "you", "your",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STORE_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feedback_responses (
            query_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            mode TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            rating TEXT NOT NULL,
            note TEXT,
            review_status TEXT NOT NULL,
            reviewer TEXT,
            review_note TEXT,
            timestamp TEXT NOT NULL,
            reviewed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS gold_entries (
            gold_id TEXT PRIMARY KEY,
            query_id TEXT,
            query TEXT NOT NULL,
            canonical_query TEXT NOT NULL,
            mode TEXT NOT NULL,
            gold_answer TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            query_variants_json TEXT NOT NULL,
            review_status TEXT NOT NULL,
            reviewer TEXT,
            review_note TEXT,
            feedback_timestamp TEXT,
            reviewed_at TEXT,
            promoted_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gold_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event TEXT NOT NULL,
            gold_id TEXT,
            query_id TEXT,
            mode TEXT,
            reviewer TEXT,
            version INTEGER,
            source_count INTEGER,
            payload_json TEXT
        );
        """
    )
    conn.commit()


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _tokenize(text: str) -> set[str]:
    import re

    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9']+", text or "")
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _row_to_feedback(record: sqlite3.Row) -> dict:
    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "answer": record["answer"],
        "mode": record["mode"],
        "sources": _json_loads(record["sources_json"], []),
        "rating": record["rating"],
        "note": record["note"],
        "review_status": record["review_status"],
        "reviewer": record["reviewer"],
        "review_note": record["review_note"],
        "timestamp": record["timestamp"],
        "reviewed_at": record["reviewed_at"],
    }


def _row_to_gold(record: sqlite3.Row) -> dict:
    return {
        "gold_id": record["gold_id"],
        "query_id": record["query_id"],
        "query": record["query"],
        "canonical_query": record["canonical_query"],
        "mode": record["mode"],
        "gold_answer": record["gold_answer"],
        "evidence": _json_loads(record["evidence_json"], []),
        "source_count": record["source_count"],
        "query_variants": _json_loads(record["query_variants_json"], []),
        "review_status": record["review_status"],
        "reviewer": record["reviewer"],
        "review_note": record["review_note"],
        "feedback_timestamp": record["feedback_timestamp"],
        "reviewed_at": record["reviewed_at"],
        "promoted_at": record["promoted_at"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "version": record["version"],
    }


def gold_id_for(query: str, mode: str) -> str:
    """
    Build a stable identifier for a canonical gold answer.

    We intentionally key on normalized query + mode so paraphrases can be
    clustered under the same benchmark record when the reviewer approves them.
    """
    digest = hashlib.sha1(f"{mode}|{_normalize_text(query)}".encode("utf-8")).hexdigest()
    return f"{mode}:{digest[:16]}"


def load_feedback_responses() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM feedback_responses ORDER BY timestamp ASC").fetchall()
    return [_row_to_feedback(row) for row in rows]


def save_feedback_response(record: dict) -> dict:
    """
    Persist a raw feedback response.

    The record should already include the query, answer, sources, rating, and
    review_status fields. We add a timestamp if one was not provided.
    """
    normalized = dict(record)
    normalized.setdefault("timestamp", _timestamp())
    normalized.setdefault("review_status", "pending")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback_responses (
                query_id, query, answer, mode, sources_json, rating, note,
                review_status, reviewer, review_note, timestamp, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query_id) DO UPDATE SET
                query=excluded.query,
                answer=excluded.answer,
                mode=excluded.mode,
                sources_json=excluded.sources_json,
                rating=excluded.rating,
                note=excluded.note,
                review_status=excluded.review_status,
                reviewer=excluded.reviewer,
                review_note=excluded.review_note,
                timestamp=excluded.timestamp,
                reviewed_at=excluded.reviewed_at
            """,
            (
                normalized["query_id"],
                normalized["query"],
                normalized["answer"],
                normalized["mode"],
                _json_dumps(normalized.get("sources", [])),
                normalized["rating"],
                normalized.get("note"),
                normalized["review_status"],
                normalized.get("reviewer"),
                normalized.get("review_note"),
                normalized["timestamp"],
                normalized.get("reviewed_at"),
            ),
        )
        conn.commit()
    return normalized


def list_pending_feedback() -> list[dict]:
    """
    Return upvoted responses that still need human review.

    This is the review queue that feeds the gold store.
    """
    records = load_feedback_responses()
    return [
        r for r in records
        if r.get("rating") == "up" and r.get("review_status") == "pending"
    ]


def load_gold_entries() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM gold_entries ORDER BY promoted_at ASC").fetchall()
    return [_row_to_gold(row) for row in rows]


def _audit(event: dict) -> None:
    entry = {"timestamp": _timestamp(), **event}
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gold_audit (
                timestamp, event, gold_id, query_id, mode, reviewer, version,
                source_count, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["timestamp"],
                entry.get("event"),
                entry.get("gold_id"),
                entry.get("query_id"),
                entry.get("mode"),
                entry.get("reviewer"),
                entry.get("version"),
                entry.get("source_count"),
                _json_dumps(entry),
            ),
        )
        conn.commit()


def _build_gold_entry(
    feedback_record: dict,
    *,
    reviewer: str | None,
    review_note: str | None,
    gold_id: str | None = None,
    canonical_query: str | None = None,
    version: int = 1,
    created_at: str | None = None,
) -> dict:
    query = feedback_record.get("query", "")
    mode = feedback_record.get("mode", "")
    gold_id = gold_id or gold_id_for(query, mode)
    now = _timestamp()
    sources = feedback_record.get("sources") or []
    query_variants = feedback_record.get("query_variants") or [query]

    return {
        "gold_id": gold_id,
        "query_id": feedback_record.get("query_id"),
        "query": query,
        "canonical_query": canonical_query or _normalize_text(query),
        "mode": mode,
        "gold_answer": feedback_record.get("answer", ""),
        "evidence": sources,
        "source_count": len(sources),
        "query_variants": query_variants,
        "review_status": "approved",
        "reviewer": reviewer or feedback_record.get("reviewer"),
        "review_note": review_note or feedback_record.get("review_note") or feedback_record.get("note"),
        "feedback_timestamp": feedback_record.get("timestamp"),
        "reviewed_at": feedback_record.get("reviewed_at") or now,
        "promoted_at": now,
        "created_at": created_at or feedback_record.get("timestamp") or now,
        "updated_at": now,
        "version": version,
    }


def upsert_gold_entry(
    feedback_record: dict,
    *,
    reviewer: str | None = None,
    review_note: str | None = None,
) -> dict:
    """
    Promote a reviewed feedback record into the canonical gold store.

    Gold records are deduped by `gold_id`. If a matching gold record already
    exists, its version is incremented and the new approved answer replaces the
    previous entry.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM gold_entries").fetchall()
        query = feedback_record.get("query", "")
        mode = feedback_record.get("mode", "")
        matched = None
        for existing in rows:
            existing_query = existing["query"]
            if existing["mode"] != mode:
                continue
            if _normalize_text(existing_query) == _normalize_text(query):
                matched = existing
                break
            if _overlap(query, existing_query) >= 0.85:
                matched = existing
                break

        existing_gold_id = matched["gold_id"] if matched else None
        existing_version = int(matched["version"]) if matched else 0
        existing_created_at = matched["created_at"] if matched else None
        existing_variants = set(_json_loads(matched["query_variants_json"], [])) if matched else set()
        existing_variants.add(query)

        entry = _build_gold_entry(
            feedback_record,
            reviewer=reviewer,
            review_note=review_note,
            gold_id=existing_gold_id,
            canonical_query=matched["canonical_query"] if matched else None,
            version=existing_version + 1 if matched else 1,
            created_at=existing_created_at,
        )
        entry["query_variants"] = sorted(existing_variants) if matched else [query]

        conn.execute(
            """
            INSERT INTO gold_entries (
                gold_id, query_id, query, canonical_query, mode, gold_answer,
                evidence_json, source_count, query_variants_json, review_status,
                reviewer, review_note, feedback_timestamp, reviewed_at,
                promoted_at, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gold_id) DO UPDATE SET
                query_id=excluded.query_id,
                query=excluded.query,
                canonical_query=excluded.canonical_query,
                mode=excluded.mode,
                gold_answer=excluded.gold_answer,
                evidence_json=excluded.evidence_json,
                source_count=excluded.source_count,
                query_variants_json=excluded.query_variants_json,
                review_status=excluded.review_status,
                reviewer=excluded.reviewer,
                review_note=excluded.review_note,
                feedback_timestamp=excluded.feedback_timestamp,
                reviewed_at=excluded.reviewed_at,
                promoted_at=excluded.promoted_at,
                updated_at=excluded.updated_at,
                version=excluded.version
            """,
            (
                entry["gold_id"],
                entry["query_id"],
                entry["query"],
                entry["canonical_query"],
                entry["mode"],
                entry["gold_answer"],
                _json_dumps(entry["evidence"]),
                entry["source_count"],
                _json_dumps(entry["query_variants"]),
                entry["review_status"],
                entry["reviewer"],
                entry["review_note"],
                entry["feedback_timestamp"],
                entry["reviewed_at"],
                entry["promoted_at"],
                entry["created_at"],
                entry["updated_at"],
                entry["version"],
            ),
        )
        conn.commit()

    _audit(
        {
            "event": "gold_upserted",
            "gold_id": entry["gold_id"],
            "query_id": entry.get("query_id"),
            "mode": entry.get("mode"),
            "reviewer": entry.get("reviewer"),
            "version": entry.get("version"),
            "source_count": entry.get("source_count", 0),
        }
    )
    return entry


def review_feedback_response(
    query_id: str,
    review_status: str,
    *,
    reviewer: str | None = None,
    review_note: str | None = None,
    gold_answer_override: str | None = None,
) -> dict:
    """
    Update a stored feedback response and optionally promote it into the gold set.
    """
    if review_status not in {"approved", "rejected"}:
        raise ValueError("review_status must be 'approved' or 'rejected'")

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM feedback_responses WHERE query_id = ?",
            (query_id,),
        ).fetchone()
        if not row:
            raise LookupError(f"query_id not found: {query_id!r}")

        updated = dict(_row_to_feedback(row))
        updated["review_status"] = review_status
        updated["reviewed_at"] = _timestamp()
        if review_status == "approved" and gold_answer_override is not None:
            updated["answer"] = gold_answer_override.strip()
        if reviewer is not None:
            updated["reviewer"] = reviewer
        if review_note is not None:
            updated["review_note"] = review_note

        conn.execute(
            """
            UPDATE feedback_responses
            SET answer = ?, review_status = ?, reviewer = ?, review_note = ?, reviewed_at = ?
            WHERE query_id = ?
            """,
            (
                updated["answer"],
                updated["review_status"],
                updated.get("reviewer"),
                updated.get("review_note"),
                updated.get("reviewed_at"),
                query_id,
            ),
        )
        conn.commit()

    _audit(
        {
            "event": "feedback_reviewed",
            "query_id": query_id,
            "review_status": review_status,
            "reviewer": reviewer,
            "review_note": review_note,
        }
    )

    if review_status == "approved":
        upsert_gold_entry(updated, reviewer=reviewer, review_note=review_note)

    return updated


def list_gold_examples(query: str, mode: str, n: int = 2) -> list[dict]:
    """
    Return up to n gold examples relevant to a query and mode.

    This is intentionally kept for offline analysis and prompt experiments,
    not for the live RAG serving path.
    """
    records = [
        r for r in load_gold_entries()
        if r.get("mode") == mode and r.get("gold_answer")
    ]
    if not records:
        return []
    scored = sorted(
        records,
        key=lambda r: _overlap(query, r.get("query", "")),
        reverse=True,
    )
    return [{"query": r["query"], "answer": r["gold_answer"]} for r in scored[:n]]


def find_gold_answer(query: str, mode: str) -> str | None:
    """
    Return the approved gold answer for a query/mode pair.

    Matching is exact first, then a conservative overlap fallback for
    paraphrases that were approved by the reviewer.
    """
    q_norm = _normalize_text(query)
    records = load_gold_entries()

    for record in records:
        if record.get("mode") == mode and _normalize_text(record.get("query")) == q_norm:
            return record.get("gold_answer")

    for record in records:
        if record.get("mode") == mode and _overlap(query, record.get("query", "")) >= 0.85:
            return record.get("gold_answer")

    return None

