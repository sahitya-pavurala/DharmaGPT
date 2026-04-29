"""
chunk_store.py — durable full-text sidecar store for Pinecone-backed chunks.

Primary backend: PostgreSQL when DATABASE_URL is configured.
Fallback backend: SQLite for legacy local-only runs.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.postgres_db import connect as pg_connect, ensure_schema as pg_ensure_schema, use_postgres


REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_DB_PATH = REPO_ROOT / "knowledge" / "stores" / "chunk_store.sqlite3"


def _sqlite_connect() -> sqlite3.Connection:
    STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STORE_DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_sqlite(conn)
    return conn


def _init_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_store (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            translated_text TEXT,
            source TEXT,
            source_title TEXT,
            source_type TEXT,
            citation TEXT,
            section TEXT,
            chapter INTEGER,
            verse INTEGER,
            language TEXT,
            url TEXT,
            dataset_id TEXT,
            start_time_sec REAL,
            end_time_sec REAL,
            speaker_type TEXT,
            word_count INTEGER,
            preview TEXT,
            translated_preview TEXT,
            metadata_json TEXT,
            vector_status TEXT NOT NULL DEFAULT 'pending',
            vector_index TEXT NOT NULL DEFAULT '',
            vector_namespace TEXT NOT NULL DEFAULT '',
            vector_error TEXT NOT NULL DEFAULT '',
            vector_updated_at TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _ensure_sqlite_column(conn, "chunk_store", "vector_status", "TEXT NOT NULL DEFAULT 'pending'")
    _ensure_sqlite_column(conn, "chunk_store", "vector_index", "TEXT NOT NULL DEFAULT ''")
    _ensure_sqlite_column(conn, "chunk_store", "vector_namespace", "TEXT NOT NULL DEFAULT ''")
    _ensure_sqlite_column(conn, "chunk_store", "vector_error", "TEXT NOT NULL DEFAULT ''")
    _ensure_sqlite_column(conn, "chunk_store", "vector_updated_at", "TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_store_vector_status
        ON chunk_store(vector_status, created_at)
        """
    )
    conn.commit()


def _ensure_sqlite_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _connect():
    if use_postgres():
        conn = pg_connect()
        pg_ensure_schema(conn)
        conn.commit()
        return conn
    return _sqlite_connect()


def upsert_chunk(
    chunk_id: str,
    *,
    text: str,
    translated_text: str = "",
    metadata: dict | None = None,
    vector_status: str = "pending",
) -> None:
    meta = metadata or {}
    chapter_raw = meta.get("chapter") or meta.get("sarga")
    verse_raw = meta.get("verse") or meta.get("verse_start")
    try:
        chapter = int(chapter_raw) if chapter_raw is not None else None
    except (TypeError, ValueError):
        chapter = None
    try:
        verse = int(verse_raw) if verse_raw is not None else None
    except (TypeError, ValueError):
        verse = None

    start_time = meta.get("start_time_sec")
    end_time = meta.get("end_time_sec")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        if use_postgres():
            conn.execute(
                """
                INSERT INTO chunk_store (
                    id, text, translated_text, source, source_title, source_type, citation,
                    section, chapter, verse, language, url, dataset_id, start_time_sec,
                    end_time_sec, speaker_type, word_count, preview, translated_preview,
                    metadata_json, vector_status, vector_error, vector_updated_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, '', NULL, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    translated_text = EXCLUDED.translated_text,
                    source = EXCLUDED.source,
                    source_title = EXCLUDED.source_title,
                    source_type = EXCLUDED.source_type,
                    citation = EXCLUDED.citation,
                    section = EXCLUDED.section,
                    chapter = EXCLUDED.chapter,
                    verse = EXCLUDED.verse,
                    language = EXCLUDED.language,
                    url = EXCLUDED.url,
                    dataset_id = EXCLUDED.dataset_id,
                    start_time_sec = EXCLUDED.start_time_sec,
                    end_time_sec = EXCLUDED.end_time_sec,
                    speaker_type = EXCLUDED.speaker_type,
                    word_count = EXCLUDED.word_count,
                    preview = EXCLUDED.preview,
                    translated_preview = EXCLUDED.translated_preview,
                    metadata_json = EXCLUDED.metadata_json,
                    vector_status = EXCLUDED.vector_status,
                    vector_error = EXCLUDED.vector_error,
                    vector_updated_at = EXCLUDED.vector_updated_at,
                    created_at = EXCLUDED.created_at
                """,
                (
                    chunk_id,
                    text,
                    translated_text or "",
                    str(meta.get("source") or ""),
                    str(meta.get("source_title") or ""),
                    str(meta.get("source_type") or "text"),
                    str(meta.get("citation") or ""),
                    str(meta.get("section") or meta.get("kanda") or "") or None,
                    chapter,
                    verse,
                    str(meta.get("language") or ""),
                    str(meta.get("url") or "") or None,
                    str(meta.get("dataset_id") or "") or None,
                    float(start_time) if start_time not in {None, ""} else None,
                    float(end_time) if end_time not in {None, ""} else None,
                    str(meta.get("speaker_type") or "") or None,
                    int(meta.get("word_count")) if meta.get("word_count") not in {None, ""} else None,
                    str(meta.get("text_preview") or text[:500]),
                    str(meta.get("translated_preview") or (translated_text[:500] if translated_text else "")),
                    json.dumps(meta, ensure_ascii=False),
                    vector_status,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunk_store (
                    id, text, translated_text, source, source_title, source_type, citation,
                    section, chapter, verse, language, url, dataset_id, start_time_sec,
                    end_time_sec, speaker_type, word_count, preview, translated_preview,
                    metadata_json, vector_status, vector_error, vector_updated_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', NULL, ?)
                """,
                (
                    chunk_id,
                    text,
                    translated_text or "",
                    str(meta.get("source") or ""),
                    str(meta.get("source_title") or ""),
                    str(meta.get("source_type") or "text"),
                    str(meta.get("citation") or ""),
                    str(meta.get("section") or meta.get("kanda") or "") or None,
                    chapter,
                    verse,
                    str(meta.get("language") or ""),
                    str(meta.get("url") or "") or None,
                    str(meta.get("dataset_id") or "") or None,
                    float(start_time) if start_time not in {None, ""} else None,
                    float(end_time) if end_time not in {None, ""} else None,
                    str(meta.get("speaker_type") or "") or None,
                    int(meta.get("word_count")) if meta.get("word_count") not in {None, ""} else None,
                    str(meta.get("text_preview") or text[:500]),
                    str(meta.get("translated_preview") or (translated_text[:500] if translated_text else "")),
                    json.dumps(meta, ensure_ascii=False),
                    vector_status,
                    now,
                ),
            )
            conn.commit()


def list_pending_chunks(limit: int = 100, *, source: str = "", dataset_id: str = "") -> list[dict]:
    safe_limit = max(1, min(int(limit or 100), 1000))
    filters = ["vector_status != 'indexed'"]
    params: list[object] = []
    placeholder = "%s" if use_postgres() else "?"
    if source:
        filters.append(f"source = {placeholder}")
        params.append(source)
    if dataset_id:
        filters.append(f"dataset_id = {placeholder}")
        params.append(dataset_id)

    query = f"""
        SELECT * FROM chunk_store
        WHERE {' AND '.join(filters)}
        ORDER BY created_at ASC
        LIMIT {safe_limit}
    """
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_chunk(row) for row in rows]


def mark_chunks_indexed(
    chunk_ids: list[str],
    *,
    index_name: str,
    namespace: str = "",
) -> None:
    ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    if not ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    placeholder = "%s" if use_postgres() else "?"
    placeholders = ",".join(placeholder for _ in ids)
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE chunk_store
            SET vector_status = {placeholder},
                vector_index = {placeholder},
                vector_namespace = {placeholder},
                vector_error = '',
                vector_updated_at = {placeholder}
            WHERE id IN ({placeholders})
            """,
            ["indexed", index_name, namespace, now, *ids],
        )
        if not use_postgres():
            conn.commit()


def mark_chunks_vector_error(chunk_ids: list[str], error: str) -> None:
    ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    if not ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    placeholder = "%s" if use_postgres() else "?"
    placeholders = ",".join(placeholder for _ in ids)
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE chunk_store
            SET vector_status = {placeholder},
                vector_error = {placeholder},
                vector_updated_at = {placeholder}
            WHERE id IN ({placeholders})
            """,
            ["error", error[:1000], now, *ids],
        )
        if not use_postgres():
            conn.commit()


def count_chunks_by_vector_status() -> dict[str, int]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT vector_status, COUNT(*) AS count FROM chunk_store GROUP BY vector_status"
        ).fetchall()
    return {str(row["vector_status"] or "unknown"): int(row["count"] or 0) for row in rows}


def fetch_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    if not ids:
        return {}

    placeholders = ",".join("%s" if use_postgres() else "?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM chunk_store WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

    result: dict[str, dict] = {}
    for row in rows:
        chunk = _row_to_chunk(row)
        result[chunk["id"]] = chunk
    return result


def _row_to_chunk(row) -> dict:
    try:
        meta = json.loads(row["metadata_json"] or "{}")
    except Exception:
        meta = {}
    return {
        "id": row["id"],
        "text": row["text"],
        "translated_text": row["translated_text"] or "",
        "source": row["source"] or meta.get("source") or "",
        "source_title": row["source_title"] or meta.get("source_title") or "",
        "source_type": row["source_type"] or meta.get("source_type") or "text",
        "citation": row["citation"] or meta.get("citation") or "",
        "section": row["section"] or meta.get("section") or meta.get("kanda") or "",
        "chapter": row["chapter"] if row["chapter"] is not None else meta.get("chapter") or meta.get("sarga"),
        "verse": row["verse"] if row["verse"] is not None else meta.get("verse") or meta.get("verse_start"),
        "language": row["language"] or meta.get("language") or "",
        "url": row["url"] or meta.get("url") or "",
        "dataset_id": row["dataset_id"] or meta.get("dataset_id") or "",
        "start_time_sec": row["start_time_sec"] if row["start_time_sec"] is not None else meta.get("start_time_sec"),
        "end_time_sec": row["end_time_sec"] if row["end_time_sec"] is not None else meta.get("end_time_sec"),
        "speaker_type": row["speaker_type"] or meta.get("speaker_type") or "",
        "word_count": row["word_count"] if row["word_count"] is not None else meta.get("word_count"),
        "preview": row["preview"] or meta.get("text_preview") or "",
        "translated_preview": row["translated_preview"] or meta.get("translated_text_preview") or "",
        "vector_status": row["vector_status"] or "pending",
        "vector_index": row["vector_index"] or "",
        "vector_namespace": row["vector_namespace"] or "",
        "vector_error": row["vector_error"] or "",
        "vector_updated_at": row["vector_updated_at"],
        "metadata": meta,
    }
