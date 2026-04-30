"""
Migrate legacy SQLite source-store data into the local PostgreSQL database.

This copies dataset registry rows and full chunk-store rows so a fresh local
Postgres instance can immediately serve the existing corpus.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from psycopg.rows import dict_row


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "dharmagpt"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

load_dotenv(PACKAGE_ROOT / ".env")

from core.config import get_settings
from core.postgres_db import ensure_schema
SQLITE_DB = REPO_ROOT / "knowledge" / "stores" / "local_vectors.sqlite3"
CHUNK_DB = REPO_ROOT / "knowledge" / "stores" / "chunk_store.sqlite3"


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _postgres_conn() -> psycopg.Connection:
    settings = get_settings()
    database_url = (settings.database_url or os.getenv("DATABASE_URL") or "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is not configured")
    conn = psycopg.connect(database_url)
    conn.row_factory = dict_row
    ensure_schema(conn)
    conn.commit()
    return conn


def _migrate_datasets(conn: psycopg.Connection) -> int:
    if not SQLITE_DB.exists():
        return 0
    with _connect_sqlite(SQLITE_DB) as src:
        rows = src.execute("SELECT name, display_name, active, vector_count, created_at FROM datasets").fetchall()

    count = 0
    with conn.transaction():
        for row in rows:
            conn.execute(
                """
                INSERT INTO datasets (name, display_name, active, vector_count, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    active = EXCLUDED.active,
                    vector_count = EXCLUDED.vector_count,
                    created_at = EXCLUDED.created_at
                """,
                (
                    row["name"],
                    row["display_name"],
                    bool(row["active"]),
                    int(row["vector_count"] or 0),
                    row["created_at"],
                ),
            )
            count += 1
    return count


def _migrate_chunks(conn: psycopg.Connection) -> int:
    if not CHUNK_DB.exists():
        return 0
    with _connect_sqlite(CHUNK_DB) as src:
        rows = src.execute("SELECT * FROM chunk_store").fetchall()

    count = 0
    with conn.transaction():
        for row in rows:
            metadata_json = row["metadata_json"] or "{}"
            conn.execute(
                """
                INSERT INTO chunk_store (
                    id, text, translated_text, source, source_title, source_type, citation,
                    section, chapter, verse, language, url, dataset_id, start_time_sec,
                    end_time_sec, speaker_type, word_count, preview, translated_preview,
                    metadata_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
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
                    created_at = EXCLUDED.created_at
                """,
                (
                    row["id"],
                    row["text"],
                    row["translated_text"] or "",
                    row["source"] or "",
                    row["source_title"] or "",
                    row["source_type"] or "text",
                    row["citation"] or "",
                    row["section"],
                    row["chapter"],
                    row["verse"],
                    row["language"] or "",
                    row["url"],
                    row["dataset_id"],
                    row["start_time_sec"],
                    row["end_time_sec"],
                    row["speaker_type"],
                    row["word_count"],
                    row["preview"],
                    row["translated_preview"],
                    metadata_json,
                    row["created_at"],
                ),
            )
            count += 1
    return count


def main() -> None:
    if not SQLITE_DB.exists() and not CHUNK_DB.exists():
        print("No SQLite stores found to migrate.")
        return

    conn = _postgres_conn()
    try:
        dataset_count = _migrate_datasets(conn)
        chunk_count = _migrate_chunks(conn)
    finally:
        conn.close()

    print(f"Migrated {dataset_count} dataset row(s) and {chunk_count} chunk row(s) to PostgreSQL.")


if __name__ == "__main__":
    main()
