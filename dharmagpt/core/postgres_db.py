"""
postgres_db.py — shared PostgreSQL connection helpers for local source stores.

This project keeps Pinecone for vectors and PostgreSQL for the durable source
store that holds full chunk text, dataset registry state, and richer metadata.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from core.config import get_settings


settings = get_settings()


def database_url() -> str:
    return (settings.database_url or os.getenv("DATABASE_URL") or "").strip()


def use_postgres() -> bool:
    return bool(database_url())


def connect() -> psycopg.Connection:
    if not use_postgres():
        raise RuntimeError("DATABASE_URL is not configured")
    conn = psycopg.connect(database_url())
    conn.row_factory = dict_row
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            vector_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            source_title TEXT NOT NULL DEFAULT '',
            file_name TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            dataset_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            chunks INTEGER NOT NULL DEFAULT 0,
            vectors INTEGER NOT NULL DEFAULT 0,
            vector_db TEXT NOT NULL DEFAULT '',
            embedding_backend TEXT NOT NULL DEFAULT '',
            transcription_mode TEXT NOT NULL DEFAULT '',
            transcription_version TEXT NOT NULL DEFAULT '',
            translation_backend TEXT NOT NULL DEFAULT '',
            translation_version TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingestion_runs_finished_at
        ON ingestion_runs(finished_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_runs (
            query_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            llm_backend TEXT NOT NULL DEFAULT '',
            llm_model TEXT NOT NULL DEFAULT '',
            llm_attempted_backends JSONB NOT NULL DEFAULT '[]'::jsonb,
            llm_fallback_reason TEXT NOT NULL DEFAULT '',
            source_count INTEGER NOT NULL DEFAULT 0,
            rating TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_query_runs_created_at
        ON query_runs(created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_store (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            translated_text TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            source_title TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'text',
            citation TEXT NOT NULL DEFAULT '',
            section TEXT,
            chapter INTEGER,
            verse INTEGER,
            language TEXT NOT NULL DEFAULT '',
            url TEXT,
            dataset_id TEXT,
            start_time_sec DOUBLE PRECISION,
            end_time_sec DOUBLE PRECISION,
            speaker_type TEXT,
            word_count INTEGER,
            preview TEXT,
            translated_preview TEXT,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            vector_status TEXT NOT NULL DEFAULT 'pending',
            vector_index TEXT NOT NULL DEFAULT '',
            vector_namespace TEXT NOT NULL DEFAULT '',
            vector_error TEXT NOT NULL DEFAULT '',
            vector_updated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute("ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS vector_status TEXT NOT NULL DEFAULT 'pending'")
    conn.execute("ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS vector_index TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS vector_namespace TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS vector_error TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS vector_updated_at TIMESTAMPTZ")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_store_vector_status
        ON chunk_store(vector_status, created_at)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_documents (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            source_title TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'text',
            language TEXT NOT NULL DEFAULT 'en',
            section TEXT,
            author TEXT NOT NULL DEFAULT '',
            translator TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            dataset_id TEXT,
            page_count INTEGER,
            word_count INTEGER,
            chunks_created INTEGER NOT NULL DEFAULT 0,
            vectors_upserted INTEGER NOT NULL DEFAULT 0,
            embedding_backend TEXT NOT NULL DEFAULT '',
            vector_db TEXT NOT NULL DEFAULT '',
            index_name TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            ingested_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_documents_source
        ON source_documents(source)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_documents_ingested_at
        ON source_documents(ingested_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discourse_translations (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_title TEXT NOT NULL DEFAULT '',
            chunk_index INTEGER,
            vector_chunk_id TEXT,
            original_text TEXT NOT NULL,
            original_language TEXT NOT NULL DEFAULT 'te',
            translated_text TEXT NOT NULL,
            translated_language TEXT NOT NULL DEFAULT 'en',
            translator_name TEXT NOT NULL DEFAULT '',
            section TEXT,
            start_time_sec DOUBLE PRECISION,
            end_time_sec DOUBLE PRECISION,
            notes TEXT NOT NULL DEFAULT '',
            verified BOOLEAN NOT NULL DEFAULT FALSE,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_discourse_translations_source
        ON discourse_translations(source)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_discourse_translations_chunk_id
        ON discourse_translations(vector_chunk_id)
        WHERE vector_chunk_id IS NOT NULL
        """
    )

    conn.execute(
        f"ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS embedding vector({settings.embedding_dims})"
    )


def query_similar_chunks(
    vector: list[float],
    top_k: int = 5,
    filter_section: str | None = None,
    filter_source_type: str | None = None,
) -> list[dict]:
    if not use_postgres():
        return []
    conditions: list[str] = []
    params: list = [vector]
    if filter_section:
        conditions.append("section = %s")
        params.append(filter_section)
    if filter_source_type:
        conditions.append("source_type = %s")
        params.append(filter_source_type)
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    # vector param used twice: once in SELECT for score, once in ORDER BY
    final_params = [vector] + params[1:] + [vector, top_k]
    query = f"""
        SELECT *, 1 - (embedding <=> %s::vector) AS score
        FROM chunk_store
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with connect() as conn:
        rows = conn.execute(query, final_params).fetchall()
    return [dict(row) for row in rows]


def _row_to_dict(row: dict) -> dict:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
    }


def list_discourse_translations(
    source: str | None = None,
    verified: bool | None = None,
    limit: int = 100,
) -> list[dict]:
    if not use_postgres():
        return []
    conditions: list[str] = []
    params: list = []
    if source:
        conditions.append("source = %s")
        params.append(source)
    if verified is not None:
        conditions.append("verified = %s")
        params.append(verified)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(min(limit, 500))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM discourse_translations {where} ORDER BY created_at DESC LIMIT %s",
            params,
        ).fetchall()
    return [_row_to_dict(dict(row)) for row in rows]


def create_discourse_translation(
    *,
    source: str,
    source_title: str = "",
    chunk_index: int | None = None,
    vector_chunk_id: str | None = None,
    original_text: str,
    original_language: str = "te",
    translated_text: str,
    translated_language: str = "en",
    translator_name: str = "",
    section: str | None = None,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
    notes: str = "",
    verified: bool = False,
) -> dict:
    if not use_postgres():
        raise RuntimeError("DATABASE_URL is not configured")
    now = datetime.now(timezone.utc)
    row_id = uuid.uuid4().hex
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO discourse_translations (
                id, source, source_title, chunk_index, vector_chunk_id,
                original_text, original_language, translated_text, translated_language,
                translator_name, section, start_time_sec, end_time_sec,
                notes, verified, metadata_json, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s::jsonb, %s, %s
            )
            """,
            (
                row_id,
                source,
                source_title,
                chunk_index,
                vector_chunk_id,
                original_text,
                original_language,
                translated_text,
                translated_language,
                translator_name,
                section,
                start_time_sec,
                end_time_sec,
                notes,
                verified,
                "{}",
                now,
                now,
            ),
        )
    return {"id": row_id, "created_at": now.isoformat()}


def update_discourse_translation(
    row_id: str,
    *,
    translated_text: str | None = None,
    verified: bool | None = None,
    notes: str | None = None,
    translator_name: str | None = None,
) -> bool:
    if not use_postgres():
        return False
    sets: list[str] = []
    params: list = []
    if translated_text is not None:
        sets.append("translated_text = %s")
        params.append(translated_text)
    if verified is not None:
        sets.append("verified = %s")
        params.append(verified)
    if notes is not None:
        sets.append("notes = %s")
        params.append(notes)
    if translator_name is not None:
        sets.append("translator_name = %s")
        params.append(translator_name)
    if not sets:
        return True
    sets.append("updated_at = %s")
    params.append(datetime.now(timezone.utc))
    params.append(row_id)
    with connect() as conn:
        result = conn.execute(
            f"UPDATE discourse_translations SET {', '.join(sets)} WHERE id = %s",
            params,
        )
    return (result.rowcount or 0) > 0


def delete_discourse_translation(row_id: str) -> bool:
    if not use_postgres():
        return False
    with connect() as conn:
        result = conn.execute(
            "DELETE FROM discourse_translations WHERE id = %s",
            (row_id,),
        )
    return (result.rowcount or 0) > 0
