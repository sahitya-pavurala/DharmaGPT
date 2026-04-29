"""
postgres_db.py — shared PostgreSQL connection helpers for local source stores.

This project keeps Pinecone for vectors and PostgreSQL for the durable source
store that holds full chunk text, dataset registry state, and richer metadata.
"""
from __future__ import annotations

import os

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
