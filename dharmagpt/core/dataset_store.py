"""
dataset_store.py — lightweight registry for named Pinecone datasets.

Primary backend: PostgreSQL when DATABASE_URL is configured.
Fallback backend: SQLite for legacy local-only runs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.postgres_db import connect as pg_connect, ensure_schema as pg_ensure_schema, use_postgres


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = REPO_ROOT / "knowledge" / "stores" / "datasets.sqlite3"
_DB_PATH = _DEFAULT_DB_PATH


def _sqlite_connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_sqlite(conn)
    return conn


def _init_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            name         TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            active       INTEGER NOT NULL DEFAULT 1,
            vector_count INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _connect():
    if use_postgres() and _DB_PATH == _DEFAULT_DB_PATH:
        conn = pg_connect()
        pg_ensure_schema(conn)
        conn.commit()
        return conn
    return _sqlite_connect()


def _using_postgres() -> bool:
    return use_postgres() and _DB_PATH == _DEFAULT_DB_PATH


def register(name: str, display_name: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        if _using_postgres():
            conn.execute(
                "INSERT INTO datasets (name, display_name, active, vector_count, created_at) VALUES (%s, %s, TRUE, 0, %s) ON CONFLICT (name) DO NOTHING",
                (name.strip(), (display_name or name).strip(), now),
            )
        else:
            conn.execute(
                "INSERT OR IGNORE INTO datasets (name, display_name, active, vector_count, created_at) VALUES (?, ?, 1, 0, ?)",
                (name.strip(), (display_name or name).strip(), now),
            )
            conn.commit()


def increment_count(name: str, count: int) -> None:
    with _connect() as conn:
        if _using_postgres():
            conn.execute(
                "UPDATE datasets SET vector_count = vector_count + %s WHERE name = %s",
                (count, name.strip()),
            )
        else:
            conn.execute(
                "UPDATE datasets SET vector_count = vector_count + ? WHERE name = ?",
                (count, name.strip()),
            )
            conn.commit()


def list_all() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, display_name, active, vector_count, created_at FROM datasets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def set_active(name: str, active: bool) -> bool:
    with _connect() as conn:
        if _using_postgres():
            cur = conn.execute(
                "UPDATE datasets SET active = %s WHERE name = %s",
                (active, name.strip()),
            )
        else:
            cur = conn.execute(
                "UPDATE datasets SET active = ? WHERE name = ?",
                (1 if active else 0, name.strip()),
            )
            conn.commit()
        return cur.rowcount > 0


def remove(name: str) -> bool:
    with _connect() as conn:
        if _using_postgres():
            cur = conn.execute("DELETE FROM datasets WHERE name = %s", (name.strip(),))
        else:
            cur = conn.execute("DELETE FROM datasets WHERE name = ?", (name.strip(),))
            conn.commit()
        return cur.rowcount > 0


def get_active_names() -> list[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT name FROM datasets WHERE active = TRUE" if _using_postgres() else "SELECT name FROM datasets WHERE active = 1").fetchall()
    return [r["name"] for r in rows]


def any_registered() -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM datasets").fetchone()
    return (row["cnt"] or 0) > 0
