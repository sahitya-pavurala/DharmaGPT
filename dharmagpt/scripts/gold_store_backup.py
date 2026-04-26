"""
gold_store_backup.py - back up and export the SQLite gold store.

This is a small operational helper for the local datastore used by
`evaluation.gold_store`. It can:

* create a point-in-time SQLite backup using the native sqlite3 backup API
* export the approved gold entries to JSONL for portability or migration

Usage:
    python -m dharmagpt.scripts.gold_store_backup
    python -m dharmagpt.scripts.gold_store_backup --export-jsonl
    python -m dharmagpt.scripts.gold_store_backup --backup-dir knowledge/stores/backups
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from evaluation import gold_store

DEFAULT_BACKUP_DIR = gold_store.STORE_DB_PATH.parent / "backups"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_db_path(source_db: str | Path | None) -> Path:
    return Path(source_db) if source_db is not None else gold_store.STORE_DB_PATH


def _row_to_gold(row: sqlite3.Row) -> dict:
    def _loads(value: str | None, default: object) -> object:
        if not value:
            return default
        return json.loads(value)

    return {
        "gold_id": row["gold_id"],
        "query_id": row["query_id"],
        "query": row["query"],
        "canonical_query": row["canonical_query"],
        "mode": row["mode"],
        "gold_answer": row["gold_answer"],
        "evidence": _loads(row["evidence_json"], []),
        "source_count": row["source_count"],
        "query_variants": _loads(row["query_variants_json"], []),
        "review_status": row["review_status"],
        "reviewer": row["reviewer"],
        "review_note": row["review_note"],
        "feedback_timestamp": row["feedback_timestamp"],
        "reviewed_at": row["reviewed_at"],
        "promoted_at": row["promoted_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "version": row["version"],
    }


def backup_sqlite_store(
    source_db: str | Path | None = None,
    *,
    backup_dir: str | Path | None = None,
) -> Path:
    """Create a point-in-time SQLite backup of the gold store."""
    source_path = _resolve_db_path(source_db)
    if not source_path.exists():
        raise FileNotFoundError(f"SQLite store not found: {source_path}")

    target_dir = Path(backup_dir) if backup_dir is not None else DEFAULT_BACKUP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / f"{source_path.stem}-{_timestamp_slug()}.sqlite3"
    with sqlite3.connect(str(source_path)) as source_conn, sqlite3.connect(str(target_path)) as target_conn:
        source_conn.backup(target_conn)
        target_conn.commit()
    return target_path


def export_gold_entries_jsonl(
    output_path: str | Path,
    *,
    source_db: str | Path | None = None,
) -> Path:
    """Export approved gold entries to JSONL for migration or offline review."""
    source_path = _resolve_db_path(source_db)
    if not source_path.exists():
        raise FileNotFoundError(f"SQLite store not found: {source_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(source_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM gold_entries ORDER BY promoted_at ASC").fetchall()

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_row_to_gold(row), ensure_ascii=False) + "\n")

    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up and export the local DharmaGPT gold store")
    parser.add_argument(
        "--source-db",
        type=str,
        default=str(gold_store.STORE_DB_PATH),
        help="SQLite file to back up/export (default: configured gold store)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(DEFAULT_BACKUP_DIR),
        help="Directory for timestamped SQLite backups",
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Also export approved gold entries to JSONL",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default=None,
        help="Explicit JSONL export path (default: backup dir with a timestamped filename)",
    )
    args = parser.parse_args(argv)

    backup_path = backup_sqlite_store(args.source_db, backup_dir=args.backup_dir)
    print(f"SQLite backup created: {backup_path}")

    if args.export_jsonl:
        export_path = (
            Path(args.jsonl_output)
            if args.jsonl_output
            else Path(args.backup_dir) / f"gold-entries-{_timestamp_slug()}.jsonl"
        )
        jsonl_path = export_gold_entries_jsonl(export_path, source_db=args.source_db)
        print(f"Gold export written:  {jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

