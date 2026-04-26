from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evaluation import gold_store
from scripts.gold_store_backup import backup_sqlite_store, export_gold_entries_jsonl


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "dharmagpt.sqlite3"
    monkeypatch.setattr(gold_store, "STORE_DB_PATH", db_path)
    return db_path


def _feedback_record(query_id: str, query: str, answer: str, mode: str = "guidance") -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "mode": mode,
        "sources": [
            {
                "citation": "Bhagavad Gita, Ch. 2, V. 47",
                "section": "Bhagavad Gita",
                "chapter": 2,
                "verse": 47,
                "text": "You have a right to perform your actions...",
                "score": 0.98,
            }
        ],
        "rating": "up",
        "review_status": "pending",
        "timestamp": "2026-04-25T00:00:00Z",
    }


def test_backup_sqlite_store_and_export_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_store(monkeypatch, tmp_path)
    backup_dir = tmp_path / "backups"

    gold_store.save_feedback_response(_feedback_record("q1", "How should I deal with anger?", "Stay steady."))
    gold_store.review_feedback_response("q1", "approved", reviewer="reviewer@company.com")

    backup_path = backup_sqlite_store(backup_dir=backup_dir)
    assert backup_path.exists()
    assert backup_path.parent == backup_dir

    with sqlite3.connect(str(backup_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM gold_entries").fetchone()[0]
    assert count == 1

    export_path = export_gold_entries_jsonl(tmp_path / "gold-export.jsonl")
    assert export_path.exists()
    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["query"] == "How should I deal with anger?"
    assert payload["gold_answer"] == "Stay steady."
