from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import gold_store


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gold_store, "STORE_DB_PATH", tmp_path / "dharmagpt.sqlite3")


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


def test_feedback_review_promotes_gold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_store(monkeypatch, tmp_path)

    gold_store.save_feedback_response(_feedback_record("q1", "How should I deal with anger?", "Stay steady."))
    pending = gold_store.list_pending_feedback()
    assert len(pending) == 1

    updated = gold_store.review_feedback_response("q1", "approved", reviewer="reviewer@company.com")
    assert updated["review_status"] == "approved"

    gold = gold_store.load_gold_entries()
    assert len(gold) == 1
    assert gold[0]["query"] == "How should I deal with anger?"
    assert gold[0]["gold_answer"] == "Stay steady."
    assert gold[0]["version"] == 1
    assert gold_store.find_gold_answer("How should I deal with anger?", "guidance") == "Stay steady."


def test_re_approving_same_intent_bumps_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_store(monkeypatch, tmp_path)

    gold_store.save_feedback_response(_feedback_record("q1", "How should I deal with anger?", "Answer one."))
    gold_store.review_feedback_response("q1", "approved", reviewer="reviewer@company.com")

    gold_store.save_feedback_response(_feedback_record("q2", "  HOW should I deal with anger?  ", "Answer two."))
    gold_store.review_feedback_response("q2", "approved", reviewer="reviewer@company.com")

    gold = gold_store.load_gold_entries()
    assert len(gold) == 1
    assert gold[0]["version"] == 2
    assert gold[0]["query_variants"] == ["  HOW should I deal with anger?  ", "How should I deal with anger?"]
    assert gold_store.find_gold_answer("How should I deal with anger?", "guidance") == "Answer two."

