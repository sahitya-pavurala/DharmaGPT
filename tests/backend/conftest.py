from __future__ import annotations

import sys
from pathlib import Path

import pytest

DHARMAGPT_DIR = Path(__file__).resolve().parents[2] / "dharmagpt"
if str(DHARMAGPT_DIR) not in sys.path:
    sys.path.insert(0, str(DHARMAGPT_DIR))


@pytest.fixture(autouse=True)
def isolated_local_stores(tmp_path, monkeypatch):
    """Keep backend tests independent from the live dev database."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "database_url", "")

    import core.dataset_store as dataset_store
    import core.chunk_store as chunk_store
    import core.insight_store as insight_store

    monkeypatch.setattr(dataset_store, "_DB_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(chunk_store, "STORE_DB_PATH", tmp_path / "chunk_store.sqlite3")
    monkeypatch.setattr(insight_store, "STORE_DB_PATH", tmp_path / "insight_store.sqlite3")
