"""
conftest.py — shared pytest fixtures for all test suites.

Unit tests import from this file for lightweight helpers.
Integration tests use the fixtures in tests/integration/conftest.py
which build on top of these.
"""

import pytest
from models.schemas import QueryMode, QueryRequest, SourceChunk


@pytest.fixture(autouse=True)
def isolated_local_stores(tmp_path, monkeypatch):
    """Keep tests off the developer's live Postgres/SQLite stores."""
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


@pytest.fixture
def sample_source():
    return SourceChunk(
        text="Hanuman leapt across the ocean with the strength of devotion.",
        citation="Valmiki Ramayana, Sundara Kanda, Sarga 1",
        section="Sundara Kanda",
        chapter=1,
        score=0.88,
        source_type="text",
    )


@pytest.fixture
def sample_sources(sample_source):
    return [
        sample_source,
        SourceChunk(
            text="Rama stood firm in dharma even in exile.",
            citation="Valmiki Ramayana, Ayodhya Kanda, Sarga 20",
            section="Ayodhya Kanda",
            chapter=20,
            score=0.76,
            source_type="text",
        ),
        SourceChunk(
            text="Sita endured with grace, her faith in Rama unwavering.",
            citation="Valmiki Ramayana, Sundara Kanda, Sarga 15",
            section="Sundara Kanda",
            chapter=15,
            score=0.71,
            source_type="text",
        ),
    ]


@pytest.fixture
def guidance_request():
    return QueryRequest(
        query="How should I deal with anger and frustration in daily life?",
        mode=QueryMode.guidance,
    )


@pytest.fixture
def story_request():
    return QueryRequest(
        query="Tell me the story of Hanuman crossing the ocean to reach Lanka.",
        mode=QueryMode.story,
    )
