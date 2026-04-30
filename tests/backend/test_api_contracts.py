from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from api.routes import health as health_route
    from api.main import app

    class FakePinecone:
        def list_indexes(self):
            return []

    class FakeAnthropicModels:
        def list(self):
            return []

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.models = FakeAnthropicModels()

    class FakeEmbedder:
        def embed_query(self, text):
            return [0.0] * 3072

    async def fake_http_get(self, url, **kwargs):
        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen2.5:7b"}]}

        return Response()

    monkeypatch.setattr(health_route, "get_pinecone", lambda: FakePinecone())
    monkeypatch.setattr(health_route, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(health_route.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(health_route.httpx.AsyncClient, "get", fake_http_get)
    monkeypatch.setattr(health_route.settings, "llm_backend", "ollama")
    monkeypatch.setattr(health_route.settings, "ollama_model", "qwen2.5:7b")
    monkeypatch.setattr(health_route.settings, "admin_api_key", "test-admin-key")

    return TestClient(app)


@pytest.mark.smoke
def test_admin_page_serves_html(client):
    response = client.get("/admin")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "X-Admin-Key" in response.text
    assert "/admin/datasets" in response.text


@pytest.mark.smoke
def test_health_contract(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_backend"]
    assert body["llm_backend"] == "ollama"
    assert isinstance(body["vector_store"], bool)


@pytest.mark.contract
def test_query_validation_contract(client):
    response = client.post(
        "/api/v1/query",
        json={"query": "?", "mode": "guidance"},
        headers={"X-API-Key": "test-admin-key"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.contract
def test_feedback_admin_auth_contract(client):
    response = client.get("/api/v1/feedback/pending")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing admin API key"
