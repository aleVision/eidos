"""Integration test for the API.

Uses FastAPI's TestClient, which runs requests against the real app in-process —
no running server needed. This is the "the whole pipe works" check.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Eidos"
    assert "version" in body
