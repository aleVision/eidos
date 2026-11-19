"""API tests: other websites must not be able to change data in a local Eidos.

CORS stops a foreign page from *reading* responses, but a "simple" cross-origin
POST or DELETE still reaches the server. Browsers always send an Origin header
on those, so the backend rejects writes from origins it does not know.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1:9999"])
def test_writes_from_foreign_origins_are_rejected(client: TestClient, origin: str) -> None:
    response = client.post("/people", json={"name": "Mallory"}, headers={"Origin": origin})
    assert response.status_code == 403
    assert client.get("/people").json() == []


@pytest.mark.parametrize(
    "headers",
    [
        {},  # curl, scripts and tests send no Origin
        {"Origin": "http://127.0.0.1:5173"},  # the Eidos frontend
        {"Origin": "http://localhost:5173"},
        {"Origin": "http://testserver"},  # the API's own pages (e.g. /docs)
    ],
)
def test_writes_from_eidos_itself_are_allowed(client: TestClient, headers: dict[str, str]) -> None:
    assert client.post("/people", json={"name": "Ana"}, headers=headers).status_code == 201


def test_reads_are_not_affected(client: TestClient) -> None:
    assert client.get("/people", headers={"Origin": "https://evil.example"}).status_code == 200
