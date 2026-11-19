"""API tests: the ORL demo dataset, against a small fake ORL tree."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.demo import get_orl_root
from tests.orl_fake import write_fake_orl


@pytest.fixture
def demo_client(client: TestClient, tmp_path: Path) -> Iterator[TestClient]:
    root = write_fake_orl(tmp_path / "orl", subjects=31, per_subject=10)
    app.dependency_overrides[get_orl_root] = lambda: root
    yield client


def test_demo_load_is_idempotent_and_removable(demo_client: TestClient) -> None:
    assert demo_client.post("/demo/orl").json() == {"people_added": 30}
    assert demo_client.post("/demo/orl").json() == {"people_added": 0}
    people = demo_client.get("/people").json()
    assert len(people) == 30
    assert all(p["is_demo"] and p["n_samples"] == 6 for p in people)
    assert people[0]["name"].startswith("ORL s")

    demo_client.post("/people", json={"name": "Me"})
    assert demo_client.delete("/demo/orl").status_code == 204
    assert [p["name"] for p in demo_client.get("/people").json()] == ["Me"]


def test_demo_test_photos_lists_held_out_and_strangers(demo_client: TestClient) -> None:
    photos = demo_client.get("/demo/orl/test-photos").json()
    enrolled = [p for p in photos if p["enrolled"]]
    strangers = [p for p in photos if not p["enrolled"]]
    assert len(enrolled) == 30 * 4 and all(p["index"] >= 7 for p in enrolled)
    assert len(strangers) == 10 and all(p["subject"] == 31 for p in strangers)

    response = demo_client.get(photos[0]["url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert demo_client.get("/demo/orl/photos/99/1.png").status_code == 404
