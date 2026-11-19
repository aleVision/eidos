"""API tests: training the eigenface model and inspecting it."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from tests.images import face_like_png
from tests.test_app.enroll import BOX, enroll


def test_get_model_before_training_404(client: TestClient) -> None:
    assert client.get("/model").status_code == 404
    assert client.get("/model/mean.png").status_code == 404


def test_train_then_get_model(client: TestClient) -> None:
    enroll(client, n_people=3, n_samples=3)
    response = client.post("/model/train", json={"n_components": 4})
    assert response.status_code == 201
    trained = response.json()
    assert trained["n_people"] == 3 and trained["n_samples"] == 9
    assert trained["n_components"] == 4
    assert trained["theta_person"] > 0 and trained["theta_face"] > 0
    assert trained["explained_variance"][-1] == pytest.approx(1.0)
    assert trained["stale"] is False
    assert client.get("/model").json() == trained


def test_train_uses_default_components(client: TestClient) -> None:
    enroll(client, n_people=2, n_samples=3)
    assert client.post("/model/train", json={}).json()["n_components"] == 5  # clipped to M - 1


def test_train_with_one_person_is_400_and_writes_nothing(client: TestClient) -> None:
    enroll(client, n_people=1, n_samples=4)
    response = client.post("/model/train", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Training needs at least 2 different people"
    assert client.get("/model").status_code == 404
    assert list(settings.models_dir.iterdir()) == []


def test_model_becomes_stale_after_new_sample(client: TestClient) -> None:
    people = enroll(client, n_people=2, n_samples=3)
    client.post("/model/train", json={})
    files = {"image": ("f.png", face_like_png(1, 9), "image/png")}
    client.post(f"/people/{people[0]}/samples", files=files, data=BOX)
    assert client.get("/model").json()["stale"] is True


def test_model_becomes_stale_after_person_deleted(client: TestClient) -> None:
    people = enroll(client, n_people=3, n_samples=3)
    client.post("/model/train", json={})
    client.delete(f"/people/{people[2]}")
    assert client.get("/model").json()["stale"] is True


def test_patch_thresholds(client: TestClient) -> None:
    enroll(client, n_people=2, n_samples=3)
    client.post("/model/train", json={})
    response = client.patch("/model/thresholds", json={"person": 12.5, "face_space": 40.0})
    assert response.status_code == 200
    assert (response.json()["theta_person"], response.json()["theta_face"]) == (12.5, 40.0)
    bad = client.patch("/model/thresholds", json={"person": 0, "face_space": 40.0})
    assert bad.status_code == 422


def test_mean_and_eigenface_png(client: TestClient) -> None:
    enroll(client, n_people=2, n_samples=3)
    client.post("/model/train", json={"n_components": 3})
    for path in ("/model/mean.png", "/model/eigenfaces/1.png", "/model/eigenfaces/3.png"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    assert client.get("/model/eigenfaces/0.png").status_code == 404
    assert client.get("/model/eigenfaces/4.png").status_code == 404
