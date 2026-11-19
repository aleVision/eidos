"""API tests: recognition (four outcomes) and reconstruction."""

from fastapi.testclient import TestClient

from tests.images import face_like_png, random_png
from tests.test_app.enroll import BOX, enroll


def _recognize(client: TestClient, data: bytes) -> dict:  # type: ignore[type-arg]
    files = {"image": ("q.png", data, "image/png")}
    response = client.post("/recognize", files=files, data=BOX)
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


def test_recognize_before_training_409(client: TestClient) -> None:
    files = {"image": ("q.png", random_png(1), "image/png")}
    response = client.post("/recognize", files=files, data=BOX)
    assert response.status_code == 409
    assert response.json()["detail"] == "Train the model first"


def test_recognize_enrolled_person_known(client: TestClient) -> None:
    people = enroll(client, n_people=3, n_samples=4)
    client.post("/model/train", json={"n_components": 3})
    result = _recognize(client, face_like_png(2, 99))
    assert result["outcome"] == "known"
    assert result["quadrant"] == 1
    assert result["nearest"]["person_id"] == people[1]
    assert result["nearest"]["name"] == "Person 2"
    distances = [d["distance"] for d in result["distances"]]
    assert distances == sorted(distances) and len(distances) == 3
    assert result["input_png"].startswith("data:image/png;base64,")
    assert result["projection_png"].startswith("data:image/png;base64,")
    assert len(result["weights"]) == 3
    assert result["stale"] is False


def test_recognize_noise_not_a_face(client: TestClient) -> None:
    enroll(client, n_people=3, n_samples=4)
    client.post("/model/train", json={"n_components": 3})
    assert _recognize(client, random_png(7))["outcome"] == "not_a_face"


def test_recognize_rejects_bad_box(client: TestClient) -> None:
    enroll(client, n_people=2, n_samples=3)
    client.post("/model/train", json={})
    files = {"image": ("q.png", random_png(1), "image/png")}
    response = client.post("/recognize", files=files, data={**BOX, "w": 1000})
    assert response.status_code == 422


def test_recognize_after_person_deleted_names_deleted_person(client: TestClient) -> None:
    people = enroll(client, n_people=3, n_samples=4)
    client.post("/model/train", json={"n_components": 3})
    client.delete(f"/people/{people[2]}")
    result = _recognize(client, face_like_png(3, 99))
    assert result["nearest"]["person_id"] == people[2]
    assert result["nearest"]["name"] == "(deleted person)"
    assert result["stale"] is True


def test_reconstruct_sample_returns_png(client: TestClient) -> None:
    people = enroll(client, n_people=2, n_samples=3)
    client.post("/model/train", json={"n_components": 3})
    sample_id = client.get(f"/people/{people[0]}/samples").json()[0]["id"]
    for k in (0, 3):
        response = client.get(f"/model/reconstruct/{sample_id}", params={"k": k})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    assert client.get(f"/model/reconstruct/{sample_id}", params={"k": 4}).status_code == 422


def test_reconstruct_before_training_409(client: TestClient) -> None:
    people = enroll(client, n_people=1, n_samples=1)
    sample_id = client.get(f"/people/{people[0]}/samples").json()[0]["id"]
    assert client.get(f"/model/reconstruct/{sample_id}", params={"k": 1}).status_code == 409


def test_new_person_does_not_inherit_a_deleted_persons_id(client: TestClient) -> None:
    """SQLite reuses the highest freed rowid unless AUTOINCREMENT is set: a person
    created after a delete must not be named for the deleted person's face."""
    people = enroll(client, n_people=3, n_samples=4)
    client.post("/model/train", json={"n_components": 3})
    client.delete(f"/people/{people[2]}")
    newcomer = client.post("/people", json={"name": "Newcomer"}).json()["id"]
    assert newcomer != people[2]
    result = _recognize(client, face_like_png(3, 99))
    assert result["nearest"]["name"] == "(deleted person)"
