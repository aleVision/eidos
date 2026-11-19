"""Enroll synthetic people through the API (shared by model and recognition tests)."""

from fastapi.testclient import TestClient

from tests.images import face_like_png

BOX = {"x": 10, "y": 20, "w": 80, "h": 100}


def enroll(client: TestClient, n_people: int, n_samples: int) -> list[int]:
    ids = []
    for person in range(1, n_people + 1):
        person_id = client.post("/people", json={"name": f"Person {person}"}).json()["id"]
        for sample in range(1, n_samples + 1):
            files = {"image": ("f.png", face_like_png(person, sample), "image/png")}
            response = client.post(f"/people/{person_id}/samples", files=files, data=BOX)
            assert response.status_code == 201
        ids.append(person_id)
    return ids
