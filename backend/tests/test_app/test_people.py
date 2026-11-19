"""API tests: people and their face samples."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from tests.images import random_png

BOX = {"x": 10, "y": 20, "w": 80, "h": 100}


def _person(client: TestClient, name: str = "Ana") -> int:
    response = client.post("/people", json={"name": name})
    assert response.status_code == 201
    person_id: int = response.json()["id"]
    return person_id


def _upload(client: TestClient, person_id: int, data: bytes, box: dict[str, int] = BOX):
    files = {"image": ("face.png", data, "image/png")}
    return client.post(f"/people/{person_id}/samples", files=files, data=box)


def test_create_list_delete_person(client: TestClient) -> None:
    person_id = _person(client, "  Ana  ")
    people = client.get("/people").json()
    assert people == [{"id": person_id, "name": "Ana", "is_demo": False, "n_samples": 0}]

    assert client.delete(f"/people/{person_id}").status_code == 204
    assert client.get("/people").json() == []
    assert client.delete(f"/people/{person_id}").status_code == 404


def test_person_name_validation(client: TestClient) -> None:
    assert client.post("/people", json={"name": "   "}).status_code == 422
    assert client.post("/people", json={"name": "x" * 121}).status_code == 422


def test_upload_sample_and_fetch_image(client: TestClient) -> None:
    person_id = _person(client)
    response = _upload(client, person_id, random_png(1))
    assert response.status_code == 201
    sample = response.json()
    assert sample["person_id"] == person_id
    assert sample["box"] == BOX

    listed = client.get(f"/people/{person_id}/samples").json()
    assert [s["id"] for s in listed] == [sample["id"]]
    image = client.get(f"/samples/{sample['id']}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert client.get("/people").json()[0]["n_samples"] == 1


def test_upload_rejects_box_outside_image(client: TestClient) -> None:
    person_id = _person(client)
    response = _upload(client, person_id, random_png(1), {"x": 150, "y": 0, "w": 100, "h": 50})
    assert response.status_code == 422
    assert response.json()["detail"] == "Face box must lie inside the image"


def test_upload_rejects_corrupt_image(client: TestClient) -> None:
    person_id = _person(client)
    response = _upload(client, person_id, b"definitely not an image")
    assert response.status_code == 422
    assert response.json()["detail"] == "Unsupported or corrupt image"


def test_upload_to_missing_person_404(client: TestClient) -> None:
    assert _upload(client, 999, random_png(1)).status_code == 404


def test_exif_rotated_jpeg_box_uses_upright_pixels(client: TestClient) -> None:
    upright = np.zeros((100, 60, 3), dtype=np.uint8)  # 60 wide x 100 tall, as displayed
    upright[:20, :20] = 255  # white square in the displayed top-left corner
    stored = Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise to display
    buffer = io.BytesIO()
    stored.save(buffer, "JPEG", exif=exif.tobytes(), quality=95)

    person_id = _person(client)
    response = _upload(client, person_id, buffer.getvalue(), {"x": 0, "y": 0, "w": 20, "h": 20})
    assert response.status_code == 201

    saved = client.get(f"/samples/{response.json()['id']}/image").content
    pixels = np.asarray(Image.open(io.BytesIO(saved)).convert("L"))
    assert pixels.shape == (100, 60)
    assert pixels[:20, :20].mean() > 200


def test_delete_sample_and_person_remove_files(client: TestClient) -> None:
    person_id = _person(client)
    first = _upload(client, person_id, random_png(1)).json()["id"]
    _upload(client, person_id, random_png(2))
    assert len(list(settings.images_dir.iterdir())) == 2

    assert client.delete(f"/samples/{first}").status_code == 204
    assert len(list(settings.images_dir.iterdir())) == 1

    client.delete(f"/people/{person_id}")
    assert list(settings.images_dir.iterdir()) == []
