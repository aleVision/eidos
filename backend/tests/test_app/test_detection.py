"""API tests: setting up the face detector and suggesting face boxes."""

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.routers import detection as detection_router
from app.routers.demo import get_orl_root
from tests.images import png_bytes
from tests.orl_fake import write_fake_orl
from tests.test_app.enroll import enroll


@pytest.fixture
def orl_client(client: TestClient, tmp_path: Path) -> Iterator[TestClient]:
    root = write_fake_orl(tmp_path / "orl", subjects=40, per_subject=10)
    app.dependency_overrides[get_orl_root] = lambda: root
    yield client


@pytest.fixture
def detect_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record what the API passes to eidos_core's detect, then run the real one."""
    calls: list[dict[str, Any]] = []
    real_detect = detection_router.detect

    def spy(model: Any, image: np.ndarray, use_window: bool = True, normalize: bool = False) -> Any:
        calls.append(
            {"shape": image.shape, "use_window": use_window, "normalize": normalize, "model": model}
        )
        return real_detect(model, image, use_window=use_window, normalize=normalize)

    monkeypatch.setattr(detection_router, "detect", spy)
    return calls


def _photo(width: int = 400, height: int = 300) -> bytes:
    return png_bytes(np.random.default_rng(4).integers(0, 256, (height, width, 3)))


def _detect(client: TestClient, data: bytes) -> Any:
    return client.post("/detect", files={"image": ("p.png", data, "image/png")})


def test_detect_before_setup_409(client: TestClient) -> None:
    response = _detect(client, _photo())
    assert response.status_code == 409
    assert response.json()["detail"] == "Face detection is not set up yet"
    assert client.get("/detector").json() == {"available": False}


def test_detector_setup_then_status(orl_client: TestClient) -> None:
    created = orl_client.post("/detector")
    assert created.status_code == 201
    assert created.json()["n_components"] == 20
    assert created.json()["threshold"] > 0
    assert orl_client.get("/detector").json() == {"available": True}
    again = orl_client.post("/detector")
    assert again.status_code == 200
    assert again.json() == created.json()


def test_detect_returns_box_inside_image(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    response = _detect(orl_client, _photo(400, 300))
    assert response.status_code == 200
    body = response.json()
    box = body["box"]
    assert box["x"] >= 0 and box["y"] >= 0
    assert box["x"] + box["w"] <= 400 and box["y"] + box["h"] <= 300
    assert body["distance"] >= 0
    assert isinstance(body["confident"], bool)
    assert body["map_png"].startswith("data:image/png;base64,")


def test_detect_exif_rotated_photo_uses_upright_pixels(
    orl_client: TestClient, detect_calls: list[dict[str, Any]]
) -> None:
    orl_client.post("/detector")
    upright = np.random.default_rng(6).integers(0, 256, (400, 120, 3)).astype(np.uint8)
    stored = Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise to display
    buffer = io.BytesIO()
    stored.save(buffer, "JPEG", exif=exif.tobytes())

    response = orl_client.post(
        "/detect", files={"image": ("p.jpg", buffer.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert detect_calls[-1]["shape"][:2] == (400, 120)
    box = response.json()["box"]
    assert box["x"] + box["w"] <= 120 and box["y"] + box["h"] <= 400


def test_detector_keeps_its_window_flag(
    orl_client: TestClient, detect_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    orl_client.post("/detector")  # trained with the window on (the default)
    monkeypatch.setattr(settings, "gaussian_window", False)
    _detect(orl_client, _photo())
    assert detect_calls[-1]["use_window"] is True


def test_people_model_training_does_not_touch_detector(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    files = [
        settings.models_dir / name
        for name in ("detector.npz", "detector-paper.npz", "detector.json")
    ]
    before = [f.stat().st_mtime_ns for f in files]

    enroll(orl_client, n_people=2, n_samples=3)
    assert orl_client.post("/model/train", json={}).status_code == 201
    orl_client.delete("/demo/orl")

    assert [f.stat().st_mtime_ns for f in files] == before
    assert orl_client.get("/detector").json() == {"available": True}


def test_detect_tiny_image_422(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    response = _detect(orl_client, _photo(30, 30))
    assert response.status_code == 422
    assert response.json()["detail"] == "Image is too small to search for a face"


def test_detect_corrupt_image_422(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    response = orl_client.post("/detect", files={"image": ("p.png", b"nope", "image/png")})
    assert response.status_code == 422


def test_detector_download_failure_502(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "orl_url", (tmp_path / "missing.zip").as_uri())
    app.dependency_overrides[get_orl_root] = lambda: tmp_path / "orl"
    response = client.post("/detector")
    assert response.status_code == 502
    assert response.json()["detail"].startswith("Could not download the ORL dataset")
    assert client.get("/detector").json() == {"available": False}


def test_detect_default_method_is_normalized(
    orl_client: TestClient, detect_calls: list[dict[str, Any]]
) -> None:
    orl_client.post("/detector")
    assert _detect(orl_client, _photo()).status_code == 200
    assert detect_calls[-1]["normalize"] is True


def test_detect_paper_method_uses_raw_face_space(
    orl_client: TestClient, detect_calls: list[dict[str, Any]]
) -> None:
    orl_client.post("/detector")
    files = {"image": ("p.png", _photo(), "image/png")}
    normalized = orl_client.post("/detect", files=files, data={"method": "normalized"})
    paper = orl_client.post("/detect", files=files, data={"method": "paper"})
    assert normalized.status_code == paper.status_code == 200
    assert detect_calls[-1]["normalize"] is False
    # two different face spaces: the raw one was trained on unstandardized faces
    assert not np.allclose(detect_calls[-1]["model"].mean, detect_calls[-2]["model"].mean)


def test_detect_unknown_method_422(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    files = {"image": ("p.png", _photo(), "image/png")}
    assert orl_client.post("/detect", files=files, data={"method": "magic"}).status_code == 422


def test_detect_blank_image_422(orl_client: TestClient) -> None:
    orl_client.post("/detector")
    blank = png_bytes(np.full((300, 300, 3), 77))
    response = _detect(orl_client, blank)
    assert response.status_code == 422
    assert response.json()["detail"] == "No face-like region found"
