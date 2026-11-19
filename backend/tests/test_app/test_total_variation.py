"""API tests: total variation denoising (1992), stateless /tv endpoints."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import total_variation
from tests.images import png_bytes


def _decode(data_url_or_bytes: str | bytes) -> np.ndarray:
    raw = data_url_or_bytes
    if isinstance(raw, str):
        raw = base64.b64decode(raw.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


def _photo(width: int, height: int, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(5)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    pixels = rng.integers(0, 256, (height, width, channels)).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels[..., 0] if mode == "L" else pixels).save(buffer, "PNG")
    return buffer.getvalue()


def _preview(client: TestClient, data: bytes, strength: float = 8.0, **extra: object):
    files = {"image": ("p.png", data, "image/png")}
    return client.post("/tv/denoise", files=files, data={"strength": strength, **extra})


def test_denoise_preview_shapes(client: TestClient) -> None:
    response = _preview(client, _photo(1600, 1200))
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"]) == (640, 480)  # preview, longer side 640
    assert _decode(body["image_png"]).shape == (480, 640, 3)
    assert _decode(body["noise_png"]).shape == (480, 640, 3)
    assert len(body["energies"]) == 100
    assert body["lam"] == 8.0


def test_denoise_png_full_size(client: TestClient) -> None:
    files = {"image": ("p.png", _photo(300, 200), "image/png")}
    response = client.post("/tv/denoise.png", files=files, data={"strength": 8.0})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "attachment" in response.headers["content-disposition"]
    assert _decode(response.content).shape == (200, 300, 3)


def test_denoise_png_is_capped(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(total_variation, "DOWNLOAD_MAX_SIDE", 100)
    files = {"image": ("p.png", _photo(300, 200), "image/png")}
    response = client.post("/tv/denoise.png", files=files, data={"strength": 8.0})
    assert _decode(response.content).shape == (67, 100, 3)


def test_exif_rotated_upload_gives_upright_size(client: TestClient) -> None:
    stored = Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8))  # 120 wide, 80 tall
    exif = Image.Exif()
    exif[0x0112] = 6  # displayed rotated: 80 wide, 120 tall
    buffer = io.BytesIO()
    stored.save(buffer, "JPEG", exif=exif.tobytes())
    files = {"image": ("p.jpg", buffer.getvalue(), "image/jpeg")}
    body = client.post("/tv/denoise", files=files, data={"strength": 8.0}).json()
    assert (body["width"], body["height"]) == (80, 120)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_greyscale_and_rgba_uploads(client: TestClient, mode: str) -> None:
    response = _preview(client, _photo(60, 40, mode))
    assert response.status_code == 200
    assert _decode(response.json()["image_png"]).shape == (40, 60, 3)


@pytest.mark.parametrize("strength", [0, -2])
def test_strength_must_be_positive_422(client: TestClient, strength: float) -> None:
    assert _preview(client, _photo(30, 20), strength=strength).status_code == 422


def test_corrupt_image_422(client: TestClient) -> None:
    files = {"image": ("p.png", b"not an image", "image/png")}
    assert client.post("/tv/denoise", files=files, data={"strength": 8.0}).status_code == 422


def test_sample_is_deterministic_and_noise_changes_it(client: TestClient) -> None:
    first = client.get("/tv/sample.png", params={"noise": 25})
    assert first.status_code == 200 and first.headers["content-type"] == "image/png"
    assert first.content == client.get("/tv/sample.png", params={"noise": 25}).content
    assert first.content != client.get("/tv/sample.png", params={"noise": 0}).content
    assert _decode(first.content).shape == (512, 768, 3)
    assert client.get("/tv/sample.png", params={"noise": 81}).status_code == 422


def test_denoising_the_sample_reduces_error_to_clean_scene(client: TestClient) -> None:
    clean = _decode(client.get("/tv/sample.png", params={"noise": 0}).content).astype(float)
    noisy_png = client.get("/tv/sample.png", params={"noise": 25}).content
    noisy = _decode(noisy_png).astype(float)
    body = _preview(client, noisy_png, strength=8.0, max_side=768).json()
    cleaned = _decode(body["image_png"]).astype(float)
    assert np.abs(cleaned - clean).mean() < 0.6 * np.abs(noisy - clean).mean()


def test_rgb_png_helper_roundtrip() -> None:
    assert _decode(png_bytes(np.zeros((2, 3, 3)))).shape == (2, 3, 3)
