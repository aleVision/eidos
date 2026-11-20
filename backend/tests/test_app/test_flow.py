"""API tests: robust optical flow (1993), stateless /flow endpoints."""

import base64
import io

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.routers import flow
from tests.images import png_bytes


def _decode(data_url: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def _texture(width: int, height: int, seed: int = 3) -> np.ndarray:
    noise = np.random.default_rng(seed).uniform(0, 255, (height, width)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    return (smooth - smooth.min()) / (smooth.max() - smooth.min()) * 255


def _pair(width: int = 120, height: int = 90, dx: int = 2) -> tuple[bytes, bytes]:
    """A textured photo and the same photo with its content moved right by dx pixels."""
    image = _texture(width + dx, height)
    return png_bytes(image[:, dx:]), png_bytes(image[:, :width])


def _estimate(client: TestClient, frame1: bytes, frame2: bytes, **data: object) -> httpx.Response:
    files = {"frame1": ("a.png", frame1, "image/png"), "frame2": ("b.png", frame2, "image/png")}
    response: httpx.Response = client.post("/flow/estimate", files=files, data=data)
    return response


def test_estimate_payload(client: TestClient) -> None:
    body = _estimate(client, *_pair()).json()
    assert (body["width"], body["height"]) == (120, 90)
    assert body["method"] == "robust" and body["smoothness"] == 1.0
    for key in ("flow_png", "outliers_png", "discontinuities_png"):
        assert _decode(body[key]).shape[:2] == (90, 120)
    assert len(body["stages"]) == 6
    x, y, u, v = body["arrows"][0]
    assert 0 <= x < 120 and 0 <= y < 90
    # the content moved right by 2 px
    mean_u = np.mean([arrow[2] for arrow in body["arrows"]])
    assert mean_u == pytest.approx(2.0, abs=0.3)


def test_colour_scale_ignores_a_few_wild_pixels(client: TestClient) -> None:
    """Pixels at the edge where new content appears get wild flow; the colour scale
    must follow the motion of the image (99th percentile), not its single worst pixel."""
    body = _estimate(client, *_pair(dx=2)).json()
    assert body["max_magnitude"] == pytest.approx(2.0, abs=0.5)


def test_quadratic_differs_from_robust(client: TestClient) -> None:
    pair = _pair()
    robust = _estimate(client, *pair).json()
    quadratic = _estimate(client, *pair, method="quadratic").json()
    assert quadratic["method"] == "quadratic"
    assert len({s["sigma_d"] for s in quadratic["stages"]}) == 1
    assert robust["flow_png"] != quadratic["flow_png"]


def test_large_photos_are_scaled_to_the_preview_size(client: TestClient) -> None:
    body = _estimate(client, *_pair(width=800, height=600)).json()
    assert max(body["width"], body["height"]) == flow.PREVIEW_MAX_SIDE


def test_mismatched_sizes_422(client: TestClient) -> None:
    first, _ = _pair(120, 90)
    _, second = _pair(100, 90)
    response = _estimate(client, first, second)
    assert response.status_code == 422
    assert "same size" in response.json()["detail"]


def test_sizes_are_compared_after_exif_rotation(client: TestClient) -> None:
    stored = Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # displayed rotated: 80 wide, 120 tall
    buffer = io.BytesIO()
    stored.save(buffer, "JPEG", exif=exif.tobytes())
    rotated = buffer.getvalue()
    assert _estimate(client, rotated, rotated).json()["width"] == 80
    plain = png_bytes(np.full((80, 120, 3), 128))
    assert _estimate(client, rotated, plain).status_code == 422


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_greyscale_and_rgba_uploads(client: TestClient, mode: str) -> None:
    first, second = _pair()
    convert = lambda data: _repack(data, mode)  # noqa: E731
    assert _estimate(client, convert(first), convert(second)).status_code == 200


def _repack(data: bytes, mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.open(io.BytesIO(data)).convert(mode).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("smoothness", [0.1, 11, "nan"])
def test_smoothness_out_of_range_422(client: TestClient, smoothness: object) -> None:
    assert _estimate(client, *_pair(), smoothness=smoothness).status_code == 422


def test_unknown_method_and_corrupt_image_422(client: TestClient) -> None:
    assert _estimate(client, *_pair(), method="magic").status_code == 422
    assert _estimate(client, b"not an image", b"nope").status_code == 422


def test_sample_is_deterministic_with_truth_and_errors(client: TestClient) -> None:
    first = client.get("/flow/sample").json()
    second = client.get("/flow/sample").json()
    assert first == second
    for key in ("frame1_png", "frame2_png", "truth_png", "flow_png"):
        assert first[key].startswith("data:image/png;base64,")
    assert first["epe"] > 0 and first["angular_error"] > 0


def test_sample_robust_beats_quadratic_with_noise(client: TestClient) -> None:
    robust = client.get("/flow/sample", params={"noise": 0.1}).json()
    quadratic = client.get("/flow/sample", params={"noise": 0.1, "method": "quadratic"}).json()
    assert robust["epe"] < quadratic["epe"]


@pytest.mark.parametrize("params", [{"motion": 9}, {"noise": 0.5}, {"smoothness": 20}])
def test_sample_bounds_422(client: TestClient, params: dict[str, float]) -> None:
    assert client.get("/flow/sample", params=params).status_code == 422
