"""1992 · Total variation denoising (Rudin, Osher & Fatemi). Stateless: nothing is stored.

Model 1992 (the ROF energy), solver 2011 (Chambolle & Pock); see eidos_core.denoise.
"""

from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Query, Response, UploadFile, status

from app.imaging import array_png, png_data_url, read_image
from app.schemas import DenoiseOut
from eidos_core.denoise import denoise_tv

router = APIRouter(prefix="/tv", tags=["1992 · total variation"])

# Measured (colour, λ = 8 / strongest λ = 1): 640 px ≈ 1.7 / 2.9 s for a live preview;
# 2048 px (3 MP) ≈ 23 / 44 s for a download. Larger photos are scaled down to these sizes.
PREVIEW_MAX_SIDE = 640
DOWNLOAD_MAX_SIDE = 2048
_NOISE_GAIN = 4.0  # the removed noise is faint; amplify it so it can be seen

Strength = Annotated[float, Form(gt=0, description="λ: larger stays closer to the photo")]


def _load(image: UploadFile) -> np.ndarray:
    try:
        return read_image(image.file.read()).astype(np.float32) / 255.0
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error


def _fit(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """Scale down so the longer side is at most `max_side` (never scale up)."""
    height, width = pixels.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return pixels
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))


def _png(image: np.ndarray) -> bytes:
    return array_png((np.clip(image, 0.0, 1.0) * 255 + 0.5).astype(np.uint8))


def _denoise(pixels: np.ndarray, strength: float) -> tuple[np.ndarray, list[float]]:
    try:
        result = denoise_tv(pixels, strength)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return result.image, result.energies


@router.post("/denoise", response_model=DenoiseOut)
def denoise_preview(
    image: UploadFile,
    strength: Strength,
    max_side: Annotated[int, Form(ge=16, le=DOWNLOAD_MAX_SIDE)] = PREVIEW_MAX_SIDE,
) -> DenoiseOut:
    """A quick preview: the cleaned photo, the removed noise, and E(u) per iteration."""
    noisy = _fit(_load(image), max_side)
    cleaned, energies = _denoise(noisy, strength)
    height, width = cleaned.shape[:2]
    return DenoiseOut(
        image_png=png_data_url(_png(cleaned)),
        noise_png=png_data_url(_png(0.5 + _NOISE_GAIN * (noisy - cleaned))),
        energies=energies,
        lam=strength,
        width=width,
        height=height,
    )


@router.post("/denoise.png")
def denoise_download(image: UploadFile, strength: Strength) -> Response:
    """The cleaned photo as a PNG file, up to DOWNLOAD_MAX_SIDE pixels on the longer side."""
    cleaned, _ = _denoise(_fit(_load(image), DOWNLOAD_MAX_SIDE), strength)
    return Response(
        content=_png(cleaned),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="eidos-denoised.png"'},
    )


def _sample_scene() -> np.ndarray:
    """A clean 768 x 512 test scene: flat shapes, a gradient, thin lines and text."""
    height, width = 512, 768
    x = np.linspace(0, 1, width, dtype=np.float32)
    scene = np.empty((height, width, 3), dtype=np.float32)
    scene[...] = np.stack([0.25 + 0.5 * x, 0.35 + 0.2 * x, 0.75 - 0.4 * x], axis=-1)
    canvas = (scene * 255).astype(np.uint8)
    cv2.rectangle(canvas, (60, 60), (300, 220), (230, 200, 60), thickness=-1)
    cv2.circle(canvas, (520, 160), 100, (40, 90, 170), thickness=-1)
    cv2.circle(canvas, (520, 160), 50, (245, 245, 240), thickness=-1)
    for i in range(8):  # thin stripes: hard for blur, easy for TV
        cv2.line(canvas, (80 + 18 * i, 280), (80 + 18 * i, 450), (20, 20, 20), thickness=3)
    cv2.putText(canvas, "1992", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (250, 250, 250), 9)
    return canvas.astype(np.float32)


@router.get("/sample.png")
def sample(noise: Annotated[int, Query(ge=0, le=80)] = 25) -> Response:
    """The test scene plus Gaussian noise of `noise` grey levels (same every time)."""
    rng = np.random.default_rng(1992)
    noisy = _sample_scene() + rng.normal(0.0, noise, (512, 768, 3)).astype(np.float32)
    return Response(content=_png(noisy / 255.0), media_type="image/png")
