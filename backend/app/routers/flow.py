"""1993 · Robust optical flow (Black & Anandan). Stateless: nothing is stored.

Model 1993 (the robust energy), solver 1993 (SOR + graduated non-convexity +
coarse-to-fine); see eidos_core.flow.
"""

from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, status

from app.imaging import array_png, png_data_url, read_image
from app.schemas import FlowOut, FlowSampleOut, FlowStageOut
from eidos_core.flow import FlowResult, angular_error, endpoint_error, estimate_flow, flow_to_rgb
from eidos_core.synthetic_flow import two_surfaces

router = APIRouter(prefix="/flow", tags=["1993 · robust optical flow"])

# Measured: a robust solve takes ≈ 2 s at 240 px (40 sweeps per level and stage).
PREVIEW_MAX_SIDE = 240
ARROW_STEP = 16  # one arrow every 16 preview pixels

Method = Literal["robust", "quadratic"]
# Below 0.3 the problem is ill-posed: noise invents motion (measured, see eidos_core.flow).
SMOOTHNESS_MIN, SMOOTHNESS_MAX = 0.3, 10.0


def _unprocessable(error: ValueError) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error))


def _load(upload: UploadFile) -> np.ndarray:
    try:
        return read_image(upload.file.read())
    except ValueError as error:
        raise _unprocessable(error) from error


def _fit(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """Scale down so the longer side is at most `max_side` (never scale up)."""
    height, width = pixels.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return pixels
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))


def _mask_png(mask: np.ndarray) -> str:
    return png_data_url(array_png(np.where(mask, 0, 255).astype(np.uint8)))


def _arrows(u: np.ndarray, v: np.ndarray) -> list[list[float]]:
    rows = []
    for y in range(ARROW_STEP // 2, u.shape[0], ARROW_STEP):
        for x in range(ARROW_STEP // 2, u.shape[1], ARROW_STEP):
            rows.append([float(x), float(y), round(float(u[y, x]), 3), round(float(v[y, x]), 3)])
    return rows


def _payload(result: FlowResult, max_magnitude: float, method: Method, smoothness: float) -> dict:
    height, width = result.u.shape
    return {
        "flow_png": png_data_url(array_png(flow_to_rgb(result.u, result.v, max_magnitude))),
        "outliers_png": _mask_png(result.data_outliers),
        "discontinuities_png": _mask_png(result.discontinuities),
        "arrows": _arrows(result.u, result.v),
        "stages": [FlowStageOut(**vars(stage)) for stage in result.stages],
        "max_magnitude": max_magnitude,
        "width": width,
        "height": height,
        "method": method,
        "smoothness": smoothness,
    }


def _colour_scale(result: FlowResult) -> float:
    """Speed shown as full colour: the 99th percentile, so that a few wild pixels (e.g.
    at an edge where new content appears) do not wash out the whole map."""
    return float(np.percentile(np.hypot(result.u, result.v), 99)) or 1.0


def _solve(frame1: np.ndarray, frame2: np.ndarray, smoothness: float, method: Method) -> FlowResult:
    try:
        return estimate_flow(frame1, frame2, smoothness=smoothness, robust=method == "robust")
    except ValueError as error:
        raise _unprocessable(error) from error


@router.post("/estimate", response_model=FlowOut)
def estimate(
    frame1: UploadFile,
    frame2: UploadFile,
    smoothness: Annotated[float, Form(ge=SMOOTHNESS_MIN, le=SMOOTHNESS_MAX)] = 1.0,
    method: Annotated[Method, Form()] = "robust",
) -> FlowOut:
    """Motion of every pixel from frame 1 to frame 2, at preview size (≤ 240 px)."""
    first, second = _load(frame1), _load(frame2)
    if first.shape != second.shape:  # compared upright (after EXIF), before scaling
        raise _unprocessable(ValueError("Frames must have the same size"))
    result = _solve(
        _fit(first, PREVIEW_MAX_SIDE), _fit(second, PREVIEW_MAX_SIDE), smoothness, method
    )
    return FlowOut(**_payload(result, _colour_scale(result), method, smoothness))


@router.get("/sample", response_model=FlowSampleOut)
def sample(
    motion: Annotated[float, Query(ge=0, le=8)] = 2.0,
    noise: Annotated[float, Query(ge=0, le=0.3)] = 0.03,
    smoothness: Annotated[float, Query(ge=SMOOTHNESS_MIN, le=SMOOTHNESS_MAX)] = 1.0,
    method: Method = "robust",
) -> FlowSampleOut:
    """The paper's test: a textured square moving left over a still textured surface,
    with uniform noise of ±noise·255 on frame 2. Includes the true flow and the errors."""
    frame1, frame2, u_true, v_true = two_surfaces(size=PREVIEW_MAX_SIDE, motion=motion, noise=noise)
    result = _solve(frame1, frame2, smoothness, method)
    scale = max(motion, 1.0)  # same colour scale for truth, robust and quadratic
    grey = lambda frame: png_data_url(array_png(np.clip(frame, 0, 255).astype(np.uint8)))  # noqa: E731
    return FlowSampleOut(
        **_payload(result, scale, method, smoothness),
        frame1_png=grey(frame1),
        frame2_png=grey(frame2),
        truth_png=png_data_url(array_png(flow_to_rgb(u_true, v_true, scale))),
        epe=endpoint_error(result.u, result.v, u_true, v_true),
        angular_error=angular_error(result.u, result.v, u_true, v_true),
    )
