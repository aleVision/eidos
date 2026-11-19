"""Face detection with the face map (Turk & Pentland 1991): suggest a face box."""

import threading
from typing import Annotated, Literal

from fastapi import APIRouter, Form, HTTPException, Response, UploadFile, status

from app.imaging import face_map_png, png_data_url, read_image
from app.routers.demo import OrlRoot, download_orl
from app.schemas import Box, DetectionOut, DetectorOut, DetectorStatus
from app.services import NoDetectorError, detector_ready, load_detector, train_detector
from eidos_core.detection import detect

router = APIRouter(tags=["detection"])

# Two setup requests at once (e.g. a double click) must not write the same files.
_setup_lock = threading.Lock()


@router.get("/detector", response_model=DetectorStatus)
def detector_status() -> DetectorStatus:
    return DetectorStatus(available=detector_ready())


@router.post("/detector", response_model=DetectorOut, status_code=status.HTTP_201_CREATED)
def set_up_detector(root: OrlRoot, response: Response) -> DetectorOut:
    """Download ORL if needed and train the detector once. Adds no demo people."""
    with _setup_lock:
        if detector_ready():
            _, info = load_detector()
            response.status_code = status.HTTP_200_OK  # already set up
        else:
            download_orl(root)
            info = train_detector(root)
    return DetectorOut(
        threshold=info.threshold,
        paper_threshold=info.paper_threshold,
        n_components=info.n_components,
    )


@router.post("/detect", response_model=DetectionOut)
def detect_face(
    image: UploadFile,
    method: Annotated[Literal["normalized", "paper"], Form()] = "normalized",
) -> DetectionOut:
    """Suggest the most face-like box.

    method "normalized" (default) standardizes each box first, which is not in the
    1991 paper; "paper" uses the paper's raw ε, for comparison.
    """
    normalized = method == "normalized"
    try:
        model, info = load_detector(normalized)
    except NoDetectorError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, "Face detection is not set up yet") from error
    try:
        pixels = read_image(image.file.read())
        # Use the window the detector was trained with, not today's setting.
        found = detect(model, pixels, use_window=info.use_window, normalize=normalized)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    x, y, w, h = found.box
    height, width = pixels.shape[:2]
    face_map = face_map_png(found.face_map, model.image_shape, (width, height))
    return DetectionOut(
        box=Box(x=x, y=y, w=w, h=h),
        distance=found.distance,
        confident=found.distance <= (info.threshold if normalized else info.paper_threshold),
        map_png=png_data_url(face_map),
    )
