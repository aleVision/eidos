"""Recognize a face, and rebuild stored samples from their eigenface weights."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.imaging import png_data_url, read_image, to_png
from app.models import FaceSample, Person
from app.schemas import PersonDistance, RecognitionOut, ThresholdsOut
from app.services import NoModelError, is_stale, load_active_model, sample_vector
from eidos_core.eigenfaces import EigenfaceModel, project, reconstruct
from eidos_core.preprocess import prepare
from eidos_core.recognition import Thresholds, classify

router = APIRouter(tags=["recognition"])

DbSession = Annotated[Session, Depends(get_db)]
DELETED = "(deleted person)"


def _model(db: Session) -> tuple[EigenfaceModel, Thresholds, bool]:
    try:
        model, thresholds, run = load_active_model(db)
    except NoModelError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, "Train the model first") from error
    return model, thresholds, is_stale(db, run)


@router.post("/recognize", response_model=RecognitionOut)
def recognize(
    image: UploadFile,
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
    w: Annotated[int, Form()],
    h: Annotated[int, Form()],
    db: DbSession,
) -> RecognitionOut:
    model, thresholds, stale = _model(db)
    try:
        face = prepare(read_image(image.file.read()), (x, y, w, h), settings.gaussian_window)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    result = classify(model, face, thresholds)
    names = dict(db.execute(select(Person.id, Person.name)).tuples().all())
    distances = sorted(
        (
            PersonDistance(person_id=pid, name=names.get(pid, DELETED), distance=d)
            for pid, d in result.class_distances.items()
        ),
        key=lambda item: item.distance,
    )
    return RecognitionOut(
        outcome=result.outcome.value,
        quadrant=result.quadrant,
        nearest=distances[0],
        face_space_distance=result.face_space_distance,
        distances=distances,
        thresholds=ThresholdsOut(person=thresholds.person, face_space=thresholds.face_space),
        weights=result.weights.tolist(),
        input_png=png_data_url(to_png(face, model.image_shape, stretch=False)),
        projection_png=png_data_url(
            to_png(reconstruct(model, result.weights), model.image_shape, stretch=False)
        ),
        stale=stale,
    )


@router.get("/model/reconstruct/{sample_id}")
def reconstruct_sample(sample_id: int, k: Annotated[int, Query(ge=0)], db: DbSession) -> Response:
    """The sample rebuilt as Ψ + Σ ω_i u_i over its first k eigenfaces."""
    sample = db.get(FaceSample, sample_id)
    if sample is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample not found")
    model, _, _ = _model(db)
    if k > model.eigenfaces.shape[1]:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "k exceeds the eigenfaces kept")
    weights = project(model, sample_vector(sample))
    png = to_png(reconstruct(model, weights, k), model.image_shape, stretch=False)
    return Response(content=png, media_type="image/png")
