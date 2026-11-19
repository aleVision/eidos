"""Train the eigenface model and look inside it (mean face, eigenfaces, variance)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.imaging import to_png
from app.models import ModelRun
from app.schemas import ModelOut, ThresholdsIn, TrainRequest
from app.services import NoModelError, is_stale, load_active_model, train_model
from eidos_core.eigenfaces import EigenfaceModel, explained_variance

router = APIRouter(prefix="/model", tags=["model"])

DbSession = Annotated[Session, Depends(get_db)]


def _model_out(db: Session, run: ModelRun, model: EigenfaceModel) -> ModelOut:
    return ModelOut(
        id=run.id,
        created_at=run.created_at,
        n_people=run.n_people,
        n_samples=run.n_samples,
        n_components=run.n_components,
        theta_person=run.theta_person,
        theta_face=run.theta_face,
        explained_variance=explained_variance(model).tolist(),
        stale=is_stale(db, run),
    )


def _active(db: Session) -> tuple[EigenfaceModel, ModelRun]:
    try:
        model, _, run = load_active_model(db)
    except NoModelError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No model trained yet") from error
    return model, run


def _png(content: bytes) -> Response:
    return Response(content=content, media_type="image/png")


@router.post("/train", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
def train(body: TrainRequest, db: DbSession) -> ModelOut:
    try:
        train_model(db, body.n_components or settings.default_components)
    except ValueError as error:
        db.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    model, run = _active(db)
    return _model_out(db, run, model)


@router.get("", response_model=ModelOut)
def get_model(db: DbSession) -> ModelOut:
    model, run = _active(db)
    return _model_out(db, run, model)


@router.patch("/thresholds", response_model=ModelOut)
def set_thresholds(body: ThresholdsIn, db: DbSession) -> ModelOut:
    model, run = _active(db)
    run.theta_person, run.theta_face = body.person, body.face_space
    db.commit()
    return _model_out(db, run, model)


@router.get("/mean.png")
def mean_face(db: DbSession) -> Response:
    model, _ = _active(db)
    return _png(to_png(model.mean, model.image_shape, stretch=False))


@router.get("/eigenfaces/{k}.png")
def eigenface(k: int, db: DbSession) -> Response:
    model, _ = _active(db)
    if not 1 <= k <= model.eigenfaces.shape[1]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such eigenface")
    return _png(to_png(model.eigenfaces[:, k - 1], model.image_shape, stretch=True))
