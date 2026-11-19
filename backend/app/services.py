"""Glue between storage (DB rows, files) and eidos_core. No HTTP concerns here."""

from datetime import datetime

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.imaging import read_image
from app.models import FaceSample, ModelRun
from eidos_core.eigenfaces import EigenfaceModel, train
from eidos_core.persistence import load_model, save_model
from eidos_core.preprocess import prepare
from eidos_core.recognition import Thresholds, suggest_thresholds


class NoModelError(Exception):
    """Raised when an operation needs a trained model and none exists."""


def sample_vector(sample: FaceSample) -> np.ndarray:
    """The prepared face vector Γ of a stored sample."""
    pixels = read_image((settings.images_dir / sample.image_path).read_bytes())
    box = (sample.box_x, sample.box_y, sample.box_w, sample.box_h)
    return prepare(pixels, box, settings.gaussian_window)


def load_training_set(db: Session) -> tuple[np.ndarray, list[int]]:
    samples = db.scalars(select(FaceSample).order_by(FaceSample.id)).all()
    faces = np.array([sample_vector(s) for s in samples])
    return faces, [s.person_id for s in samples]


def train_model(db: Session, n_components: int) -> ModelRun:
    """Train on every enrolled sample and store the result as the active run.

    Raises ValueError (from eidos_core) when there is not enough data. The model
    file is written before the row is committed and removed if the commit fails.
    """
    faces, labels = load_training_set(db)
    model = train(faces, labels, n_components)
    thresholds = suggest_thresholds(model, faces, labels)
    run = ModelRun(
        n_people=len(set(labels)),
        n_samples=len(labels),
        n_components=model.eigenfaces.shape[1],
        theta_person=thresholds.person,
        theta_face=thresholds.face_space,
        path="",
    )
    db.add(run)
    db.flush()  # assigns run.id
    run.path = f"run-{run.id}.npz"
    path = settings.models_dir / run.path
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, path)
    try:
        db.commit()
    except Exception:
        path.unlink(missing_ok=True)
        db.rollback()
        raise
    return run


def active_run(db: Session) -> ModelRun | None:
    return db.scalars(select(ModelRun).order_by(ModelRun.id.desc()).limit(1)).first()


def load_active_model(db: Session) -> tuple[EigenfaceModel, Thresholds, ModelRun]:
    run = active_run(db)
    if run is None:
        raise NoModelError
    model = load_model(settings.models_dir / run.path)
    return model, Thresholds(person=run.theta_person, face_space=run.theta_face), run


def is_stale(db: Session, run: ModelRun) -> bool:
    """True when people or samples changed since `run` was trained."""
    n_samples = db.scalar(select(func.count(FaceSample.id))) or 0
    n_people = db.scalar(select(func.count(func.distinct(FaceSample.person_id)))) or 0
    newest: datetime | None = db.scalar(select(func.max(FaceSample.created_at)))
    changed_counts = (n_samples, n_people) != (run.n_samples, run.n_people)
    return changed_counts or (newest is not None and newest > run.created_at)
