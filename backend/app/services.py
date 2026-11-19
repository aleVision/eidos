"""Glue between storage (DB rows, files) and eidos_core. No HTTP concerns here."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.imaging import read_image
from app.models import FaceSample, ModelRun
from eidos_core.datasets.orl import load_orl_image
from eidos_core.eigenfaces import EigenfaceModel, distance_from_face_space, train
from eidos_core.persistence import load_model, save_model
from eidos_core.preprocess import prepare
from eidos_core.recognition import Thresholds, suggest_thresholds


class NoModelError(Exception):
    """Raised when an operation needs a trained model and none exists."""


class NoDetectorError(Exception):
    """Raised when face detection is used before the detector is set up."""


@dataclass(frozen=True)
class DetectorInfo:
    threshold: float  # θ on ε for a "confident" detection, normalized method
    paper_threshold: float  # the same for the paper's raw ε
    n_components: int
    use_window: bool  # the window setting the detector was trained with


# The detector's own face space, trained once from ORL and independent of people.
DETECTOR_SUBJECTS = range(1, 41)
DETECTOR_TRAIN_PHOTOS = range(1, 9)  # photos 9-10 measure held-out ε for the threshold
DETECTOR_HELD_OUT_PHOTOS = range(9, 11)
DETECTOR_COMPONENTS = 20
DETECTOR_MARGIN = 1.5


def _detector_paths() -> tuple[Path, Path, Path]:
    """(normalized face space, the paper's raw face space, settings)."""
    folder = settings.models_dir
    return folder / "detector.npz", folder / "detector-paper.npz", folder / "detector.json"


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


def _train_face_space(
    orl_root: Path, use_window: bool, standardize_face: bool
) -> tuple[EigenfaceModel, float]:
    """One detector face space from ORL, and its threshold on held-out ε."""

    def vector(subject: int, index: int) -> np.ndarray:
        pixels = load_orl_image(orl_root, subject, index)
        rows, cols = pixels.shape
        return prepare(pixels, (0, 0, cols, rows), use_window, standardize_face)

    faces = np.array([vector(s, i) for s in DETECTOR_SUBJECTS for i in DETECTOR_TRAIN_PHOTOS])
    labels = [s for s in DETECTOR_SUBJECTS for _ in DETECTOR_TRAIN_PHOTOS]
    model = train(faces, labels, DETECTOR_COMPONENTS)
    held_out = [
        distance_from_face_space(model, vector(s, i))
        for s in DETECTOR_SUBJECTS
        for i in DETECTOR_HELD_OUT_PHOTOS
    ]
    return model, DETECTOR_MARGIN * float(np.percentile(held_out, 95))


def train_detector(orl_root: Path) -> DetectorInfo:
    """Learn both detector face spaces from ORL and store them next to the models.

    "normalized" standardizes every face first (not in the 1991 paper); "paper"
    keeps the raw pixels, for comparison.
    """
    use_window = settings.gaussian_window
    normalized, threshold = _train_face_space(orl_root, use_window, standardize_face=True)
    paper, paper_threshold = _train_face_space(orl_root, use_window, standardize_face=False)
    info = DetectorInfo(
        threshold=threshold,
        paper_threshold=paper_threshold,
        n_components=normalized.eigenfaces.shape[1],
        use_window=use_window,
    )
    normalized_path, paper_path, info_path = _detector_paths()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    save_model(normalized, normalized_path)
    save_model(paper, paper_path)
    info_path.write_text(json.dumps(asdict(info)))
    return info


def load_detector(normalized: bool = True) -> tuple[EigenfaceModel, DetectorInfo]:
    """The detector face space for one method, with the detector's settings."""
    normalized_path, paper_path, info_path = _detector_paths()
    if not all(p.exists() for p in (normalized_path, paper_path, info_path)):
        raise NoDetectorError
    model = load_model(normalized_path if normalized else paper_path)
    return model, DetectorInfo(**json.loads(info_path.read_text()))
