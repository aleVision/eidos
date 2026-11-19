"""Database models.

These define *what we store*, not *how recognition works*; the CV logic lives
in eidos_core.

- Person:     someone the user has enrolled (e.g. "Ana").
- FaceSample: one enrolled face image belonging to a Person, with the face box
              the user drew by hand (V1 has no automatic detection).
- ModelRun:   one training of the eigenface model and its thresholds.
"""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Person(Base):
    __tablename__ = "people"
    # Never reuse a deleted person's id: trained models refer to people by id.
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Demo people come from the public ORL dataset and are kept apart from real ones.
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class FaceSample(Base):
    __tablename__ = "face_samples"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), nullable=False
    )

    # File name inside settings.images_dir (images are not kept in the DB).
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # The user-marked face box: top-left corner plus width/height, in pixels.
    box_x: Mapped[int] = mapped_column(Integer, nullable=False)
    box_y: Mapped[int] = mapped_column(Integer, nullable=False)
    box_w: Mapped[int] = mapped_column(Integer, nullable=False)
    box_h: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    person: Mapped[Person] = relationship(back_populates="samples")


class ModelRun(Base):
    """One training of the eigenface model. The latest run is the active model."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    n_people: Mapped[int] = mapped_column(Integer, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    n_components: Mapped[int] = mapped_column(Integer, nullable=False)
    theta_person: Mapped[float] = mapped_column(Float, nullable=False)
    theta_face: Mapped[float] = mapped_column(Float, nullable=False)
    # File name of the saved model inside settings.models_dir.
    path: Mapped[str] = mapped_column(String(500), nullable=False)
