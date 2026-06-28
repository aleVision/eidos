"""Database models.

These define *what we store*, not *how recognition works*. The CV logic lives
in eidos_core and arrives in the next step. For now we model the two core
entities the product revolves around:

- Person:     someone the user has enrolled (e.g. "Ana").
- FaceSample: one enrolled face image belonging to a Person.

The marked face region (the box the user draws by hand in V1) is stored on the
sample so we can crop consistently later. It is nullable now because the
scaffold creates no rows yet.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class FaceSample(Base):
    __tablename__ = "face_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), nullable=False
    )

    # Path to the stored image file on disk (images are not kept in the DB).
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # The user-marked face box: top-left corner plus width/height, in pixels.
    # Nullable for now; populated once enrollment exists.
    box_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box_h: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    person: Mapped[Person] = relationship(back_populates="samples")
