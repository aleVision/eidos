"""The ORL demo: 30 enrolled public-dataset people plus held-out and stranger photos.

Credit: Olivetti Research Laboratory (AT&T Laboratories Cambridge).
"""

import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.imaging import array_png
from app.models import FaceSample, Person
from app.routers.people import delete_sample_file
from app.schemas import DemoLoaded, TestPhoto
from eidos_core.datasets.orl import ensure_orl, load_orl_image

router = APIRouter(prefix="/demo/orl", tags=["demo"])

DbSession = Annotated[Session, Depends(get_db)]

ENROLLED_SUBJECTS = range(1, 31)  # s1-s30 are enrolled; s31-s40 stay strangers
TRAIN_PHOTOS = range(1, 7)  # photos 1-6 are samples; 7-10 are test photos
ALL_SUBJECTS, ALL_PHOTOS = range(1, 41), range(1, 11)


def get_orl_root() -> Path:
    return settings.datasets_dir / "orl"


OrlRoot = Annotated[Path, Depends(get_orl_root)]


def download_orl(root: Path) -> None:
    """Make sure ORL is on disk; a failed download becomes a clear 502."""
    try:
        ensure_orl(root, settings.orl_url)
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Could not download the ORL dataset: {error}"
        ) from error


def _demo_people(db: Session) -> list[Person]:
    return list(db.scalars(select(Person).where(Person.is_demo)).all())


@router.post("", response_model=DemoLoaded)
def load_demo(db: DbSession, root: OrlRoot) -> DemoLoaded:
    if _demo_people(db):
        return DemoLoaded(people_added=0)
    download_orl(root)
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    for subject in ENROLLED_SUBJECTS:
        person = Person(name=f"ORL s{subject}", is_demo=True)
        db.add(person)
        for index in TRAIN_PHOTOS:
            pixels = load_orl_image(root, subject, index)
            name = f"orl-s{subject}-{index}.png"
            (settings.images_dir / name).write_bytes(array_png(pixels))
            rows, cols = pixels.shape
            db.add(
                FaceSample(person=person, image_path=name, box_x=0, box_y=0, box_w=cols, box_h=rows)
            )
    db.commit()
    return DemoLoaded(people_added=len(ENROLLED_SUBJECTS))


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def remove_demo(db: DbSession) -> Response:
    for person in _demo_people(db):
        for sample in person.samples:
            delete_sample_file(sample)
        db.delete(person)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/test-photos", response_model=list[TestPhoto])
def test_photos(root: OrlRoot) -> list[TestPhoto]:
    """Photos never used for training: held-out ones of enrolled people, and strangers."""
    photos = []
    for subject in ALL_SUBJECTS:
        enrolled = subject in ENROLLED_SUBJECTS
        for index in ALL_PHOTOS:
            if enrolled and index in TRAIN_PHOTOS:
                continue
            if (root / f"s{subject}" / f"{index}.pgm").exists():
                url = f"/demo/orl/photos/{subject}/{index}.png"
                photos.append(TestPhoto(subject=subject, index=index, enrolled=enrolled, url=url))
    return photos


@router.get("/photos/{subject}/{index}.png")
def photo(subject: int, index: int, root: OrlRoot) -> Response:
    if not (root / f"s{subject}" / f"{index}.pgm").exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such ORL photo")
    return Response(content=array_png(load_orl_image(root, subject, index)), media_type="image/png")
