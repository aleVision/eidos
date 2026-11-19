"""People and their face samples (photo + hand-drawn face box)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.imaging import array_png, read_image
from app.models import FaceSample, Person
from app.schemas import Box, PersonCreate, PersonOut, SampleOut
from eidos_core.preprocess import prepare

router = APIRouter(tags=["people"])

DbSession = Annotated[Session, Depends(get_db)]


def _person_out(person: Person) -> PersonOut:
    return PersonOut(
        id=person.id, name=person.name, is_demo=person.is_demo, n_samples=len(person.samples)
    )


def _sample_out(sample: FaceSample) -> SampleOut:
    box = Box(x=sample.box_x, y=sample.box_y, w=sample.box_w, h=sample.box_h)
    return SampleOut(id=sample.id, person_id=sample.person_id, box=box)


def _get_person(db: Session, person_id: int) -> Person:
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Person not found")
    return person


def _get_sample(db: Session, sample_id: int) -> FaceSample:
    sample = db.get(FaceSample, sample_id)
    if sample is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample not found")
    return sample


def delete_sample_file(sample: FaceSample) -> None:
    (settings.images_dir / sample.image_path).unlink(missing_ok=True)


@router.get("/people", response_model=list[PersonOut])
def list_people(db: DbSession) -> list[PersonOut]:
    people = db.scalars(select(Person).order_by(Person.is_demo, Person.name)).all()
    return [_person_out(p) for p in people]


@router.post("/people", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
def create_person(body: PersonCreate, db: DbSession) -> PersonOut:
    person = Person(name=body.name)
    db.add(person)
    db.commit()
    return _person_out(person)


@router.delete("/people/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(person_id: int, db: DbSession) -> Response:
    person = _get_person(db, person_id)
    for sample in person.samples:
        delete_sample_file(sample)
    db.delete(person)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/people/{person_id}/samples", response_model=list[SampleOut])
def list_samples(person_id: int, db: DbSession) -> list[SampleOut]:
    return [_sample_out(s) for s in _get_person(db, person_id).samples]


@router.post(
    "/people/{person_id}/samples", response_model=SampleOut, status_code=status.HTTP_201_CREATED
)
def add_sample(
    person_id: int,
    image: UploadFile,
    x: Annotated[int, Form()],
    y: Annotated[int, Form()],
    w: Annotated[int, Form()],
    h: Annotated[int, Form()],
    db: DbSession,
) -> SampleOut:
    person = _get_person(db, person_id)
    try:
        pixels = read_image(image.file.read())
        prepare(pixels, (x, y, w, h), settings.gaussian_window)  # validates the box
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    # Store the upright image, so the box always refers to the pixels on disk.
    name = f"{uuid.uuid4().hex}.png"
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    (settings.images_dir / name).write_bytes(array_png(pixels))
    sample = FaceSample(person=person, image_path=name, box_x=x, box_y=y, box_w=w, box_h=h)
    db.add(sample)
    db.commit()
    return _sample_out(sample)


@router.get("/samples/{sample_id}/image")
def sample_image(sample_id: int, db: DbSession) -> FileResponse:
    sample = _get_sample(db, sample_id)
    return FileResponse(settings.images_dir / sample.image_path, media_type="image/png")


@router.delete("/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(sample_id: int, db: DbSession) -> Response:
    sample = _get_sample(db, sample_id)
    delete_sample_file(sample)
    db.delete(sample)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
