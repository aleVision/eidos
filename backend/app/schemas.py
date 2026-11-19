"""Pydantic schemas: the shapes of API requests and responses.

These are separate from the database models on purpose. DB models describe
storage; schemas describe the API contract. Keeping them apart means we can
change one without forcing a change in the other.
"""

from typing import Annotated

from pydantic import BaseModel, StringConstraints


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class PersonCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class PersonOut(BaseModel):
    id: int
    name: str
    is_demo: bool
    n_samples: int


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class SampleOut(BaseModel):
    id: int
    person_id: int
    box: Box
