"""Pydantic schemas: the shapes of API requests and responses.

These are separate from the database models on purpose. DB models describe
storage; schemas describe the API contract. Keeping them apart means we can
change one without forcing a change in the other.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, PositiveFloat, PositiveInt, StringConstraints


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


class TrainRequest(BaseModel):
    n_components: PositiveInt | None = None


class ThresholdsIn(BaseModel):
    person: PositiveFloat
    face_space: PositiveFloat


class ModelOut(BaseModel):
    id: int
    created_at: datetime
    n_people: int
    n_samples: int
    n_components: int
    theta_person: float
    theta_face: float
    explained_variance: list[float]
    stale: bool


class PersonDistance(BaseModel):
    person_id: int
    name: str
    distance: float


class ThresholdsOut(BaseModel):
    person: float
    face_space: float


class RecognitionOut(BaseModel):
    outcome: str  # "known" | "unknown_face" | "not_a_face"
    quadrant: int  # 1-4, as numbered by Turk & Pentland
    nearest: PersonDistance
    face_space_distance: float
    distances: list[PersonDistance]
    thresholds: ThresholdsOut
    weights: list[float]
    input_png: str  # data URL of the prepared face Γ
    projection_png: str  # data URL of its projection onto face space
    stale: bool


class DemoLoaded(BaseModel):
    people_added: int


class TestPhoto(BaseModel):
    subject: int
    index: int
    enrolled: bool
    url: str


class DetectorStatus(BaseModel):
    available: bool


class DetectorOut(BaseModel):
    threshold: float  # normalized method
    paper_threshold: float  # the paper's raw ε
    n_components: int


class DetectionOut(BaseModel):
    box: Box
    distance: float  # ε at the suggested box
    confident: bool  # ε below the detector's threshold
    map_png: str  # data URL of the face map, sized like the photo
