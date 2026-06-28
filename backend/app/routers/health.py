"""Health endpoint.

A liveness check the frontend (and later, monitoring or CI) can hit to confirm
the backend is up. It's the simplest possible endpoint, which makes it the right
first one: if this works end to end, the whole pipe is wired correctly.
"""

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version
    )
