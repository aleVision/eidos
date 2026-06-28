"""Pydantic schemas: the shapes of API requests and responses.

These are separate from the database models on purpose. DB models describe
storage; schemas describe the API contract. Keeping them apart means we can
change one without forcing a change in the other.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
