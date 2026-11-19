"""FastAPI application entry point.

This file is the *web adapter*. It wires together configuration, the database,
CORS, and the routers. It deliberately contains no business or CV logic — that
belongs in eidos_core (and, for storage concerns, the models).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importing models registers their tables on Base.metadata. Without this import,
# create_all() below sees no tables and the database comes up empty. The
# "noqa: F401" tells ruff this import is intentional even though it looks unused.
from app import models  # noqa: F401
from app.config import DATA_DIR, settings
from app.database import Base, engine
from app.routers import health, people


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runs once at startup: ensure the data directory exists and the tables are
    # created. For a single-developer local product this is enough; a larger
    # project would use migrations (e.g. Alembic) instead of create_all.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (settings.images_dir, settings.models_dir, settings.datasets_dir):
        folder.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield
    # Nothing to tear down yet.


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Allow the React dev server to call this API from its own origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(people.router)
