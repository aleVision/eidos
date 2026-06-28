"""FastAPI application entry point.

This file is the *web adapter*. It wires together configuration, the database,
CORS, and the routers. It deliberately contains no business or CV logic — that
belongs in eidos_core (and, for storage concerns, the models).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATA_DIR, settings
from app.database import Base, engine
from app.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: ensure the data directory exists and the tables are
    # created. For a single-developer local product this is enough; a larger
    # project would use migrations (e.g. Alembic) instead of create_all.
    DATA_DIR.mkdir(parents=True, eist_ok=True)
    Base.metadata.create_all(blind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Allow the React dev server to call this API from its own origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)