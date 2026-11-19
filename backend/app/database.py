"""Database setup: engine, session factory, and the declarative base.

Uses SQLAlchemy 2.0 style. SQLite is a single local file, which suits a
local-first product. Because the data access goes through SQLAlchemy, moving to
Postgres later would not require rewriting the models.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# check_same_thread is a SQLite-specific setting required when the connection
# is shared across threads, as it is under a web server.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class all ORM models inherit from."""


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, closing it when the request finishes.

    Used as a FastAPI dependency so endpoints receive a ready session and never
    have to manage its lifecycle by hand.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
