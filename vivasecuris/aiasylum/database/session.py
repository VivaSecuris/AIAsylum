"""Database session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from vivasecuris.aiasylum.database.models import Base


# Create engine with larger connection pool
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
    pool_size=20,  # Increase pool size
    max_overflow=40,  # Increase overflow
    pool_pre_ping=True,  # Verify connections before using
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Get a database session."""
    return SessionLocal()


def init_db():
    """Initialize the database (create tables)."""
    Base.metadata.create_all(bind=engine)
