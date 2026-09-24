"""Pytest configuration and fixtures."""

import os
import pytest
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vivasecuris.aiasylum.database.models import Base
from vivasecuris.aiasylum.database.session import SessionLocal


@pytest.fixture(autouse=True)
def isolate_database(tmp_path_factory, monkeypatch):
    """Point every database session at a throwaway file, never the real one.

    ``get_session()`` resolves ``SessionLocal`` from its own module globals at
    call time, so patching that one symbol redirects every caller -- including
    route modules that already did ``from ...database import get_session`` and
    call it directly rather than through a FastAPI dependency.

    Without this, any test driving the API through ``TestClient(app)`` writes
    real rows into ``data/aiasylum.db``. That is exactly how surgery runs with
    ``source_model='m'`` and dead ``pytest-of-*`` output paths ended up listed
    in the development UI, where opening one failed with "'m' was not found on
    the Hub or on disk". test_weights_routes.py's ``roots`` fixture already
    redirects the artifact directories; this is the database half of the same
    guarantee.
    """
    from vivasecuris.aiasylum.database import session as db_session

    db_file = tmp_path_factory.mktemp("testdb") / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    monkeypatch.setattr(db_session, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(db_session, "engine", engine)
    yield TestingSessionLocal
    engine.dispose()


@pytest.fixture
def test_db(isolate_database):
    """A session on the *same* isolated database every component sees.

    Binding this to ``isolate_database``'s factory rather than a private
    in-memory engine is what makes a test's own rows visible to code that opens
    its own session -- ``TestRunner.list_test_runs`` and friends. Previously
    these were two different databases, so a test could seed a row here, have
    the route read the developer's real database instead, and still pass
    whenever the autoincrement ids happened to line up.
    """
    session = isolate_database()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def db_session(test_db):
    """Provide a database session for each test."""
    yield test_db
    test_db.rollback()


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret")


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
