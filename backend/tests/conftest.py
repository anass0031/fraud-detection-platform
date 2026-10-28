"""Shared test fixtures: a separate `fraud_test` database, migrated with Alembic.

Tests never touch the demo database. The test database is dropped and re-created once per run.
"""

import os
from collections.abc import Callable, Generator
from pathlib import Path

# Fast password hashing in tests (must be set before the settings are first read).
os.environ["BCRYPT_ROUNDS"] = "4"

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.models import User

BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_DB_NAME = "fraud_test"


def alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    # Escape % so ConfigParser does not treat it as interpolation.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Create an empty `fraud_test` database on the same server and return its URL."""
    main_url = make_url(get_settings().database_url)
    test_url = main_url.set(database=TEST_DB_NAME)
    admin = create_engine(main_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {TEST_DB_NAME}"))
    admin.dispose()
    return test_url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def migrated_engine(test_db_url: str) -> Generator[Engine, None, None]:
    command.upgrade(alembic_config(test_db_url), "head")
    engine = create_engine(test_db_url)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(migrated_engine: Engine) -> Generator[Session, None, None]:
    """A session whose changes are rolled back after each test."""
    conn = migrated_engine.connect()
    trans = conn.begin()
    session = sessionmaker(bind=conn, join_transaction_mode="create_savepoint")()
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        conn.close()


TEST_PASSWORD = "correct-horse-battery"


@pytest.fixture
def make_user(db_session: Session) -> Callable[..., User]:
    """Create a user in the test database: make_user(role="analyst", email=..., active=True)."""
    counter = iter(range(1, 10_000))

    def _make(role: str = "analyst", email: str | None = None, active: bool = True) -> User:
        user = User(
            email=email or f"{role}{next(counter)}@example.com",
            full_name=f"Test {role.title()}",
            password_hash=hash_password(TEST_PASSWORD),
            role=role,
            is_active=active,
        )
        db_session.add(user)
        db_session.flush()
        return user

    return _make


@pytest.fixture
def api(db_session: Session) -> Generator:
    """TestClient whose requests use the test database session."""
    from fastapi.testclient import TestClient

    from app.db.session import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def login(client, email: str, password: str = TEST_PASSWORD):
    return client.post("/api/v1/auth/login", data={"username": email, "password": password})


def auth_header(client, user: User) -> dict[str, str]:
    resp = login(client, user.email)
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
