"""Shared test fixtures: a separate `fraud_test` database, migrated with Alembic.

Tests never touch the demo database. The test database is dropped and re-created once per run.
"""

import datetime as dt
import itertools
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
from app.db.models import ModelVersion, Prediction, Transaction, User

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


@pytest.fixture
def make_prediction(db_session: Session) -> Callable[..., Prediction]:
    """A stored transaction + prediction without model artifacts (for queue tests).

    make_prediction(amount=100.0, score=0.3, alert=True, unusual=False, when=datetime,
    customer="c1", ground_truth=None, product="W", risk="high") -> Prediction.
    """
    ids = itertools.count(1_000_000)
    model: list[ModelVersion] = []

    def _model() -> ModelVersion:
        if not model:
            mv = ModelVersion(
                version="test-model",
                algorithm="lightgbm",
                artifact_path="unused",
                feature_list=["x"],
                policy="expected_loss",
                threshold=20.0,
                status="active",
            )
            db_session.add(mv)
            db_session.flush()
            model.append(mv)
        return model[0]

    def _make(
        amount: float = 100.0,
        score: float = 0.3,
        alert: bool = True,
        unusual: bool = False,
        when: dt.datetime = dt.datetime(2018, 4, 28, 12, 0),
        customer: str | None = "c1",
        ground_truth: bool | None = None,
        product: str = "W",
        risk: str = "high",
    ) -> Prediction:
        ext = next(ids)
        txn = Transaction(
            external_id=ext,
            transaction_dt=int((when - dt.datetime(2017, 12, 1)).total_seconds()),
            transaction_time=when,
            amount=amount,
            product_code=product,
            customer_key=customer,
            raw={"TransactionID": ext, "TransactionAmt": amount},
            ground_truth=ground_truth,
            source="batch",
        )
        db_session.add(txn)
        db_session.flush()
        pred = Prediction(
            transaction_id=txn.id,
            model_version_id=_model().id,
            fraud_score=score,
            expected_loss=score * amount,
            is_alert=alert,
            anomaly_score=0.7 if unusual else 0.1,
            is_unusual=unusual,
            risk_level=risk,
            shap_values={"base_value": 0.03, "factors": []},
            base_value=0.03,
        )
        db_session.add(pred)
        db_session.flush()
        return pred

    return _make
