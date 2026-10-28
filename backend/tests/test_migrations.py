import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tests.conftest import alembic_config

TABLES = {"transactions", "model_versions", "predictions"}


def _model_row(version: str, status: str) -> dict:
    return {
        "version": version,
        "algorithm": "lightgbm",
        "artifact_path": f"models/{version}",
        "feature_list": "[]",
        "policy": "expected_loss",
        "threshold": 21.82,
        "status": status,
    }


_INSERT = text(
    "INSERT INTO model_versions (version, algorithm, artifact_path, feature_list, policy,"
    " threshold, status) VALUES (:version, :algorithm, :artifact_path, CAST(:feature_list AS"
    " JSONB), :policy, :threshold, :status)"
)


def test_upgrade_creates_tables(migrated_engine):
    assert TABLES <= set(inspect(migrated_engine).get_table_names())


def test_at_most_one_active_model(db_session: Session):
    db_session.execute(_INSERT, _model_row("m-1", "active"))
    db_session.execute(_INSERT, _model_row("m-2", "candidate"))
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            db_session.execute(_INSERT, _model_row("m-3", "active"))


def test_unknown_status_rejected(db_session: Session):
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            db_session.execute(_INSERT, _model_row("m-4", "bogus"))


def test_downgrade_then_upgrade(migrated_engine, test_db_url):
    cfg = alembic_config(test_db_url)
    command.downgrade(cfg, "base")
    engine = create_engine(test_db_url)
    try:
        assert not TABLES & set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    command.upgrade(cfg, "head")
    assert TABLES <= set(inspect(migrated_engine).get_table_names())
