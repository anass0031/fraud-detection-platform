import json
import shutil
from pathlib import Path

import pytest
from fraudml.registry import MODELS_DIR
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ModelVersion
from app.services.model_store import (
    ENCODER_PATH,
    ModelStore,
    RegistryError,
    activate_version,
    register,
)

MODEL = "lgbm-2026.10.27-1"
ANOMALY = "iforest-2026.10.27-1"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / MODEL).exists() or not (MODELS_DIR / ANOMALY).exists(),
    reason="trained artifacts not available (see README: train the models first)",
)


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    """Copy of the real artifacts + a second LightGBM version, in a temporary folder."""
    for v in (MODEL, ANOMALY):
        shutil.copytree(MODELS_DIR / v, tmp_path / v)
    shutil.copytree(MODELS_DIR / MODEL, tmp_path / "lgbm-test-2")
    return tmp_path


def _active(db: Session) -> list[str]:
    return list(db.scalars(select(ModelVersion.version).where(ModelVersion.status == "active")))


def test_register_and_activate(db_session: Session, models_root: Path):
    row = register(db_session, MODEL, ANOMALY, activate=True, root=models_root)
    assert row.status == "active"
    assert row.policy == "expected_loss"
    assert row.threshold == pytest.approx(21.82, abs=0.01)
    assert row.anomaly_version == ANOMALY
    assert len(row.feature_list) == 59
    assert row.artifact_path == f"models/{MODEL}"
    assert row.metrics["validation"]["overall"]["pr_auc"] > 0.3
    assert _active(db_session) == [MODEL]


def test_activating_another_version_archives_the_old_one(db_session: Session, models_root):
    register(db_session, MODEL, ANOMALY, activate=True, root=models_root)
    register(db_session, "lgbm-test-2", ANOMALY, root=models_root)
    assert _active(db_session) == [MODEL]
    activate_version(db_session, "lgbm-test-2")
    assert _active(db_session) == ["lgbm-test-2"]
    old = db_session.scalar(select(ModelVersion).where(ModelVersion.version == MODEL))
    assert old.status == "archived"


def test_register_twice_is_refused(db_session: Session, models_root):
    register(db_session, MODEL, ANOMALY, root=models_root)
    with pytest.raises(RegistryError, match="already registered"):
        register(db_session, MODEL, ANOMALY, root=models_root)


def test_missing_artifact_is_refused(db_session: Session, models_root):
    with pytest.raises(RegistryError, match="no artifact folder"):
        register(db_session, "lgbm-does-not-exist", root=models_root)
    with pytest.raises(RegistryError, match="no artifact folder"):
        register(db_session, MODEL, "iforest-does-not-exist", root=models_root)


def test_other_encoder_is_refused(db_session: Session, models_root, tmp_path):
    other = tmp_path / "other_encoder.json"
    other.write_text(ENCODER_PATH.read_text("utf-8") + " ", "utf-8")
    with pytest.raises(RegistryError, match="another feature encoder"):
        register(db_session, MODEL, root=models_root, encoder_path=other)


def test_other_library_version_is_refused(db_session: Session, models_root):
    meta_path = models_root / "lgbm-test-2" / "metadata.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["library_versions"]["lightgbm"] = "0.0.1"
    meta_path.write_text(json.dumps(meta), "utf-8")
    with pytest.raises(RegistryError, match="lightgbm 0.0.1"):
        register(db_session, "lgbm-test-2", root=models_root)


def test_store_loads_the_active_bundle(db_session: Session, models_root):
    store = ModelStore(root=models_root)
    assert store.load_active(db_session) is None
    assert store.error == "no active model in model_versions"

    register(db_session, MODEL, ANOMALY, activate=True, root=models_root)
    bundle = store.load_active(db_session)
    assert bundle is not None and store.error is None
    assert bundle.version == MODEL
    assert bundle.anomaly_version == ANOMALY
    assert bundle.model.metadata["threshold"] == pytest.approx(21.82, abs=0.01)


def test_store_keeps_error_when_artifact_disappears(db_session: Session, models_root):
    register(db_session, MODEL, ANOMALY, activate=True, root=models_root)
    shutil.rmtree(models_root / MODEL)
    store = ModelStore(root=models_root)
    assert store.load_active(db_session) is None
    assert "no artifact folder" in store.error
