"""Model registry in the database + the active scoring bundle held in memory by the API.

A scoring bundle = one `model_versions` row: the LightGBM model, its calibrator and alert policy
(artifacts/models/<version>/), the linked Isolation Forest (`anomaly_version`) and the feature
encoder they were trained with. Artifacts are produced offline by the ml/ scripts; this module
only registers, activates and loads them.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import sklearn
from fastapi import Request
from fraudml import config
from fraudml.features.encoder import FeatureEncoder
from fraudml.registry import (
    MODELS_DIR,
    LoadedAnomaly,
    LoadedModel,
    file_sha256,
    load_anomaly,
    load_model,
)
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import ModelVersion, User
from app.services import audit

ENCODER_PATH = config.ARTIFACTS_DIR / "feature_encoder.json"


class RegistryError(RuntimeError):
    """An artifact cannot be registered or loaded safely."""


@dataclass
class ScoringBundle:
    model_version_id: int
    version: str
    model: LoadedModel
    anomaly: LoadedAnomaly | None
    encoder: FeatureEncoder
    loaded_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def anomaly_version(self) -> str | None:
        return self.anomaly.metadata["version"] if self.anomaly else None


def _check_compatible(metadata: dict, encoder_path: Path) -> None:
    """Refuse artifacts trained with another encoder or other library versions."""
    version = metadata["version"]
    expected = metadata.get("encoder_sha256")
    actual = file_sha256(encoder_path)
    if expected and expected != actual:
        raise RegistryError(f"{version}: trained with another feature encoder ({encoder_path})")
    libs = metadata.get("library_versions", {})
    running = {"lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__}
    for lib, saved in libs.items():
        if lib in running and saved != running[lib]:
            raise RegistryError(f"{version}: saved with {lib} {saved}, running {running[lib]}")


def load_artifacts(
    version: str,
    anomaly_version: str | None,
    root: Path = MODELS_DIR,
    encoder_path: Path = ENCODER_PATH,
) -> tuple[LoadedModel, LoadedAnomaly | None, FeatureEncoder]:
    if not (root / version / "metadata.json").exists():
        raise RegistryError(f"no artifact folder {root / version}")
    model = load_model(version, root)
    _check_compatible(model.metadata, encoder_path)
    anomaly = None
    if anomaly_version:
        if not (root / anomaly_version / "metadata.json").exists():
            raise RegistryError(f"no artifact folder {root / anomaly_version}")
        try:
            anomaly = load_anomaly(anomaly_version, root)
        except RuntimeError as exc:  # scikit-learn version guard in fraudml.registry
            raise RegistryError(f"{anomaly_version}: {exc}") from exc
        _check_compatible(anomaly.metadata, encoder_path)
        missing = set(anomaly.metadata["features"]) - set(model.features)
        if missing:
            raise RegistryError(f"{anomaly_version} needs features the model lacks: {missing}")
    return model, anomaly, FeatureEncoder.load(encoder_path)


def register(
    db: Session,
    version: str,
    anomaly_version: str | None = None,
    activate: bool = False,
    root: Path = MODELS_DIR,
    encoder_path: Path = ENCODER_PATH,
) -> ModelVersion:
    """Add a model_versions row for an artifact folder (loaded once to prove it works)."""
    if db.scalar(select(ModelVersion).where(ModelVersion.version == version)) is not None:
        raise RegistryError(f"{version} is already registered")
    model, anomaly, _ = load_artifacts(version, anomaly_version, root, encoder_path)
    meta = model.metadata
    created = meta.get("created_at")
    row = ModelVersion(
        version=version,
        algorithm=meta["algorithm"],
        artifact_path=f"models/{version}",
        anomaly_version=anomaly_version,
        feature_list=list(meta["features"]),
        hyperparams=meta.get("params"),
        metrics={
            "validation": meta.get("validation_metrics"),
            "anomaly_validation": anomaly.metadata.get("validation_metrics") if anomaly else None,
            # Champion vs challenger on a recent untouched week (written by the retrain script).
            "comparison": meta.get("comparison"),
        },
        calibration_method=meta.get("calibration_method"),
        policy=meta["policy"],
        threshold=float(meta["threshold"]),
        status="candidate",
        trained_at=dt.datetime.fromisoformat(created) if created else None,
        training_data_window={
            "train": meta.get("training_data_window"),
            "calibration": meta.get("calibration_window"),
        },
        notes=meta.get("notes"),
    )
    db.add(row)
    db.flush()
    audit.record(
        db,
        "model.register",
        entity_type="model_version",
        entity_id=row.id,
        after={
            "version": version,
            "anomaly_version": anomaly_version,
            "policy": row.policy,
            "threshold": row.threshold,
            "status": row.status,
            "via": "register_model script",
        },
    )
    if activate:
        activate_version(db, version)
    return row


def activate_version(
    db: Session, version: str, actor: User | None = None, request: Request | None = None
) -> ModelVersion:
    """Make `version` the only active bundle (the previous active one is archived)."""
    row = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if row is None:
        raise RegistryError(f"{version} is not registered")
    if row.status == "active":
        return row
    previous = active_version(db)
    before = {"status": row.status, "previous_active": previous.version if previous else None}
    db.execute(
        update(ModelVersion).where(ModelVersion.status == "active").values(status="archived")
    )
    row.status = "active"
    db.flush()
    audit.record(
        db,
        "model.activate",
        actor=actor,
        entity_type="model_version",
        entity_id=row.id,
        before=before,
        after={
            "status": "active",
            "version": row.version,
            "previous_archived": before["previous_active"],
        },
        request=request,
    )
    return row


def archive_version(
    db: Session, row: ModelVersion, actor: User, request: Request | None = None
) -> ModelVersion:
    """Candidate -> archived (audited). The active model cannot be archived: activate another."""
    if row.status == "active":
        raise RegistryError("The active model cannot be archived; activate another one first")
    if row.status == "archived":
        return row
    row.status = "archived"
    db.flush()
    audit.record(
        db,
        "model.archive",
        actor=actor,
        entity_type="model_version",
        entity_id=row.id,
        before={"status": "candidate"},
        after={"status": "archived", "version": row.version},
        request=request,
    )
    return row


def active_version(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))


class ModelStore:
    """Holds the active scoring bundle for the API process."""

    def __init__(self, root: Path = MODELS_DIR, encoder_path: Path = ENCODER_PATH):
        self.root = root
        self.encoder_path = encoder_path
        self.bundle: ScoringBundle | None = None
        self.error: str | None = "not loaded yet"
        self._lock = threading.Lock()

    def prepare(self, row: ModelVersion) -> ScoringBundle:
        """Load a version's artifacts without touching the bundle being served.
        Raises RegistryError / OSError if the artifacts are missing or incompatible."""
        model, anomaly, encoder = load_artifacts(
            row.version, row.anomaly_version, self.root, self.encoder_path
        )
        return ScoringBundle(row.id, row.version, model, anomaly, encoder)

    def swap(self, bundle: ScoringBundle) -> None:
        """Serve `bundle` from now on (a single reference assignment: requests use either
        the old or the new bundle, never a mix)."""
        with self._lock:
            self.bundle, self.error = bundle, None

    def load_active(self, db: Session) -> ScoringBundle | None:
        """(Re)load the active bundle. On failure keep the error and no bundle."""
        with self._lock:
            row = active_version(db)
            if row is None:
                self.bundle, self.error = None, "no active model in model_versions"
                return None
            try:
                model, anomaly, encoder = load_artifacts(
                    row.version, row.anomaly_version, self.root, self.encoder_path
                )
            except Exception as exc:  # any loader error: the API starts without a model
                self.bundle, self.error = None, f"{type(exc).__name__}: {exc}"
                return None
            self.bundle = ScoringBundle(row.id, row.version, model, anomaly, encoder)
            self.error = None
            return self.bundle


store = ModelStore()


def get_model_store() -> ModelStore:
    """FastAPI dependency (overridden in tests)."""
    return store
