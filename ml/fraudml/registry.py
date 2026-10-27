"""File-based model artifacts (L05). L08 registers these folders in the `model_versions` table.

artifacts/models/<version>/
    model.txt        LightGBM native text model (no pickle: portable and readable)
    calibrator.json  isotonic breakpoints
    metadata.json    version, features, params, alert policy + threshold, costs, metrics, ...
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from fraudml import config
from fraudml.models.calibration import IsotonicCalibrator, expected_loss

MODELS_DIR = config.ARTIFACTS_DIR / "models"
POLICIES = ("probability", "expected_loss")


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def next_version(root: Path, algorithm: str = "lgbm", today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    base = f"{algorithm}-{today:%Y.%m.%d}"
    n = 1
    while (root / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


@dataclass
class LoadedModel:
    booster: lgb.Booster
    calibrator: IsotonicCalibrator
    metadata: dict

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.raw_score(X))

    def alert_score(self, prob, amount) -> np.ndarray:
        """The score the alert threshold applies to (depends on the chosen policy)."""
        if self.metadata["policy"] == "expected_loss":
            return expected_loss(prob, amount)
        return np.asarray(prob, dtype="float64")

    def is_alert(self, prob, amount) -> np.ndarray:
        return self.alert_score(prob, amount) >= self.metadata["threshold"]


def save_model(
    model: lgb.LGBMClassifier,
    calibrator: IsotonicCalibrator,
    metadata: dict,
    root: Path = MODELS_DIR,
    version: str | None = None,
) -> Path:
    for key in ("features", "policy", "threshold"):
        if key not in metadata:
            raise ValueError(f"metadata needs {key!r}")
    if metadata["policy"] not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    version = version or next_version(root)
    folder = root / version
    folder.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(folder / "model.txt"))
    (folder / "calibrator.json").write_text(json.dumps(calibrator.to_dict()), "utf-8")
    meta = {
        "version": version,
        "algorithm": "lightgbm",
        "calibration_method": "isotonic",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "encoder_sha256": file_sha256(config.ARTIFACTS_DIR / "feature_encoder.json"),
        "library_versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        **metadata,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=float), "utf-8")
    return folder


def load_model(version: str, root: Path = MODELS_DIR) -> LoadedModel:
    folder = root / version
    booster = lgb.Booster(model_file=str(folder / "model.txt"))
    calibrator = IsotonicCalibrator.from_dict(
        json.loads((folder / "calibrator.json").read_text("utf-8"))
    )
    metadata = json.loads((folder / "metadata.json").read_text("utf-8"))
    return LoadedModel(booster, calibrator, metadata)
