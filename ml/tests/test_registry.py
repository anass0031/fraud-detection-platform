import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from fraudml.models.calibration import IsotonicCalibrator
from fraudml.registry import load_model, next_version, save_model


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    X = pd.DataFrame(
        {
            "a": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n) + y),
            "c": pd.Categorical(rng.choice(["x", "y", "z", "(missing)"], n)),
        }
    )
    model = LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    cal = IsotonicCalibrator().fit(model.predict_proba(X)[:, 1], y)
    return X, y, model, cal


def _meta(policy="probability", threshold=0.2):
    return {"features": ["a", "c"], "policy": policy, "threshold": threshold, "params": {"x": 1}}


def test_round_trip_gives_identical_scores(tmp_path, fitted):
    X, _, model, cal = fitted
    folder = save_model(model, cal, _meta(), root=tmp_path, version="lgbm-test-1")
    loaded = load_model("lgbm-test-1", root=tmp_path)
    raw = model.predict_proba(X)[:, 1]
    assert np.max(np.abs(loaded.raw_score(X) - raw)) < 1e-12
    assert np.array_equal(loaded.predict_proba(X), cal.predict(raw))
    meta = json.loads((folder / "metadata.json").read_text("utf-8"))
    assert meta["version"] == "lgbm-test-1" and meta["library_versions"]["lightgbm"]


def test_column_order_does_not_matter(tmp_path, fitted):
    X, _, model, cal = fitted
    save_model(model, cal, _meta(), root=tmp_path, version="v")
    loaded = load_model("v", root=tmp_path)
    assert np.array_equal(loaded.raw_score(X[["c", "a"]]), loaded.raw_score(X))


def test_policies(tmp_path, fitted):
    _, _, model, cal = fitted
    save_model(model, cal, _meta("expected_loss", 100.0), root=tmp_path, version="el")
    loaded = load_model("el", root=tmp_path)
    assert loaded.is_alert([0.3, 0.6], [2000, 20]).tolist() == [True, False]
    with pytest.raises(ValueError):
        save_model(model, cal, _meta("bogus"), root=tmp_path, version="bad")


def test_version_naming_and_no_overwrite(tmp_path, fitted):
    _, _, model, cal = fitted
    day = dt.date(2026, 11, 12)
    assert next_version(tmp_path, today=day) == "lgbm-2026.11.12-1"
    (tmp_path / "lgbm-2026.11.12-1").mkdir()
    assert next_version(tmp_path, today=day) == "lgbm-2026.11.12-2"
    save_model(model, cal, _meta(), root=tmp_path, version="same")
    with pytest.raises(FileExistsError):
        save_model(model, cal, _meta(), root=tmp_path, version="same")
