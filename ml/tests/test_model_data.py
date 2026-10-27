import numpy as np
import pandas as pd
import pytest

from fraudml import config
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.pipeline import FEATURES_CURATED, build_features_batch, raw_features_batch
from fraudml.models.baselines import AmountRule, make_logistic
from fraudml.models.data import load_period, prepare


@pytest.fixture
def feature_frame(synthetic):
    raw, _ = raw_features_batch(synthetic)
    enc = FeatureEncoder(min_count=20).fit(raw)
    feats = build_features_batch(synthetic, enc)
    feats[config.EVAL_FLAG_COL] = False
    for col in (config.ID_COL, config.TIME_COL, config.TARGET):
        feats[col] = synthetic.loc[feats.index, col].to_numpy()
    return feats


def test_prepare_splits_x_y_meta(feature_frame, synthetic):
    d = prepare(feature_frame, synthetic[[config.ID_COL, "TransactionAmt"]])
    assert list(d.X.columns) == list(FEATURES_CURATED)
    assert not set(d.X.columns) & config.FORBIDDEN_FEATURES
    assert len(d.X) == len(d.y) == len(d.meta)
    # exact amount joined by id, day = seconds // 86400
    amt = synthetic.set_index(config.ID_COL)["TransactionAmt"]
    assert np.allclose(d.meta["amount"], d.meta["TransactionID"].map(amt))
    assert (d.meta["day"] == feature_frame[config.TIME_COL] // 86_400).all()


def test_prepare_rejects_forbidden_features(feature_frame, synthetic):
    with pytest.raises(ValueError, match="forbidden"):
        prepare(feature_frame, synthetic[[config.ID_COL, "TransactionAmt"]], ("hour", "uid"))


def test_prepare_rejects_missing_amounts(feature_frame, synthetic):
    with pytest.raises(ValueError, match="without TransactionAmt"):
        prepare(feature_frame, synthetic[[config.ID_COL, "TransactionAmt"]].iloc[5:])


def test_test_period_is_refused():
    with pytest.raises(PermissionError):
        load_period("test")


def test_amount_rule():
    rule = AmountRule().fit(pd.Series(np.arange(1, 101, dtype=float)))
    s = rule.score([1.0, 50.0, 100.0])
    assert (s >= rule.RULE_THRESHOLD).tolist() == [False, False, True]
    assert s[0] < s[1]  # amount breaks ties among unflagged rows


def test_logistic_fits_on_features_with_nan(feature_frame, synthetic):
    d = prepare(feature_frame, synthetic[[config.ID_COL, "TransactionAmt"]])
    assert d.X.isna().any().any()  # real NaNs are present
    model = make_logistic(d.X).fit(d.X, d.y)
    p = model.predict_proba(d.X)[:, 1]
    assert p.shape == (len(d.X),) and np.isfinite(p).all()


def test_join_raw_columns_matches_by_id():
    from fraudml.models.data import join_raw_columns

    feats = pd.DataFrame({config.ID_COL: [3, 1, 2]}, index=[10, 11, 12])
    raw = pd.DataFrame({config.ID_COL: [1, 2, 3], "V1": [0.1, 0.2, np.nan]})
    out = join_raw_columns(feats, raw, ["V1"])
    assert list(out.index) == [10, 11, 12]
    assert np.isnan(out.loc[10, "V1"]) and out.loc[11, "V1"] == np.float32(0.1)
    assert out["V1"].dtype == "float32"
    assert join_raw_columns(feats, raw, []) is feats
