"""Leakage tests on the full feature pipeline."""

import pandas as pd

from fraudml import config
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES_CURATED,
    FEATURES_FULL,
    build_features_batch,
    raw_features_batch,
)


def _encoder(df):
    raw, _ = raw_features_batch(df)
    return FeatureEncoder(min_count=20).fit(raw)


def test_no_forbidden_columns_in_feature_lists():
    assert not set(FEATURES_CURATED) & config.FORBIDDEN_FEATURES
    assert not set(FEATURES_FULL) & config.FORBIDDEN_FEATURES
    assert len(set(FEATURES_CURATED)) == len(FEATURES_CURATED)


def test_output_columns_and_dtypes(synthetic):
    out = build_features_batch(synthetic, _encoder(synthetic))
    assert list(out.columns) == list(FEATURES_CURATED)
    assert out.index.equals(synthetic.index)
    for c in FEATURES_CURATED:
        if c in CATEGORICAL_FEATURES:
            assert isinstance(out[c].dtype, pd.CategoricalDtype), c
        else:
            assert out[c].dtype == "float64", c


def test_label_independence(synthetic):
    """Shuffling or flipping the fraud label must not change a single feature value."""
    enc = _encoder(synthetic)
    base = build_features_batch(synthetic, enc)
    flipped = synthetic.assign(isFraud=1 - synthetic["isFraud"])
    no_label = synthetic.drop(columns="isFraud")
    pd.testing.assert_frame_equal(base, build_features_batch(flipped, enc))
    pd.testing.assert_frame_equal(base, build_features_batch(no_label, enc))


def test_future_invariance(synthetic):
    """Removing every transaction after time t must not change features of rows at or before t."""
    enc = _encoder(synthetic)
    full = build_features_batch(synthetic, enc)
    for q in (0.3, 0.6, 0.9):
        cut = synthetic["TransactionDT"].quantile(q)
        past = synthetic[synthetic["TransactionDT"] <= cut]
        pd.testing.assert_frame_equal(full.loc[past.index], build_features_batch(past, enc))


def test_row_order_invariance(synthetic):
    enc = _encoder(synthetic)
    base = build_features_batch(synthetic, enc)
    shuffled = synthetic.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(base, build_features_batch(shuffled, enc).loc[base.index])


def test_every_curated_feature_is_documented():
    doc = (config.REPO_ROOT / "docs" / "features.md").read_text(encoding="utf-8")
    # C1..C14 and M1..M9 are documented as ranges (`C1`–`C14`, `M1`–`M9`).
    ranged = {f for f in FEATURES_CURATED if f[0] in "CM" and f[1:].isdigit()}
    assert "`C1`–`C14`" in doc and "`M1`–`M9`" in doc
    missing = [f for f in FEATURES_CURATED if f not in ranged and f"`{f}`" not in doc]
    assert missing == []
