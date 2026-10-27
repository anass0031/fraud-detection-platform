import numpy as np
import pandas as pd
import pytest

from fraudml.features.encoder import MISSING, OTHER, FeatureEncoder
from fraudml.features.pipeline import raw_features_batch


def _fit(df, min_count=20):
    raw, _ = raw_features_batch(df)
    return FeatureEncoder(min_count=min_count).fit(raw), raw


def test_rare_unseen_and_missing_categories(synthetic):
    enc, raw = _fit(synthetic)
    assert enc.categories["card4"] == ["mastercard", "visa", OTHER, MISSING]  # discover is rare
    new = raw.head(3).copy()
    new["browser_family"] = ["chrome", "brand_new_browser", np.nan]
    out = enc.transform(new)
    assert out["browser_family"].tolist() == ["chrome", OTHER, MISSING]
    assert list(out["browser_family"].cat.categories) == enc.categories["browser_family"]


def test_frequency_encoding(synthetic):
    enc, raw = _fit(synthetic)
    out = enc.transform(raw)
    card1_share = synthetic["card1"].value_counts(normalize=True)
    expected = synthetic["card1"].map(card1_share).astype("float64")
    np.testing.assert_allclose(out["card1_freq"], expected)
    new = raw.head(2).copy()
    new["card1"] = [99999.0, np.nan]  # unseen card -> 0, missing -> NaN
    f = enc.transform(new)["card1_freq"]
    assert f.iloc[0] == 0.0 and np.isnan(f.iloc[1])


def test_save_load_roundtrip(synthetic, tmp_path):
    enc, raw = _fit(synthetic)
    path = tmp_path / "enc.json"
    enc.save(path)
    loaded = FeatureEncoder.load(path)
    pd.testing.assert_frame_equal(enc.transform(raw), loaded.transform(raw))


def test_unfitted_encoder_raises(synthetic):
    raw, _ = raw_features_batch(synthetic)
    with pytest.raises(RuntimeError):
        FeatureEncoder().transform(raw)


def test_raw_value_named_other_merges_into_other_bucket(synthetic):
    raw, _ = raw_features_batch(synthetic)
    raw["browser_family"] = "other"  # frequent raw value literally called "other"
    enc = FeatureEncoder(min_count=20).fit(raw)
    cats = enc.categories["browser_family"]
    assert cats.count(OTHER) == 1
    assert enc.transform(raw.head(2))["browser_family"].tolist() == [OTHER, OTHER]
