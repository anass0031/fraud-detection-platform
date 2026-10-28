import json

import numpy as np
import pandas as pd
import pytest

from fraudml.monitoring.drift import (
    SCORE,
    build_feature_reference,
    build_reference,
    load_reference,
    measure,
    psi,
    severity,
)

rng = np.random.default_rng(42)
NORMAL = rng.normal(0, 1, 20_000)


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    # Textbook example: 10 equal bins vs a shifted distribution.
    assert psi([0.1] * 10, [0.2] * 5 + [0.0] * 5) > 1
    # Symmetric in the sense that both directions are positive.
    assert psi([0.3, 0.7], [0.7, 0.3]) == pytest.approx(psi([0.7, 0.3], [0.3, 0.7]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "stable"),
        (0.0999, "stable"),
        (0.1, "moderate"),
        (0.25, "moderate"),
        (0.2501, "significant"),
    ],
)
def test_severity_bands(value, expected):
    assert severity(value) == expected


def test_numeric_reference_uses_deciles_and_a_missing_bin():
    values = np.append(NORMAL, [np.nan] * 2_000)
    ref = build_feature_reference("x", values, "numeric")
    assert len(ref.edges) == 9
    assert len(ref.expected) == 11  # 10 bins + missing
    assert ref.expected[-1] == pytest.approx(2_000 / 22_000)
    assert sum(ref.expected) == pytest.approx(1)
    assert len(ref.sample) == 2000


def test_same_distribution_is_stable_and_shift_is_significant():
    ref = build_feature_reference("x", NORMAL, "numeric")
    same = measure(ref, rng.normal(0, 1, 3_000))
    assert same.severity == "stable"
    assert same.psi < 0.02
    assert same.ks_pvalue > 0.01

    shifted = measure(ref, rng.normal(1, 1, 3_000))
    assert shifted.severity == "significant"
    assert shifted.ks_stat > 0.3


def test_more_missing_values_is_drift():
    ref = build_feature_reference("x", NORMAL, "numeric")
    half_missing = np.where(rng.random(3_000) < 0.5, np.nan, rng.normal(0, 1, 3_000))
    assert measure(ref, half_missing).severity == "significant"


def test_heavy_ties_merge_bins():
    values = np.concatenate([np.zeros(9_000), np.arange(1, 1_001)])
    ref = build_feature_reference("x", values, "numeric")
    assert len(ref.edges) < 9  # many quantiles are 0
    assert measure(ref, values).psi == pytest.approx(0, abs=1e-9)


def test_categorical_with_unseen_values():
    train = pd.Series(["visa"] * 600 + ["mastercard"] * 300 + [None] * 100)
    ref = build_feature_reference("card4", train, "categorical")
    assert ref.categories == ["(missing)", "mastercard", "visa"]
    assert measure(ref, train).psi == pytest.approx(0, abs=1e-12)

    later = pd.Series(["visa"] * 300 + ["discover"] * 700)
    result = measure(ref, later)
    assert result.severity == "significant"
    assert result.ks_stat is None  # KS only for numeric features


def test_empty_window():
    ref = build_feature_reference("x", NORMAL, "numeric")
    assert measure(ref, np.array([])).n == 0


def test_whole_reference_round_trips_through_json():
    df = pd.DataFrame({"amt": NORMAL[:5_000], "card": ["visa", "amex"] * 2_500})
    data = build_reference(df, {"card"}, rng.random(5_000), meta={"model": "m1"})
    restored = load_reference(json.loads(json.dumps(data)))
    assert set(restored) == {"amt", "card", SCORE}
    assert restored["card"].kind == "categorical"
    assert restored[SCORE].kind == "numeric"
    assert measure(restored["amt"], NORMAL[:5_000]).psi == pytest.approx(0, abs=1e-9)
