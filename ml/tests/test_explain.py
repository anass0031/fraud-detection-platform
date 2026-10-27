import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from fraudml.explain.contributions import BASE, GROUPS, contributions, group_contributions


@pytest.fixture(scope="module")
def model_and_data():
    """Small LightGBM on columns named like the real features (masked + readable + category)."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    X = pd.DataFrame(
        {
            "amt_to_uid_mean": np.where(
                rng.random(n) < 0.2, np.nan, rng.lognormal(0, 1, n) + 2 * y
            ),
            "uid_new_device": rng.integers(0, 2, n).astype(float),
            "C1": rng.poisson(1 + 3 * y).astype(float),
            "C13": rng.poisson(2, n).astype(float),
            "D1n": rng.normal(size=n),
            "M4": pd.Categorical(rng.choice(["M0", "M1", "(missing)"], n)),
            "ProductCD": pd.Categorical(np.where(y == 1, "C", rng.choice(["W", "C", "R"], n))),
        }
    )
    model = LGBMClassifier(n_estimators=60, verbose=-1).fit(X, y)
    return model.booster_, X


def test_contributions_add_up_to_raw_score(model_and_data):
    booster, X = model_and_data
    c = contributions(booster, X)
    raw = booster.predict(X, raw_score=True)
    assert list(c.columns) == [*booster.feature_name(), BASE]
    assert np.max(np.abs(c.sum(axis=1).to_numpy() - raw)) < 1e-9


def test_grouping_preserves_totals_and_merges_masked(model_and_data):
    booster, X = model_and_data
    c = contributions(booster, X)
    g = group_contributions(c)
    assert "masked_counts" in g and "C1" not in g and "C13" not in g
    assert "masked_timing" in g and "masked_match" in g
    assert "amt_to_uid_mean" in g and "ProductCD" in g  # readable stay individual
    assert np.allclose(g["masked_counts"], c["C1"] + c["C13"])
    assert np.allclose(g.sum(axis=1), c.sum(axis=1))


def test_column_order_does_not_matter(model_and_data):
    booster, X = model_and_data
    shuffled = X[list(reversed(X.columns))]
    pd.testing.assert_frame_equal(contributions(booster, X), contributions(booster, shuffled))


@pytest.mark.filterwarnings("ignore:LightGBM binary classifier")
def test_matches_shap_tree_explainer(model_and_data):
    shap = pytest.importorskip("shap")
    booster, X = model_and_data
    sample = X.iloc[:300]
    ours = contributions(booster, sample)
    explainer = shap.TreeExplainer(booster)
    theirs = explainer.shap_values(sample)
    theirs = theirs[1] if isinstance(theirs, list) else theirs
    assert np.max(np.abs(ours.drop(columns=BASE).to_numpy() - theirs)) < 1e-6
    assert float(np.ravel(explainer.expected_value)[-1]) == pytest.approx(
        ours[BASE].iloc[0], abs=1e-6
    )


def test_groups_cover_known_masked_columns():
    from fraudml.features.pipeline import FEATURES_CURATED

    grouped = {f for members in GROUPS.values() for f in members}
    masked = {f for f in FEATURES_CURATED if f[0] in "CDM" and f[1:2].isdigit()} | {"dist1"}
    assert masked <= grouped
    assert grouped <= set(FEATURES_CURATED)


# ---------------- templates + explanation objects ----------------


def test_every_curated_feature_has_wording():
    from fraudml.explain.contributions import GROUP_OF
    from fraudml.explain.templates import TEMPLATES
    from fraudml.features.pipeline import FEATURES_CURATED

    for f in FEATURES_CURATED:
        key = GROUP_OF.get(f, f)
        assert key in TEMPLATES, f"no wording for {f}"


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("amt_to_uid_mean", 6.2, "Amount is 6.2x this customer's usual amount"),
        ("uid_new_device", 1.0, "First transaction from this device for this customer"),
        ("uid_new_device", float("nan"), "Device history unknown"),
        ("uid_n_prior", 0.0, "First transaction seen for this customer"),
        ("uid_n_prior", 1.0, "1 earlier transaction from this customer"),
        ("hour", 3.0, "Made at 03:00 (dataset clock)"),
        ("weekday", 0.0, "Made on a Monday"),
        ("amt_log", float(np.log1p(1250.0)), "Amount $1,250"),
        ("card1_freq", 0.0, "This card identifier is not seen in the training data"),
        ("card1_freq", 0.0001, "This card identifier is rare (0.01% of transactions)"),
        ("ProductCD", "C", "Product code is C"),
        ("ProductCD", "other", "Uncommon product code"),
        ("p_email", "(missing)", "Payer e-mail domain unknown"),
        ("secs_since_prev_uid", 90.0, "90 s since this customer's previous transaction"),
        (
            "amt_z_uid",
            -2.0,
            "Amount is 2.0 standard deviations below this customer's usual amounts",
        ),
    ],
)
def test_wording(key, value, expected):
    from fraudml.explain.templates import describe

    assert describe(key, value)[1] == expected


def test_strength_and_direction():
    from fraudml.explain.templates import sentence

    assert sentence("uid_new_device", 1.0, 0.9).endswith(": strongly raises risk")
    assert sentence("uid_new_device", 0.0, -0.3).endswith(": moderately lowers risk")
    assert sentence("masked_counts", None, 0.05).endswith(": slightly raises risk")


def test_explanations_are_ordered_json_ready_and_exact(model_and_data):
    import json

    from fraudml.explain.explain import explain_batch, explain_transaction

    booster, X = model_and_data
    items = explain_batch(booster, X.iloc[:50], k=3)
    raw = booster.predict(X.iloc[:50], raw_score=True)
    for item, r in zip(items, raw, strict=True):
        json.dumps(item)  # serialisable (NaN -> None)
        sizes = [abs(f["contribution"]) for f in item["factors"]]
        assert sizes == sorted(sizes, reverse=True) and len(sizes) == 3
        total = item["base_value"] + sum(f["contribution"] for f in item["factors"])
        assert total + item["other_contribution"] == pytest.approx(r, abs=1e-9)
        assert abs(item["raw_log_odds"] - r) < 1e-9
        for f in item["factors"]:
            assert "C1" not in f["text"] and "C13" not in f["text"]  # masked names never shown
            assert f["masked"] == f["key"].startswith("masked_")
    one = explain_transaction(booster, X.iloc[[0]], k=3)
    assert one == items[0]
    with pytest.raises(ValueError):
        explain_transaction(booster, X.iloc[0])


def test_explanation_includes_calibrated_probability(model_and_data):
    from fraudml.explain.explain import explain_batch
    from fraudml.models.calibration import IsotonicCalibrator

    booster, X = model_and_data
    p = booster.predict(X)
    cal = IsotonicCalibrator().fit(p, (p > 0.5).astype(int))
    item = explain_batch(booster, X.iloc[:5], calibrator=cal)[0]
    assert item["probability"] == pytest.approx(float(cal.predict(p[:1])[0]))
