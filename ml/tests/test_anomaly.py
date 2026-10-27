import numpy as np
import pandas as pd
import pytest

from fraudml.models.anomaly import anomaly_score, fit_cutoff, make_iforest, safety_net


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "b": rng.exponential(size=n),
            "c": pd.Categorical(
                rng.choice(["x", "y", "(missing)"], n), categories=["x", "y", "z", "(missing)"]
            ),
        }
    )
    X.loc[:9, "b"] = 50.0  # 10 obvious outliers
    return X


def test_handles_nan_and_categoricals_and_finds_outliers():
    X = _frame()
    model = make_iforest(X, n_estimators=100).fit(X)
    s = anomaly_score(model, X)
    assert np.isfinite(s).all()
    assert s[:10].mean() > np.quantile(s, 0.95)  # outliers score high


def test_deterministic_with_seed():
    X = _frame()
    s1 = anomaly_score(make_iforest(X, seed=3, n_estimators=50).fit(X), X)
    s2 = anomaly_score(make_iforest(X, seed=3, n_estimators=50).fit(X), X)
    assert np.array_equal(s1, s2)


def test_unseen_category_and_all_missing_row_do_not_crash():
    X = _frame()
    model = make_iforest(X, n_estimators=50).fit(X)
    new = pd.DataFrame(
        {
            "a": [np.nan],
            "b": [np.nan],
            "c": pd.Categorical(["z"], categories=["x", "y", "z", "(missing)"]),
        }
    )
    assert np.isfinite(anomaly_score(model, new)).all()


def test_cutoff_flags_top_share():
    s = np.random.default_rng(1).normal(size=100_000)
    cut = fit_cutoff(s, 0.005)
    assert (s >= cut).mean() == pytest.approx(0.005, abs=0.0002)


def test_safety_net_never_flags_main_alerts():
    main = np.array([True, False, False, True])
    score = np.array([9.0, 9.0, 0.1, 0.1])
    assert safety_net(main, score, 1.0).tolist() == [False, True, False, False]


def test_anomaly_artifact_round_trip(tmp_path):
    from fraudml.registry import load_anomaly, save_anomaly

    X = _frame()
    model = make_iforest(X, n_estimators=50).fit(X)
    s = anomaly_score(model, X)
    cut = fit_cutoff(s)
    meta = {"features": ["a", "b", "c"], "cutoff": cut, "top_share": 0.005}
    folder = save_anomaly(model, meta, root=tmp_path, version="iforest-test-1")
    loaded = load_anomaly(folder.name, root=tmp_path)
    assert np.array_equal(loaded.score(X[["c", "b", "a"]]), s)
    assert loaded.is_unusual(X).sum() == (s >= cut).sum()
    with pytest.raises(ValueError):
        save_anomaly(model, {"features": ["a"]}, root=tmp_path, version="bad")


def test_safety_net_metrics_hand_computed():
    from fraudml.models.anomaly import safety_net_metrics

    # 4 rows, 2 days: fraud 0 caught by main; fraud 1 caught only by the safety net;
    # row 2 is an extra false alarm; row 3 legit, not alerted.
    y = [1, 1, 0, 0]
    amount = [100.0, 50.0, 10.0, 10.0]
    day = [0, 0, 1, 1]
    main = [True, False, False, False]
    extra = [False, True, True, False]
    r = safety_net_metrics(y, amount, day, main, extra)
    assert r["extra_frauds"] == 1 and r["extra_precision"] == 0.5
    assert r["extra_alerts_per_day"] == 1.0 and r["share_of_missed_frauds_caught"] == 1.0
    assert r["extra_fraud_value_caught"] == pytest.approx(50 / 150)
    # +$50 fraud caught, 2 extra reviews ($10), 1 extra false alarm ($2) -> +$38
    assert r["delta_net_savings"] == pytest.approx(38.0)
