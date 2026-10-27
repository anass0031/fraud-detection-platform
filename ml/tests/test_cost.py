import numpy as np
import pandas as pd
import pytest

from fraudml.evaluate import (
    CostConfig,
    cost_breakdown,
    cost_curve,
    evaluate_report,
    optimise_threshold,
)


def test_cost_breakdown_hand_computed():
    # TP (100, caught), FN (40, missed), FP, TN
    c = cost_breakdown([1, 1, 0, 0], [100, 40, 10, 10], [1, 0, 1, 0], CostConfig())
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["missed_fraud_value"] == 40
    assert c["review_cost"] == 10  # 2 alerts x $5
    assert c["friction_cost"] == 2  # 1 FP x $2
    assert c["total_cost"] == 52
    assert c["no_model_cost"] == 140
    assert c["net_savings"] == 88


def test_no_alerts_costs_exactly_the_no_model_baseline():
    c = cost_breakdown([1, 0, 1], [5, 7, 9], [0, 0, 0])
    assert c["total_cost"] == c["no_model_cost"] == 14
    assert c["net_savings"] == 0


def _random_case(seed: int, n: int = 60, n_days: int = 4):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    score = np.round(rng.random(n) + 0.4 * y, 2)  # rounding creates ties
    amount = np.round(rng.lognormal(3, 1, n), 2)
    day = rng.integers(0, n_days, n)
    return y, score, amount, day


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("capacity", [0, 2, 5, 100])
def test_optimiser_matches_brute_force(seed, capacity):
    y, score, amount, day = _random_case(seed)
    cfg = CostConfig(capacity_per_day=capacity)
    n_days = len(np.unique(day))
    best = None
    for t in [*np.unique(score), np.inf]:
        alert = score >= t
        if alert.sum() / n_days > capacity:
            continue
        cost = cost_breakdown(y, amount, alert, cfg)["total_cost"]
        if best is None or cost < best - 1e-9:
            best = cost
    res = optimise_threshold(y, score, amount, day, cfg)
    assert res.total_cost == pytest.approx(best)
    assert res.alerts_per_day <= capacity


def test_curve_cost_matches_cost_breakdown():
    y, score, amount, day = _random_case(7)
    curve = cost_curve(y, score, amount, day)
    for row in curve.sample(10, random_state=0).itertuples():
        direct = cost_breakdown(y, amount, score >= row.threshold)
        assert row.total_cost == pytest.approx(direct["total_cost"])
        assert row.tp == direct["tp"] and row.fp == direct["fp"]


def test_zero_capacity_means_no_alerts():
    y, score, amount, day = _random_case(1)
    res = optimise_threshold(y, score, amount, day, CostConfig(capacity_per_day=0))
    assert res.threshold == np.inf and res.n_alerts == 0


def test_free_reviews_use_the_full_capacity():
    # Reviews cost nothing -> alert as much as allowed: 1/day x 2 days = 2 alerts.
    y = [1, 1, 1, 1, 0, 0]
    score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    day = [0, 0, 0, 1, 1, 1]
    cfg = CostConfig(review_cost=0, friction_cost=0, capacity_per_day=1)
    res = optimise_threshold(y, score, [10] * 6, day, cfg)
    assert res.n_alerts == 2 and res.threshold == pytest.approx(0.8)
    assert res.max_alerts_per_day == 2  # mean-per-day constraint: both fall on day 0


def test_report_excluding_reuses_threshold_and_drops_rows():
    y, score, amount, day = _random_case(3, n=200)
    flag = np.zeros(len(y), dtype=bool)
    flag[:20] = True
    rep = evaluate_report(y, score, amount, day, flag)
    assert list(rep.index) == ["overall", "excl_fraud_uids"]
    assert rep.loc["overall", "n"] == 200 and rep.loc["excl_fraud_uids", "n"] == 180
    assert rep.loc["overall", "threshold"] == rep.loc["excl_fraud_uids", "threshold"]
    assert isinstance(rep, pd.DataFrame)
