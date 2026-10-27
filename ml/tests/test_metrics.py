import numpy as np
import pytest

from fraudml.evaluate import (
    brier,
    daily_topk_mask,
    fraud_value_caught,
    pr_auc,
    precision_at_capacity,
    recall_at_capacity,
    recall_at_precision,
    roc_auc,
)


def test_pr_auc_perfect_and_worst():
    y = [1, 0, 0, 0]
    assert pr_auc(y, [3, 2, 1, 0]) == pytest.approx(1.0)
    # Fraud ranked last: precision at full recall = 1/4.
    assert pr_auc(y, [0, 1, 2, 3]) == pytest.approx(0.25)
    assert roc_auc(y, [0, 1, 2, 3]) == pytest.approx(0.0)


def test_pr_auc_random_is_close_to_fraud_rate():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.035
    assert pr_auc(y, rng.random(len(y))) == pytest.approx(0.035, abs=0.005)


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_recall_at_precision():
    y = [1, 1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert recall_at_precision(y, s, 0.5) == pytest.approx(1.0)  # top 4: precision 0.75
    assert recall_at_precision(y, s, 0.8) == pytest.approx(2 / 3)  # top 2: precision 1
    assert recall_at_precision([0, 1], [1.0, 0.0], 0.99) == pytest.approx(0.0)


def test_daily_topk_mask_ranks_within_each_day():
    day = [0, 0, 0, 1, 1]
    s = [0.1, 0.9, 0.5, 0.3, 0.2]
    assert daily_topk_mask(s, day, 1).tolist() == [False, True, False, True, False]
    assert daily_topk_mask(s, day, 2).tolist() == [False, True, True, True, True]


def test_daily_topk_ties_keep_first():
    assert daily_topk_mask([0.5, 0.5, 0.5], [0, 0, 0], 2).tolist() == [True, True, False]


def test_capacity_metrics():
    y = [0, 1, 0, 0, 1]
    day = [0, 0, 0, 1, 1]
    s = [0.1, 0.9, 0.5, 0.3, 0.2]
    # k=1 -> alerts at index 1 (fraud) and 3 (legit)
    assert precision_at_capacity(y, s, day, 1) == pytest.approx(0.5)
    assert recall_at_capacity(y, s, day, 1) == pytest.approx(0.5)


def test_fraud_value_caught():
    assert fraud_value_caught([1, 1, 0], [100, 300, 50], [0, 1, 1]) == pytest.approx(0.75)
    assert fraud_value_caught([0, 0], [1, 2], [1, 1]) == 0.0
