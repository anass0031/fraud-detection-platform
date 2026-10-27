import json

import numpy as np
import pytest

from fraudml.evaluate import brier, optimise_threshold
from fraudml.models.calibration import IsotonicCalibrator, expected_loss


def _miscalibrated(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.beta(0.5, 12, n)  # mostly small probabilities, like fraud
    y = (rng.random(n) < p_true).astype(int)
    score = np.sqrt(p_true)  # same order, wrong scale (too high)
    return p_true, y, score


def test_calibration_improves_brier_and_matches_true_probability():
    p_true, y, score = _miscalibrated()
    cal = IsotonicCalibrator().fit(score, y)
    p = cal.predict(score)
    assert brier(y, p) < brier(y, score)
    assert abs(p.mean() - y.mean()) < 0.002
    assert np.mean(np.abs(p - p_true)) < 0.02


def test_output_is_monotone_and_in_unit_interval():
    _, y, score = _miscalibrated()
    cal = IsotonicCalibrator().fit(score, y)
    grid = np.linspace(-1, 2, 1000)  # includes out-of-range scores (clipped)
    out = cal.predict(grid)
    assert (np.diff(out) >= 0).all()
    assert out.min() >= 0 and out.max() <= 1


def test_json_round_trip_is_bit_identical():
    _, y, score = _miscalibrated()
    cal = IsotonicCalibrator().fit(score, y)
    loaded = IsotonicCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert np.array_equal(cal.predict(score), loaded.predict(score))


def test_unfitted_and_unknown_method_raise():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict([0.1])
    with pytest.raises(ValueError):
        IsotonicCalibrator.from_dict({"method": "platt"})


def test_expected_loss_policy_prefers_large_amounts():
    # 30% on $2,000 (EL 600) ranks above 60% on $20 (EL 12)
    el = expected_loss([0.3, 0.6], [2000, 20])
    assert el.tolist() == pytest.approx([600, 12])
    # the capacity-constrained optimiser accepts any ranking score
    y = [1, 0, 1, 0]
    res = optimise_threshold(y, expected_loss([0.3, 0.6, 0.9, 0.1], [2000, 20, 50, 10]),
                             [2000, 20, 50, 10], [0, 0, 1, 1])  # fmt: skip
    assert res.n_alerts >= 1 and res.alerts_per_day <= 150
