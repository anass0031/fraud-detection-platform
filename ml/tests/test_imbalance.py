import numpy as np
import pandas as pd
import pytest

from fraudml.models.imbalance import correct_probability, undersample
from fraudml.models.lgbm import make_lgbm


def _xy(n=10_000, rate=0.04, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < rate).astype("int8"))
    X = pd.DataFrame({"a": rng.normal(size=n) + 2 * y, "b": rng.normal(size=n)})
    return X, y


def test_undersample_keeps_all_fraud_and_ratio():
    X, y = _xy()
    Xs, ys, beta = undersample(X, y, legit_per_fraud=5, seed=1)
    n_pos = int(y.sum())
    assert int(ys.sum()) == n_pos
    assert int((ys == 0).sum()) == 5 * n_pos
    assert beta == pytest.approx(5 * n_pos / int((y == 0).sum()))
    assert Xs.index.is_monotonic_increasing and (Xs.index == ys.index).all()


def test_undersample_is_deterministic():
    X, y = _xy()
    a = undersample(X, y, seed=3)[0].index
    b = undersample(X, y, seed=3)[0].index
    assert (a == b).all()


def test_undersample_ratio_larger_than_data_keeps_everything():
    X, y = _xy(n=200, rate=0.4)
    Xs, _, beta = undersample(X, y, legit_per_fraud=100)
    assert len(Xs) == len(X) and beta == 1.0


@pytest.mark.parametrize("beta", [1.0, 0.5, 0.1, 0.02])
def test_correction_inverts_the_sampling_bias(beta):
    p_true = np.array([0.0, 0.001, 0.03, 0.2, 0.7, 1.0])
    odds_s = p_true / np.maximum(1 - p_true, 1e-300) / beta  # odds seen after undersampling
    p_sampled = np.where(p_true == 1.0, 1.0, odds_s / (1 + odds_s))
    assert np.allclose(correct_probability(p_sampled, beta), p_true)


def test_correction_keeps_ranking():
    p = np.linspace(0.01, 0.99, 50)
    assert (np.diff(correct_probability(p, 0.05)) > 0).all()


def test_corrected_mean_matches_true_rate_on_simulated_data():
    X, y = _xy(n=40_000, rate=0.04, seed=5)
    Xs, ys, beta = undersample(X, y, legit_per_fraud=5)
    model = make_lgbm(n_estimators=50).fit(Xs, ys)
    p = correct_probability(model.predict_proba(X)[:, 1], beta)
    assert p.mean() == pytest.approx(y.mean(), abs=0.01)


def test_make_lgbm_scale_pos_weight():
    assert make_lgbm(scale_pos_weight=28.0).get_params()["scale_pos_weight"] == 28.0
    assert make_lgbm().get_params()["n_estimators"] == 500
