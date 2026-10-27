import numpy as np
import optuna
import pandas as pd

from fraudml.models.cv import forward_folds
from fraudml.models.tune import DEFAULT_PARAMS, cv_score, final_n_estimators, make_objective


def _data(n_days=40, per_day=150, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    n = len(day)
    y = pd.Series((rng.random(n) < 0.1).astype("int8"))
    X = pd.DataFrame({"a": rng.normal(size=n) + 1.5 * y, "b": rng.normal(size=n)})
    return X, y, day


def test_cv_score_returns_one_result_per_fold():
    X, y, day = _data()
    folds = forward_folds(day, n_folds=2, val_days=5, gap_days=2)
    res = cv_score(DEFAULT_PARAMS, X, y, folds)
    assert len(res["fold_scores"]) == len(res["best_iterations"]) == 2
    assert 0.3 < res["score"] <= 1.0  # signal in "a" is easy to learn
    assert all(i >= 1 for i in res["best_iterations"])


def test_objective_runs_in_a_study_and_stores_attrs():
    X, y, day = _data()
    folds = forward_folds(day, n_folds=2, val_days=5, gap_days=2)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(DEFAULT_PARAMS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(make_objective(X, y, folds), n_trials=2)
    assert study.trials[0].params == DEFAULT_PARAMS
    assert len(study.best_trial.user_attrs["fold_scores"]) == 2


def test_final_n_estimators():
    assert final_n_estimators([100, 200, 300]) == 220
