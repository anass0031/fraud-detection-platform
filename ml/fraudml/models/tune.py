"""Optuna tuning of LightGBM with forward-chaining folds inside the train period.

Objective = mean validation PR-AUC over the folds. Each fold uses early stopping (on the fold's
own validation slice) to find its tree count. Trials that are clearly worse than the median
after the first fold are pruned to save time. Trial 0 = LightGBM defaults (reference point).
"""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from fraudml.evaluate import pr_auc
from fraudml.models.cv import Fold

LEARNING_RATE = 0.05
MAX_TREES = 3000
EARLY_STOP = 100

# Library defaults expressed in the search-space names (enqueued as trial 0).
DEFAULT_PARAMS = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "reg_alpha": 1e-3,
    "reg_lambda": 1e-3,
    "min_split_gain": 0.0,
}


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def model_params(params: dict, n_estimators: int = MAX_TREES, seed: int = 42) -> dict:
    """Full LGBMClassifier arguments for a set of tuned parameters."""
    return {
        **params,
        "learning_rate": LEARNING_RATE,
        "n_estimators": n_estimators,
        "subsample_freq": 1,  # row subsampling is active only when subsample_freq > 0
        "metric": "average_precision",
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def cv_score(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
    trial: optuna.Trial | None = None,
    seed: int = 42,
) -> dict:
    scores, iters = [], []
    t0 = time.perf_counter()
    for step, f in enumerate(folds):
        model = LGBMClassifier(**model_params(params, seed=seed))
        model.fit(
            X.iloc[f.train_idx],
            y.iloc[f.train_idx],
            eval_X=(X.iloc[f.val_idx],),
            eval_y=(y.iloc[f.val_idx],),
            callbacks=[early_stopping(EARLY_STOP, verbose=False)],
        )
        p = model.predict_proba(X.iloc[f.val_idx], num_iteration=model.best_iteration_)[:, 1]
        scores.append(pr_auc(y.iloc[f.val_idx], p))
        iters.append(int(model.best_iteration_))
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return {
        "score": float(np.mean(scores)),
        "fold_scores": scores,
        "best_iterations": iters,
        "seconds": time.perf_counter() - t0,
    }


def make_objective(X: pd.DataFrame, y: pd.Series, folds: list[Fold], seed: int = 42):
    def objective(trial: optuna.Trial) -> float:
        res = cv_score(suggest_params(trial), X, y, folds, trial, seed)
        trial.set_user_attr("fold_scores", res["fold_scores"])
        trial.set_user_attr("best_iterations", res["best_iterations"])
        trial.set_user_attr("seconds", round(res["seconds"], 1))
        return res["score"]

    return objective


def final_n_estimators(best_iterations: list[int], factor: float = 1.1) -> int:
    """Tree count for the final model: it sees more days than any fold, so slightly more trees."""
    return int(round(np.median(best_iterations) * factor))
