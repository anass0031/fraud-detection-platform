"""Isolation Forest safety net: flags unusual transactions the main model did not alert.

The detector is unsupervised (labels are never used to fit it) and works on the same curated
features as the main model, so the scoring API needs no extra feature code. Preprocessing lives
inside the saved pipeline: median imputation + missing flags for numbers, ordinal codes for
categories (unseen -> -1, missing -> -2). Tree splits do not need scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fraudml.evaluate.cost import CostConfig, cost_breakdown

TOP_SHARE = 0.005  # top 0.5% most unusual transactions (fixed rule, not tuned)


def make_iforest(X: pd.DataFrame, seed: int = 42, n_estimators: int = 200) -> Pipeline:
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
                ),
                cat_cols,
            ),
        ]
    )
    forest = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    return Pipeline([("pre", pre), ("iforest", forest)])


def anomaly_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more unusual (negated sklearn `score_samples`)."""
    return -model.score_samples(X)


def fit_cutoff(train_scores, top_share: float = TOP_SHARE) -> float:
    """Score above which a transaction is among the `top_share` most unusual of the train period."""
    return float(np.quantile(np.asarray(train_scores, dtype="float64"), 1.0 - top_share))


def safety_net(main_alert, score, cutoff: float) -> np.ndarray:
    """Extra alert: NOT alerted by the main rule AND anomaly score at or above the cut-off."""
    return ~np.asarray(main_alert, dtype=bool) & (np.asarray(score, dtype="float64") >= cutoff)


def safety_net_metrics(y, amount, day, main_alert, extra, cfg: CostConfig | None = None) -> dict:
    """What the safety net adds on top of the main rule, for one population."""
    cfg = cfg or CostConfig()
    amount = np.asarray(amount, dtype="float64")
    main_alert = np.asarray(main_alert, dtype=bool)
    extra = np.asarray(extra, dtype=bool)
    y = np.asarray(y).astype(bool)
    n_days = len(np.unique(day))
    base = cost_breakdown(y, amount, main_alert, cfg)
    both = cost_breakdown(y, amount, main_alert | extra, cfg)
    fraud_value = amount[y].sum()
    missed = y & ~main_alert
    per_day = pd.Series(extra).groupby(day).sum()
    return {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "main_alerts_per_day": float(main_alert.sum() / n_days),
        "main_fraud_value_caught": float(amount[y & main_alert].sum() / fraud_value),
        "extra_alerts_per_day": float(extra.sum() / n_days),
        "extra_alerts_max_day": int(per_day.max()),
        "extra_frauds": int((extra & y).sum()),
        "extra_precision": float(y[extra].mean()) if extra.any() else 0.0,
        "extra_lift": float(y[extra].mean() / y.mean()) if extra.any() else 0.0,
        "share_of_missed_frauds_caught": float((extra & y).sum() / missed.sum()),
        "extra_fraud_value_caught": float(amount[extra & y].sum() / fraud_value),
        "combined_fraud_value_caught": float(amount[y & (main_alert | extra)].sum() / fraud_value),
        "main_net_savings": base["net_savings"],
        "combined_net_savings": both["net_savings"],
        "delta_net_savings": both["net_savings"] - base["net_savings"],
    }
