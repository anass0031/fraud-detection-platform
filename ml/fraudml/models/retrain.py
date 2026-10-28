"""Helpers of the offline retraining (champion vs challenger on replayed, labelled days).

Windows of the replayed weeks (each [start, end)):
- `train_extra` (W1) is added to the original train + validation data;
- `gap` (W2) is never used for fitting (same 7-day gap rule as the original split);
- `calibration` (W3): isotonic calibration + cost-optimal threshold;
- `comparison` (W4): champion and challenger scored on the same rows, never used for fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudml.evaluate.cost import CostConfig, cost_breakdown
from fraudml.evaluate.metrics import pr_auc

WINDOWS: dict[str, tuple[str, str]] = {
    "train_extra": ("2018-04-28", "2018-05-05"),
    "gap": ("2018-05-05", "2018-05-12"),
    "calibration": ("2018-05-12", "2018-05-19"),
    "comparison": ("2018-05-19", "2018-05-26"),
}


def split_windows(replayed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Labelled replayed rows of each window (rows without a label are dropped)."""
    times = pd.to_datetime(replayed["transaction_time"])
    labelled = replayed["label"].notna()
    out = {}
    for name, (start, end) in WINDOWS.items():
        mask = labelled & (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))
        out[name] = replayed[mask].reset_index(drop=True)
    return out


def align_features(df: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """The model features of `df` with exactly the dtypes of `template` (the training frame).

    Categorical values must already be in the training categories (the shared encoder maps
    rare and unseen values to "other" / "(missing)"); anything else is an error, because it
    would silently become a missing value.
    """
    out = {}
    for col in template.columns:
        dtype = template[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            values = df[col].astype("object")
            unknown = values.notna() & ~values.isin(dtype.categories)
            if unknown.any():
                raise ValueError(
                    f"{col}: {int(unknown.sum())} values outside the training categories, "
                    f"e.g. {values[unknown].iloc[0]!r}"
                )
            out[col] = pd.Categorical(values, dtype=dtype)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return pd.DataFrame(out, index=df.index)


def fraud_customers(customer: pd.Series, y: pd.Series | np.ndarray) -> set[str]:
    """Customers (pseudo uid) with at least one fraud in the given labelled rows."""
    mask = np.asarray(y).astype(bool) & customer.notna().to_numpy()
    return set(customer[mask].astype(str))


def window_metrics(
    y, prob, amount, alert, day, cfg: CostConfig | None = None
) -> dict[str, float | int | None]:
    """Business view of one model on one window (same cost model as the offline evaluation)."""
    y = np.asarray(y).astype(bool)
    alert = np.asarray(alert).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    c = cost_breakdown(y, amount, alert, cfg or CostConfig())
    alerts = int(alert.sum())
    n_fraud = int(y.sum())
    fraud_value = float(amount[y].sum())
    days = len(np.unique(np.asarray(day)))
    return {
        "n": int(len(y)),
        "n_fraud": n_fraud,
        "alerts": alerts,
        "alerts_per_day": alerts / days if days else 0.0,
        "precision": c["tp"] / alerts if alerts else None,
        "recall": c["tp"] / n_fraud if n_fraud else None,
        "pr_auc": pr_auc(y, np.asarray(prob)) if 0 < n_fraud < len(y) else None,
        "fraud_value_caught": float(amount[y & alert].sum()) / fraud_value if fraud_value else None,
        "total_cost": c["total_cost"],
        "net_savings": c["net_savings"],
    }


def paired_bootstrap_pr_auc(
    y, score_a, score_b, day, n_boot: int = 500, seed: int = 42
) -> dict[str, float]:
    """Day-block bootstrap of PR-AUC(b) - PR-AUC(a) on the same rows: whole days are resampled
    with replacement (rows of one day are not independent). Few days -> a wide interval."""
    y = np.asarray(y).astype(bool)
    a, b = np.asarray(score_a, dtype="float64"), np.asarray(score_b, dtype="float64")
    day = np.asarray(day)
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, size=len(days))])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(pr_auc(y[idx], b[idx]) - pr_auc(y[idx], a[idx]))
    d = np.array(diffs)
    return {
        "diff": float(pr_auc(y, b) - pr_auc(y, a)),
        "low": float(np.quantile(d, 0.025)),
        "high": float(np.quantile(d, 0.975)),
        "share_positive": float((d > 0).mean()),
        "days": int(len(days)),
    }
