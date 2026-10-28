"""Data drift: Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) against a reference.

The reference is built once from the training period (features) and the validation period
(model score). A monitored window (one replayed day) is binned with the reference's bins:

- numeric: 10 quantile bins from the reference, plus one bin for missing values;
- categorical: one bin per reference category, plus one bin for anything unseen.

PSI = Σ (actual% − expected%) · ln(actual% / expected%). Rule of thumb (idea.md §3.8):
< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant. KS compares the raw numeric values with
a stored reference sample; with thousands of rows its p-value is almost always tiny, so the
severity is decided by PSI only and KS is reported for information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

N_BINS = 10
SAMPLE_SIZE = 2000
EPS = 1e-4  # floor for empty bins (avoids log(0) and division by 0)
UNSEEN = "__unseen__"
SCORE = "__score__"

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25

Kind = Literal["numeric", "categorical"]
Severity = Literal["stable", "moderate", "significant"]


@dataclass
class FeatureReference:
    name: str
    kind: Kind
    expected: list[float]  # proportion per bin (last bin: missing / unseen)
    edges: list[float] = field(default_factory=list)  # numeric: inner bin edges
    categories: list[str] = field(default_factory=list)  # categorical: reference values
    sample: list[float] = field(default_factory=list)  # numeric: values for KS
    n: int = 0


@dataclass
class DriftResult:
    feature: str
    n: int
    psi: float
    ks_stat: float | None
    ks_pvalue: float | None
    severity: Severity


def _numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")


def _numeric_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    missing = np.isnan(x)
    idx = np.searchsorted(edges, x[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype("float64")
    return np.append(counts, missing.sum())


def _categorical_counts(values: pd.Series | np.ndarray, categories: list[str]) -> np.ndarray:
    s = pd.Series(values).astype("object").where(pd.Series(values).notna(), "(missing)")
    s = s.astype(str)
    known = s.value_counts()
    counts = [float(known.get(c, 0)) for c in categories]
    unseen = float(len(s) - sum(counts))
    return np.array([*counts, unseen])


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def build_feature_reference(
    name: str,
    values: pd.Series | np.ndarray,
    kind: Kind,
    bins: int = N_BINS,
    sample_size: int = SAMPLE_SIZE,
    seed: int = 0,
) -> FeatureReference:
    """Bins and expected proportions of one feature on the reference data."""
    if kind == "categorical":
        s = pd.Series(values).astype("object").where(pd.Series(values).notna(), "(missing)")
        categories = sorted(s.astype(str).unique().tolist())
        expected = _proportions(_categorical_counts(s, categories))
        return FeatureReference(
            name=name, kind=kind, expected=expected.tolist(), categories=categories, n=len(s)
        )

    x = _numeric(values)
    present = x[~np.isnan(x)]
    if len(present) == 0:
        edges = np.array([], dtype="float64")
    else:
        quantiles = np.quantile(present, np.linspace(0, 1, bins + 1)[1:-1])
        edges = np.unique(quantiles)  # heavy ties (e.g. many zeros) merge bins
    rng = np.random.default_rng(seed)
    sample = rng.choice(present, size=min(sample_size, len(present)), replace=False)
    expected = _proportions(_numeric_counts(x, edges))
    return FeatureReference(
        name=name,
        kind=kind,
        expected=expected.tolist(),
        edges=edges.tolist(),
        sample=np.sort(sample).tolist(),
        n=len(x),
    )


def actual_proportions(ref: FeatureReference, values: pd.Series | np.ndarray) -> np.ndarray:
    if ref.kind == "categorical":
        return _proportions(_categorical_counts(values, ref.categories))
    return _proportions(_numeric_counts(_numeric(values), np.asarray(ref.edges)))


def psi(expected: np.ndarray | list[float], actual: np.ndarray | list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype="float64"), EPS, None)
    a = np.clip(np.asarray(actual, dtype="float64"), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def severity(value: float) -> Severity:
    if value < PSI_MODERATE:
        return "stable"
    if value <= PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def measure(ref: FeatureReference, values: pd.Series | np.ndarray) -> DriftResult:
    """PSI (+ KS for numeric features) of `values` against the reference."""
    n = len(values)
    if n == 0:
        return DriftResult(ref.name, 0, 0.0, None, None, "stable")
    value = psi(ref.expected, actual_proportions(ref, values))
    ks_stat = ks_p = None
    if ref.kind == "numeric" and ref.sample:
        x = _numeric(values)
        x = x[~np.isnan(x)]
        if len(x) > 0:
            result = stats.ks_2samp(np.asarray(ref.sample), x)
            ks_stat, ks_p = float(result.statistic), float(result.pvalue)
    return DriftResult(ref.name, n, value, ks_stat, ks_p, severity(value))


# ---------------------------------------------------------------- whole reference (JSON)


def build_reference(
    features: pd.DataFrame,
    categorical: set[str] | tuple[str, ...],
    score: np.ndarray | pd.Series,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reference for every column of `features` plus the model score, as a JSON-ready dict."""
    refs = [
        build_feature_reference(c, features[c], "categorical" if c in categorical else "numeric")
        for c in features.columns
    ]
    refs.append(build_feature_reference(SCORE, np.asarray(score), "numeric"))
    return {"meta": meta or {}, "features": [asdict(r) for r in refs]}


def load_reference(data: dict[str, Any]) -> dict[str, FeatureReference]:
    return {f["name"]: FeatureReference(**f) for f in data["features"]}
