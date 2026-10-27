"""FeatureEncoder: category lists and frequency maps learned from the TRAIN period only.

- Categorical columns: values seen fewer than `min_count` times in train become "other";
  values never seen in train (e.g. new browser families at scoring time) also become "other";
  missing values become "(missing)". The category list is fixed, so train / validation / test /
  API all produce the same categories (needed by LightGBM's native categorical support).
- Frequency columns: share of train transactions with the same value (unseen -> 0, missing -> NaN).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraudml.features.static import RAW_CATEGORICAL, RAW_FREQUENCY

OTHER = "other"
MISSING = "(missing)"
ENCODER_VERSION = 1


def _keys(s: pd.Series) -> pd.Series:
    """String keys for frequency maps; integer-valued numbers lose their '.0' (1000.0 -> '1000')."""
    if pd.api.types.is_numeric_dtype(s):
        x = s.astype("float64")
        out = pd.Series(np.nan, index=s.index, dtype="object")
        ok = x.notna()
        is_int = ok & (x == np.round(x))
        out[is_int] = x[is_int].astype("int64").astype(str)
        out[ok & ~is_int] = x[ok & ~is_int].astype(str)
        return out
    return s.astype("object")


@dataclass
class FeatureEncoder:
    min_count: int = 100
    n_train: int = 0
    categories: dict[str, list[str]] = field(default_factory=dict)
    frequencies: dict[str, dict[str, float]] = field(default_factory=dict)

    # ---------------------------------------------------------------- fit / transform
    def fit(self, raw_train: pd.DataFrame) -> FeatureEncoder:
        self.n_train = len(raw_train)
        for col in RAW_CATEGORICAL:
            counts = raw_train[col].dropna().value_counts()
            # A raw value literally named "other" (e.g. browser family) joins the OTHER bucket.
            kept = sorted(
                str(v) for v in counts[counts >= self.min_count].index if v not in (OTHER, MISSING)
            )
            self.categories[col] = [*kept, OTHER, MISSING]
        for out_col, src in RAW_FREQUENCY.items():
            counts = _keys(raw_train[src]).dropna().value_counts()
            self.frequencies[out_col] = {str(k): float(v) / self.n_train for k, v in counts.items()}
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.categories:
            raise RuntimeError("FeatureEncoder is not fitted")
        out = raw.copy()
        for col, cats in self.categories.items():
            s = raw[col].astype("object")
            known = set(cats) - {OTHER, MISSING}
            mapped = np.where(s.isna(), MISSING, np.where(s.isin(known), s, OTHER))
            out[col] = pd.Categorical(mapped, categories=cats)
        for out_col, src in RAW_FREQUENCY.items():
            keys = _keys(raw[src])
            freq = keys.map(self.frequencies[out_col]).astype("float64")
            out[out_col] = freq.where(keys.isna(), freq.fillna(0.0))
        return out

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": ENCODER_VERSION,
            "min_count": self.min_count,
            "n_train": self.n_train,
            "categories": self.categories,
            "frequencies": self.frequencies,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureEncoder:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("version") != ENCODER_VERSION:
            raise ValueError(f"Unsupported encoder version {d.get('version')}")
        return cls(
            min_count=d["min_count"],
            n_train=d["n_train"],
            categories=d["categories"],
            frequencies=d["frequencies"],
        )
