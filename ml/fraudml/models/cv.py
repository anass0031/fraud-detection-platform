"""Forward-chaining (expanding-window) folds inside one period, by day.

Fold k trains on every day before `train_end`, skips `gap_days`, then validates on the next
`val_days` days. The last fold ends on the last day of the data; earlier folds step back by
`val_days`. This mirrors production: always train on the past, predict the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_days: tuple[int, int]  # [first, last] day, inclusive
    val_days: tuple[int, int]


def forward_folds(day, n_folds: int = 3, val_days: int = 15, gap_days: int = 7) -> list[Fold]:
    """Folds ordered oldest -> newest. Raises if the earliest fold would have no training days."""
    day = np.asarray(day)
    first, last = int(day.min()), int(day.max())
    folds = []
    for k in reversed(range(n_folds)):
        val_end = last + 1 - k * val_days  # exclusive
        val_start = val_end - val_days
        train_end = val_start - gap_days  # exclusive
        if train_end <= first:
            raise ValueError("not enough days for the requested folds")
        train_idx = np.flatnonzero(day < train_end)
        val_idx = np.flatnonzero((day >= val_start) & (day < val_end))
        folds.append(Fold(train_idx, val_idx, (first, train_end - 1), (val_start, val_end - 1)))
    return folds
