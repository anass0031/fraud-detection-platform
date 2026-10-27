"""Isotonic calibration: maps a model score to an observed fraud probability.

Isotonic regression fits a non-decreasing step curve from score to fraud rate. The order of
scores is preserved, but flat steps create ties, so ranking metrics can drop very slightly.
Prediction is done with linear interpolation between the fitted breakpoints (`np.interp`),
both before and after saving, so a saved calibrator gives bit-identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicCalibrator:
    x: np.ndarray | None = None  # score breakpoints (non-decreasing)
    y: np.ndarray | None = None  # calibrated probability at each breakpoint

    def fit(self, score, y) -> IsotonicCalibrator:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(np.asarray(score, dtype="float64"), np.asarray(y, dtype="float64"))
        self.x = np.asarray(iso.X_thresholds_, dtype="float64")
        self.y = np.asarray(iso.y_thresholds_, dtype="float64")
        return self

    def predict(self, score) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("calibrator is not fitted")
        return np.interp(np.asarray(score, dtype="float64"), self.x, self.y)

    def to_dict(self) -> dict:
        return {"method": "isotonic", "x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        if d.get("method") != "isotonic":
            raise ValueError(f"unknown calibration method {d.get('method')!r}")
        return cls(np.asarray(d["x"], dtype="float64"), np.asarray(d["y"], dtype="float64"))


def expected_loss(prob, amount) -> np.ndarray:
    """Expected fraud loss of a transaction: calibrated probability x amount (alert policy B)."""
    return np.asarray(prob, dtype="float64") * np.asarray(amount, dtype="float64")
