"""Baselines: a simple amount rule and a logistic regression.

Logistic regression needs numbers without gaps, so it gets its own preprocessing (fitted on
train only, inside the sklearn Pipeline): median imputation + missing-value flags, a signed log
(count/amount features are very skewed), scaling, and one-hot encoding of categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


class AmountRule:
    """Flag transactions above the train 99th percentile of the amount.

    score = flag + amount / (1 + amount): flagged rows always score >= 1 and the amount breaks
    ties, so ranking metrics can still be computed. `RULE_THRESHOLD` = the rule as written.
    """

    RULE_THRESHOLD = 1.0

    def fit(self, amount: pd.Series) -> AmountRule:
        self.cutoff_ = float(np.quantile(np.asarray(amount, dtype="float64"), 0.99))
        return self

    def score(self, amount: pd.Series) -> np.ndarray:
        a = np.asarray(amount, dtype="float64")
        return (a > self.cutoff_).astype("float64") + a / (1.0 + a)


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_logistic(X: pd.DataFrame, seed: int = 42) -> Pipeline:
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat_cols),
        ]
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])
