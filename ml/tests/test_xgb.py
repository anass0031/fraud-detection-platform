import numpy as np
import pandas as pd

from fraudml.models.xgb import make_xgb, predict_proba_cpu


def test_xgb_cpu_with_categoricals_and_nan():
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.2).astype(int)
    X = pd.DataFrame(
        {
            "a": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n) + y),
            "c": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    model = make_xgb(device="cpu").set_params(n_estimators=20).fit(X, y)
    p = predict_proba_cpu(model, X)
    assert p.shape == (n,) and ((p >= 0) & (p <= 1)).all()
