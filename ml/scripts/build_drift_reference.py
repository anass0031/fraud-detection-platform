"""Build the drift reference of a model: training-period features + validation-period scores.

Writes artifacts/monitoring/drift_reference-<model>.json (the model bundle itself is not
changed). The backend's monitoring compares each replayed day with this reference.

Example (repo root):
    ml/.venv/Scripts/python ml/scripts/build_drift_reference.py --model lgbm-2026.10.27-1
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from fraudml import config
from fraudml.features.pipeline import CATEGORICAL_FEATURES
from fraudml.monitoring.drift import SCORE, build_reference, severity
from fraudml.registry import load_model

MONITORING_DIR = config.ARTIFACTS_DIR / "monitoring"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="lgbm-2026.10.27-1")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = load_model(args.model)
    features = list(model.features)
    train = pd.read_parquet(config.PROCESSED_DIR / "features_train.parquet", columns=features)
    validation = pd.read_parquet(
        config.PROCESSED_DIR / "features_validation.parquet", columns=features
    )
    score = model.predict_proba(validation)  # calibrated, out-of-sample
    print(
        f"model {args.model}: {len(features)} features · reference rows train {len(train):,} · "
        f"score rows validation {len(validation):,}"
    )

    data = build_reference(
        train,
        set(CATEGORICAL_FEATURES),
        score,
        meta={
            "model": args.model,
            "features_from": "train " + " -> ".join(config.SPLIT_DATES["train"]),
            "score_from": "validation " + " -> ".join(config.SPLIT_DATES["validation"]),
            "bins": "10 quantile bins + missing (numeric), one bin per category + unseen",
        },
    )
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    out = MONITORING_DIR / f"drift_reference-{args.model}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    n_cat = sum(f["kind"] == "categorical" for f in data["features"])
    print(
        f"wrote {out.relative_to(config.REPO_ROOT)} ({out.stat().st_size / 1024:.0f} kB): "
        f"{len(data['features'])} entries ({n_cat} categorical, score as {SCORE!r})"
    )

    # Sanity check: the training data compared with itself must be stable.
    from fraudml.monitoring.drift import load_reference, measure  # noqa: PLC0415

    refs = load_reference(data)
    worst = max(measure(refs[c], train[c]).psi for c in features)
    print(f"self-check: max PSI train vs its own reference = {worst:.2e} ({severity(worst)})")
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
