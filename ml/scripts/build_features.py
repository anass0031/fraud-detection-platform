"""Build the curated feature datasets for train / validation / test.

- Features are computed on the whole timeline at once, so validation/test rows can use earlier
  transactions (train and gap periods) as history. Each feature only uses the past.
- The encoder is fitted on TRAIN rows only, then saved to artifacts/feature_encoder.json.
- Gap rows are not written (they are history only).

Usage: ml/.venv/Scripts/python ml/scripts/build_features.py
Output: data/processed/features_{train,validation,test}.parquet
"""

import sys
import time

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.data.ingest import load_split
from fraudml.data.split import assign_period
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.pipeline import FEATURES_CURATED, raw_features_batch

ENCODER_PATH = config.ARTIFACTS_DIR / "feature_encoder.json"
META_COLUMNS = [
    config.ID_COL,
    config.TIME_COL,
    config.PERIOD_COL,
    config.TARGET,
    config.EVAL_FLAG_COL,
]


def main() -> int:
    t0 = time.perf_counter()
    df = load_split("train")
    period = assign_period(df[config.TIME_COL])
    raw, uid = raw_features_batch(df)
    t_feat = time.perf_counter() - t0

    is_train = period == "train"
    encoder = FeatureEncoder().fit(raw[is_train])
    encoder.save(ENCODER_PATH)
    features = encoder.transform(raw)[list(FEATURES_CURATED)]

    # Evaluation-only flag: this uid already had a fraud in the train period (never a feature).
    fraud_uids = set(uid[is_train & (df[config.TARGET] == 1)].dropna())
    flag = uid.notna() & uid.isin(fraud_uids) & ~is_train

    out = pd.concat(
        [
            df[[config.ID_COL, config.TIME_COL]],
            period.rename(config.PERIOD_COL),
            df[config.TARGET],
            flag.rename(config.EVAL_FLAG_COL),
            features,
        ],
        axis=1,
    )
    assert not set(FEATURES_CURATED) & set(META_COLUMNS)

    print(
        f"features computed for {len(df):,} rows in {t_feat:.1f} s; "
        f"{len(FEATURES_CURATED)} curated features"
    )
    for p in config.MODEL_PERIODS:
        part = out[out[config.PERIOD_COL] == p].reset_index(drop=True)
        path = config.PROCESSED_DIR / f"features_{p}.parquet"
        part.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        print(
            f"  {p:<10} rows={len(part):>7,}  fraud={part[config.TARGET].mean():.2%}  "
            f"uid_fraud_in_train={int(part[config.EVAL_FLAG_COL].sum()):>4}  "
            f"-> {path.name} ({path.stat().st_size / 2**20:.1f} MiB)"
        )

    nan_share = features[is_train].isna().mean().sort_values(ascending=False)
    print(
        "highest missing share in train features:",
        {k: f"{v:.0%}" for k, v in nan_share.head(5).items()},
    )
    other_share = {
        c: f"{(features.loc[is_train, c] == 'other').mean():.1%}"
        for c in ["p_email", "browser_family", "os_family"]
    }
    print("share mapped to 'other' in train:", other_share)
    print(f"encoder saved to {ENCODER_PATH} (min_count={encoder.min_count})")
    print(f"total {time.perf_counter() - t0:.1f} s")
    return 0 if np.isfinite(t_feat) else 1


if __name__ == "__main__":
    sys.exit(main())
