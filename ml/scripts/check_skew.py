"""Real-data checks for the feature pipeline (run after build_features.py).

1. Training-serving skew: for a stratified sample of validation/test transactions, recompute the
   features with the ONLINE path (one transaction + its history) and compare with the saved BATCH
   features. Expected: 0 mismatches.
2. Future invariance: rebuild batch features on data truncated at a cut time and compare with the
   saved features for rows before the cut. Expected: identical.

Usage: ml/.venv/Scripts/python ml/scripts/check_skew.py [sample_size]
"""

import sys
import time

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.data.ingest import load_split
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.history import InMemoryHistory
from fraudml.features.keys import make_uid
from fraudml.features.pipeline import FEATURES_CURATED, build_features_batch
from fraudml.features.skew import compare_rows

ENCODER_PATH = config.ARTIFACTS_DIR / "feature_encoder.json"


def load_saved(periods) -> pd.DataFrame:
    parts = [pd.read_parquet(config.PROCESSED_DIR / f"features_{p}.parquet") for p in periods]
    return pd.concat(parts, ignore_index=True).set_index(config.ID_COL)


def stratified_sample(raw: pd.DataFrame, saved: pd.DataFrame, n: int, seed: int = 0) -> pd.Index:
    rng = np.random.default_rng(seed)
    ids = saved.index
    uid = make_uid(raw.loc[ids])
    tie = uid.notna() & raw.loc[ids].assign(uid=uid).duplicated(
        ["uid", config.TIME_COL], keep=False
    )
    groups = {
        "same-second ties": ids[tie.to_numpy()],
        "no uid": ids[saved["has_uid"].to_numpy() == 0],
        "long history (>= 10 prior)": ids[(saved["uid_n_prior"] >= 10).to_numpy()],
        "unseen/rare category": ids[
            ((saved["browser_family"] == "other") | (saved["p_email"] == "other")).to_numpy()
        ],
    }
    chosen: list = []
    per_group = n // 5
    for name, g in groups.items():
        take = g if len(g) <= per_group else rng.choice(g, per_group, replace=False)
        chosen.extend(take)
        print(f"  sample group {name:<28} {len(take):>5} rows (available {len(g):,})")
    rest = np.setdiff1d(ids, np.array(chosen))
    n_random = max(n - len(chosen), 0)
    chosen.extend(rng.choice(rest, n_random, replace=False))
    print(f"  sample group {'random':<28} {n_random:>5}")
    return pd.Index(np.unique(chosen))


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    df = load_split("train")
    raw = df.set_index(config.ID_COL, drop=False)
    encoder = FeatureEncoder.load(ENCODER_PATH)
    saved = load_saved(("validation", "test"))

    # ---- 1. skew check
    print(f"[1] Skew check on {n} validation/test transactions")
    ids = stratified_sample(raw, saved, n)
    history = InMemoryHistory(df)
    t0 = time.perf_counter()
    mismatches, max_abs = compare_rows(raw.loc[ids], saved.loc[ids], history, encoder)
    secs = time.perf_counter() - t0
    print(
        f"  compared {len(ids)} rows x {len(FEATURES_CURATED)} features in {secs:.0f} s "
        f"({secs / len(ids) * 1000:.0f} ms per transaction)"
    )
    print(f"  mismatches: {len(mismatches)} | max numeric abs diff: {max_abs:.2e}")
    for m in mismatches[:10]:
        print("   ", m)

    # ---- 2. future invariance on real data
    print("[2] Future invariance on real data")
    cut = int(saved.loc[saved[config.PERIOD_COL] == "validation", config.TIME_COL].quantile(0.5))
    truncated = df[df[config.TIME_COL] <= cut]
    rebuilt = build_features_batch(truncated, encoder)
    rebuilt.index = pd.Index(truncated[config.ID_COL].to_numpy(), name=config.ID_COL)
    check_ids = saved.index[(saved[config.TIME_COL] <= cut).to_numpy()]
    a = saved.loc[check_ids, list(FEATURES_CURATED)]
    b = rebuilt.loc[check_ids, list(FEATURES_CURATED)]
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        fi_ok = True
        print(f"  {len(check_ids):,} validation rows before the cut: identical")
    except AssertionError as e:
        fi_ok = False
        print("  DIFFERENCE:", str(e)[:500])

    ok = not mismatches and fi_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
