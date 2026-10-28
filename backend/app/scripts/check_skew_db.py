"""Training-serving skew check against PostgreSQL.

For a random sample of stored validation and test transactions, the scoring service (online
features with history read from PostgreSQL, then model, policy, anomaly flag) must reproduce
the offline results: batch features from data/processed/features_*.parquet and the offline
model scores. Read-only: nothing is written to the database.

Test-period rows are sampled only among transactions already loaded, because their history
must be complete in the database (earlier test days are only there once loaded).

    python -m app.scripts.check_skew_db [--n 1000] [--seed 0]
Prints RESULT: PASS / FAIL (exit code 1 on FAIL).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import numpy as np
import pandas as pd
from fraudml import config
from fraudml.features.pipeline import CATEGORICAL_FEATURES, FEATURES_CURATED
from fraudml.features.skew import values_match
from sqlalchemy import select

from app.db.models import Transaction
from app.db.session import SessionLocal
from app.services.model_store import ModelStore
from app.services.scoring import compute_score

PROB_TOL = 1e-9


def _offline(period: str, bundle) -> pd.DataFrame:
    """Offline batch features + offline model outputs for one period, indexed by TransactionID."""
    path = config.PROCESSED_DIR / f"features_{period}.parquet"
    f = pd.read_parquet(path).set_index(config.ID_COL)
    model = bundle.model
    f["_prob"] = model.predict_proba(f[model.features])
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="total sample size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with SessionLocal() as db:
        bundle = ModelStore().load_active(db)
        if bundle is None:
            print("error: no active model", file=sys.stderr)
            sys.exit(1)

        offline = {p: _offline(p, bundle) for p in ("validation", "test")}
        val_start, val_end = config.SPLIT_DATES["validation"]
        test_start, test_end = config.SPLIT_DATES["test"]
        ranges = {
            "validation": (dt.date.fromisoformat(val_start), dt.date.fromisoformat(val_end)),
            "test": (dt.date.fromisoformat(test_start), dt.date.fromisoformat(test_end)),
        }
        rng = np.random.default_rng(args.seed)

        sample: list[tuple[str, int]] = []
        for period, (a, b) in ranges.items():
            ids = list(
                db.scalars(
                    select(Transaction.external_id).where(
                        Transaction.transaction_time >= a, Transaction.transaction_time < b
                    )
                )
            )
            if period == "test":
                # Only the earliest loaded test days have complete history in the database.
                print(f"test rows loaded in the DB: {len(ids):,}")
            k = min(len(ids), args.n // 2 if period == "validation" else args.n - len(sample))
            chosen = rng.choice(np.array(ids), size=k, replace=False) if k else []
            sample += [(period, int(i)) for i in chosen]
        print(
            f"sample: {len(sample):,} rows "
            f"({sum(p == 'validation' for p, _ in sample)} validation, "
            f"{sum(p == 'test' for p, _ in sample)} test)"
        )

        by_id = {
            t.external_id: t
            for t in db.scalars(
                select(Transaction).where(Transaction.external_id.in_([i for _, i in sample]))
            )
        }

        feature_mismatches: list[tuple[int, str, object, object]] = []
        max_feat, max_prob = 0.0, 0.0
        alert_mm = unusual_mm = 0
        t0 = time.perf_counter()
        for period, ext_id in sample:
            txn = by_id[ext_id]
            s = compute_score(db, bundle, txn.raw)
            batch = offline[period].loc[ext_id]
            online = s.features.iloc[0]
            for f in FEATURES_CURATED:
                a, b = batch[f], online[f]
                if f not in CATEGORICAL_FEATURES:
                    fa, fb = float(a), float(b)
                    if not (np.isnan(fa) or np.isnan(fb)):
                        max_feat = max(max_feat, abs(fa - fb))
                if not values_match(f, a, b):
                    feature_mismatches.append((ext_id, f, a, b))
            max_prob = max(max_prob, abs(s.fraud_score - float(batch["_prob"])))
            amount = float(txn.raw["TransactionAmt"])
            off_alert = bool(bundle.model.is_alert(batch["_prob"], amount))
            alert_mm += off_alert != s.is_alert
            off_unusual = (not off_alert) and bool(
                bundle.anomaly.is_unusual(offline[period].loc[[ext_id]])[0]
            )
            unusual_mm += off_unusual != s.is_unusual
        elapsed = time.perf_counter() - t0
        db.rollback()  # read-only check

    print(f"checked {len(sample):,} rows x {len(FEATURES_CURATED)} features in {elapsed:.0f} s")
    print(f"feature mismatches: {len(feature_mismatches)} (max numeric abs diff {max_feat:.2e})")
    for m in feature_mismatches[:10]:
        print("   ", m)
    print(f"max |probability diff|: {max_prob:.2e} (tolerance {PROB_TOL:.0e})")
    print(f"alert mismatches: {alert_mm} · unusual-flag mismatches: {unusual_mm}")
    ok = not feature_mismatches and max_prob < PROB_TOL and alert_mm == 0 and unusual_mm == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
