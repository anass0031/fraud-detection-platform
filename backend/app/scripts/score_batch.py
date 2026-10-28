"""Score stored transactions of a date range with the active model (same service as the API).

Example (inside the backend container):
    python -m app.scripts.load_transactions --from 2018-04-28 --to 2018-04-29 --source batch
    python -m app.scripts.score_batch --from 2018-04-28 --to 2018-04-29

Dates are [from, to). Transactions that already have a prediction for the active model are
skipped, and work is committed in chunks, so the script can be stopped and re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from collections import Counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Prediction, Transaction
from app.db.session import SessionLocal
from app.services.model_store import ModelStore, ScoringBundle
from app.services.scoring import score_stored


def pending_ids(db: Session, model_version_id: int, start: dt.date, end: dt.date) -> list[int]:
    """Transactions in [start, end) without a prediction for this model, in time order."""
    scored = select(Prediction.id).where(
        Prediction.transaction_id == Transaction.id,
        Prediction.model_version_id == model_version_id,
    )
    query = (
        select(Transaction.id)
        .where(
            Transaction.transaction_time >= start,
            Transaction.transaction_time < end,
            ~scored.exists(),
        )
        .order_by(Transaction.transaction_dt, Transaction.external_id)
    )
    return list(db.scalars(query))


def score_ids(db: Session, bundle: ScoringBundle, ids: list[int], chunk_size: int = 200) -> list:
    """Score the given transactions, committing after each chunk. Returns the predictions."""
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        txns = db.scalars(
            select(Transaction)
            .where(Transaction.id.in_(chunk))
            .order_by(Transaction.transaction_dt, Transaction.external_id)
        ).all()
        for txn in txns:
            preds.append(score_stored(db, bundle, txn))
        db.commit()
        done = i + len(chunk)
        rate = done / (time.perf_counter() - t0)
        left = (len(ids) - done) / rate
        print(f"  {done:>6,}/{len(ids):,} scored ({rate:.1f}/s, ~{left:.0f} s left)")
    return preds


def summary(preds: list) -> str:
    if not preds:
        return "nothing to score"
    lat = np.array([p.latency_ms for p in preds])
    levels = Counter(p.risk_level for p in preds)
    alerts = sum(p.is_alert for p in preds)
    return (
        f"rows {len(preds):,} · alerts {alerts:,} ({alerts / len(preds):.1%}) · "
        f"unusual flags {sum(p.is_unusual for p in preds):,} · "
        f"risk levels {dict(sorted(levels.items()))} · "
        f"latency mean {lat.mean():.0f} ms, p95 {np.percentile(lat, 95):.0f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--to must be after --from")

    with SessionLocal() as db:
        bundle = ModelStore().load_active(db)
        if bundle is None:
            print("error: no active model (run register_model --activate)", file=sys.stderr)
            sys.exit(1)
        ids = pending_ids(db, bundle.model_version_id, args.start, args.end)
        print(
            f"model {bundle.version}: {len(ids):,} transactions to score "
            f"in [{args.start}, {args.end})"
        )
        t0 = time.perf_counter()
        preds = score_ids(db, bundle, ids, args.chunk_size)
        print(summary(preds))
        print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
