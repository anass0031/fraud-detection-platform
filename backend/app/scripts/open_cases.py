"""Open review-queue cases for stored predictions of the active model in a date range.

Scoring opens cases automatically; this script is for predictions stored before the review
queue existed (or to check that nothing is missing). Idempotent: one case per transaction.

Example (inside the backend container):
    python -m app.scripts.open_cases --from 2018-04-28 --to 2018-04-29

Dates are [from, to) on the transaction time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import Case, ModelVersion, Prediction, Transaction
from app.db.session import SessionLocal
from app.services.cases import open_case


def flagged_predictions(
    db: Session, model_version_id: int, start: dt.date, end: dt.date
) -> list[Prediction]:
    """Alerted or flagged predictions in [start, end) whose transaction has no case yet."""
    has_case = select(Case.id).where(Case.transaction_id == Prediction.transaction_id)
    query = (
        select(Prediction)
        .join(Transaction, Prediction.transaction_id == Transaction.id)
        .where(
            Prediction.model_version_id == model_version_id,
            Transaction.transaction_time >= start,
            Transaction.transaction_time < end,
            or_(Prediction.is_alert, Prediction.is_unusual),
            ~has_case.exists(),
        )
        .order_by(Transaction.transaction_dt, Transaction.external_id)
    )
    return list(db.scalars(query))


def open_cases(db: Session, model_version_id: int, start: dt.date, end: dt.date) -> Counter:
    """Open the missing cases (no commit). Returns the count of new cases per reason."""
    opened: Counter = Counter()
    for pred in flagged_predictions(db, model_version_id, start, end):
        case = open_case(db, pred)
        if case is not None:
            opened[case.reason] += 1
    return opened


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--to must be after --from")

    with SessionLocal() as db:
        model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
        if model is None:
            print("error: no active model (run register_model --activate)", file=sys.stderr)
            sys.exit(1)
        opened = open_cases(db, model.id, args.start, args.end)
        db.commit()
        print(
            f"model {model.version}: {sum(opened.values()):,} new cases in "
            f"[{args.start}, {args.end}) · model alerts {opened['model_score']:,} · "
            f"unusual pattern {opened['anomaly']:,}"
        )


if __name__ == "__main__":
    main()
