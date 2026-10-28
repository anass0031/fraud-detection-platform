"""Export replayed, labelled transactions for offline retraining (Parquet written to stdout).

One row per transaction of [from, to) that has stored features (it was scored):
TransactionID, time, amount, customer key, the model features, the label and where the label
comes from, plus the active model's stored score / alert (for the champion side of the
comparison).

Label rule: revealed ground truth; otherwise an analyst's final decision (confirmed fraud = 1,
false positive = 0); otherwise no label (the row is exported with label = null).

The data folder is read-only in the container, so redirect stdout on the host:
    docker compose exec -T backend python -m app.scripts.export_training_data \
        --from 2018-04-28 --to 2018-05-26 > data/processed/replayed_labelled.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Case, Decision, ModelVersion, Prediction, Transaction
from app.db.session import SessionLocal

DECISION_LABEL = {"confirmed_fraud": 1, "false_positive": 0}


def final_decisions(db: Session, start: dt.datetime, end: dt.datetime) -> dict[int, int]:
    """Transaction id -> label from the latest non-escalate decision of its case, for the
    transactions of [start, end) (filtered by a join: there are few decisions)."""
    rows = db.execute(
        select(Case.transaction_id, Decision.decision)
        .join(Decision, Decision.case_id == Case.id)
        .join(Transaction, Transaction.id == Case.transaction_id)
        .where(
            Transaction.transaction_time >= start,
            Transaction.transaction_time < end,
            Decision.decision != "escalate",
        )
        .order_by(Decision.id)
    ).all()
    return {t: DECISION_LABEL[d] for t, d in rows}  # later rows win


def export_frame(db: Session, start: dt.date, end: dt.date, model_id: int | None) -> pd.DataFrame:
    t0, t1 = dt.datetime.combine(start, dt.time()), dt.datetime.combine(end, dt.time())
    rows = db.execute(
        select(
            Transaction.id,
            Transaction.external_id,
            Transaction.transaction_time,
            Transaction.amount,
            Transaction.customer_key,
            Transaction.features,
            Transaction.ground_truth,
            Transaction.ground_truth_revealed_at,
            Prediction.fraud_score,
            Prediction.expected_loss,
            Prediction.is_alert,
        )
        .outerjoin(
            Prediction,
            (Prediction.transaction_id == Transaction.id)
            & (Prediction.model_version_id == (model_id or -1)),
        )
        .where(
            Transaction.transaction_time >= t0,
            Transaction.transaction_time < t1,
            Transaction.features.is_not(None),
        )
        .order_by(Transaction.transaction_dt, Transaction.external_id)
    ).all()
    if not rows:
        return pd.DataFrame()
    decisions = final_decisions(db, t0, t1)

    labels, sources = [], []
    for r in rows:
        if r.ground_truth_revealed_at is not None and r.ground_truth is not None:
            labels.append(int(r.ground_truth))
            sources.append("ground_truth")
        elif r.id in decisions:
            labels.append(decisions[r.id])
            sources.append("decision")
        else:
            labels.append(None)
            sources.append("none")

    meta = pd.DataFrame(
        {
            "TransactionID": [r.external_id for r in rows],
            "transaction_time": [r.transaction_time for r in rows],
            "amount": [float(r.amount) for r in rows],
            "customer_key": [r.customer_key for r in rows],
            "label": pd.array(labels, dtype="Int8"),
            "label_source": sources,
            "champion_score": [r.fraud_score for r in rows],
            "champion_expected_loss": [r.expected_loss for r in rows],
            "champion_alert": [r.is_alert for r in rows],
        }
    )
    features = pd.DataFrame([r.features for r in rows])
    return pd.concat([meta, features], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--to must be after --from")
    if sys.stdout.isatty():
        parser.error("stdout is a terminal: redirect it to a .parquet file")

    with SessionLocal() as db:
        active = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
        df = export_frame(db, args.start, args.end, active.id if active else None)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    sys.stdout.buffer.write(buf.getvalue())
    counts = df["label_source"].value_counts().to_dict() if len(df) else {}
    print(
        f"exported {len(df):,} rows [{args.start}, {args.end}) · labels {counts} · "
        f"champion {active.version if active else None}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
