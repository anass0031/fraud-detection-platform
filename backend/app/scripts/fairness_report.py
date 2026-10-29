"""Error rates of the active model by group, on replayed days with revealed labels (Markdown).

The data has no protected attributes, but some features can act as proxies for customer
groups (product, card type). This shows whether alerts and errors fall evenly on them.

Example (inside the backend container):
    python -m app.scripts.fairness_report --from 2018-04-28 --to 2018-05-26
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ModelVersion, Prediction, Transaction
from app.db.session import SessionLocal

GROUPS = {"product_code": "Product code", "card_type": "Card type", "card_network": "Card network"}
MIN_ROWS = 500  # smaller groups are merged into "other" (rates would be noise)


def labelled_rows(db: Session, model_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = db.execute(
        select(
            Transaction.product_code,
            Transaction.card_type,
            Transaction.card_network,
            Transaction.ground_truth,
            Prediction.is_alert,
        )
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .where(
            Prediction.model_version_id == model_id,
            Transaction.transaction_time >= dt.datetime.combine(start, dt.time()),
            Transaction.transaction_time < dt.datetime.combine(end, dt.time()),
            Transaction.ground_truth_revealed_at.is_not(None),
            Transaction.ground_truth.is_not(None),
        )
    ).all()
    return pd.DataFrame(rows, columns=[*GROUPS, "fraud", "alert"])


def group_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """n, fraud rate, alert rate, precision, recall and false-alarm rate per group."""
    g = df[column].fillna("(missing)")
    sizes = g.value_counts()
    g = g.where(g.map(sizes) >= MIN_ROWS, "other (small groups)")
    out = []
    for name, part in df.groupby(g):
        y, a = part["fraud"].astype(bool), part["alert"].astype(bool)
        tp, fp = int((y & a).sum()), int((~y & a).sum())
        legit = int((~y).sum())
        out.append(
            {
                "group": name,
                "transactions": len(part),
                "fraud rate": y.mean(),
                "alert rate": a.mean(),
                "precision": tp / a.sum() if a.sum() else float("nan"),
                "recall": tp / y.sum() if y.sum() else float("nan"),
                "false-alarm rate": fp / legit if legit else float("nan"),
            }
        )
    return pd.DataFrame(out).sort_values("transactions", ascending=False)


def to_markdown(table: pd.DataFrame) -> str:
    pct = ["fraud rate", "alert rate", "precision", "recall", "false-alarm rate"]
    lines = ["| Group | Transactions | " + " | ".join(c.capitalize() for c in pct) + " |",
             "|---|---:|" + "---:|" * len(pct)]  # fmt: skip
    for _, r in table.iterrows():
        cells = [f"{r[c]:.1%}" if pd.notna(r[c]) else "—" for c in pct]
        lines.append(f"| {r['group']} | {r['transactions']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    args = parser.parse_args()
    with SessionLocal() as db:
        model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
        if model is None:
            raise SystemExit("no active model")
        df = labelled_rows(db, model.id, args.start, args.end)
    print(f"Model {model.version}, {len(df):,} labelled transactions [{args.start}, {args.end})\n")
    for column, title in GROUPS.items():
        print(f"### By {title.lower()}\n")
        print(to_markdown(group_table(df, column)))
        print()


if __name__ == "__main__":
    main()
