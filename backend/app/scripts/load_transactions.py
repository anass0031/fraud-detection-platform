"""Load transactions of a date range from data/processed/train.parquet into PostgreSQL.

Examples (inside the backend container):
    python -m app.scripts.load_transactions --until 2018-04-28 --source history
    python -m app.scripts.load_transactions --from 2018-04-28 --to 2018-04-29 --source batch

Dates are [from, to): the end day is excluded. Already stored transactions are skipped.
"""

from __future__ import annotations

import argparse
import time

from fraudml import config
from sqlalchemy import func, select

from app.db.models import TRANSACTION_SOURCES, Transaction
from app.db.session import engine
from app.services.transactions_loader import TRAIN_PARQUET, load_frame, read_period


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", help="first day (YYYY-MM-DD), default: start")
    parser.add_argument("--to", "--until", dest="end", help="end day, excluded (YYYY-MM-DD)")
    parser.add_argument("--source", required=True, choices=TRANSACTION_SOURCES)
    args = parser.parse_args()

    if args.source == "history" and (args.end is None or args.end > config.SPLIT_DATES["test"][0]):
        parser.error("history rows must end on or before the test period start (2018-04-28)")

    t0 = time.perf_counter()
    df = read_period(args.start, args.end)
    print(f"read {len(df):,} rows from {TRAIN_PARQUET.name} in {time.perf_counter() - t0:.1f} s")
    if df.empty:
        return
    print(
        f"period: {df[config.DATETIME_COL].min()} -> {df[config.DATETIME_COL].max()}, "
        f"fraud rate {df[config.TARGET].mean():.2%}"
    )

    t1 = time.perf_counter()
    with engine.begin() as conn:
        inserted = load_frame(conn, df, args.source)
    elapsed = time.perf_counter() - t1
    print(
        f"inserted {inserted:,} new rows ({len(df) - inserted:,} already stored) "
        f"as source={args.source!r} in {elapsed:.1f} s"
    )

    with engine.connect() as conn:
        counts = conn.execute(
            select(Transaction.source, func.count()).group_by(Transaction.source)
        ).all()
    print("transactions in DB:", {s: n for s, n in counts})


if __name__ == "__main__":
    main()
