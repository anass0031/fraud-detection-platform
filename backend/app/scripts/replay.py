"""Replay test-period days through the real system, one day at a time.

For each day, in order: load its transactions (source `replay`) -> score them with the active
model (this opens review cases) -> reveal that day's labels -> run the monitoring hooks.
Every stage skips work that is already done, so the script can be stopped and re-run.

Example (inside the backend container):
    python -m app.scripts.replay --from 2018-04-29 --to 2018-05-26 --workers 4

`--workers N` scores each day with N processes (one math thread each). Once a day is loaded,
its rows are independent (features only read strictly earlier history), so the results are
the same as scoring them one by one.

Dates are [from, to). Days must be replayed in order: features use each customer's earlier
transactions, so the day before `--from` must already be in the database.
"""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import signal
import sys
import time
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass

import pandas as pd
from fraudml import config
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import Case, Transaction
from app.db.session import SessionLocal
from app.scripts.score_batch import pending_ids, score_ids
from app.services import audit
from app.services.model_store import ModelStore, ScoringBundle
from app.services.transactions_loader import load_frame, read_period

TEST_START = dt.date.fromisoformat(config.SPLIT_DATES["test"][0])
TEST_END = dt.date.fromisoformat(config.SPLIT_DATES["test"][1])

# Called after a day is replayed (monitoring computes that day's metrics and drift).
DayHook = Callable[[Session, ScoringBundle, dt.date], None]


@dataclass
class DayResult:
    day: dt.date
    rows: int
    loaded: int
    scored: int
    alerts: int
    unusual: int
    cases: int
    revealed: int
    seconds: float

    def line(self) -> str:
        return (
            f"{self.day}: {self.rows:,} rows · loaded {self.loaded:,} · scored {self.scored:,} "
            f"(alerts {self.alerts}, unusual {self.unusual}) · new cases {self.cases} · "
            f"labels revealed {self.revealed:,} · {self.seconds:.0f} s"
        )


def day_bounds(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime.combine(day, dt.time())
    return start, start + dt.timedelta(days=1)


def reveal_labels(db: Session, day: dt.date) -> int:
    """Mark the day's ground truth as known (used by monitoring only, never by the UI)."""
    start, end = day_bounds(day)
    result = db.execute(
        update(Transaction)
        .where(
            Transaction.transaction_time >= start,
            Transaction.transaction_time < end,
            Transaction.ground_truth.is_not(None),
            Transaction.ground_truth_revealed_at.is_(None),
        )
        .values(ground_truth_revealed_at=dt.datetime.now(dt.UTC))
    )
    return result.rowcount or 0


def count_cases(db: Session, day: dt.date) -> int:
    start, end = day_bounds(day)
    return (
        db.scalar(
            select(func.count())
            .select_from(Case)
            .join(Transaction, Case.transaction_id == Transaction.id)
            .where(Transaction.transaction_time >= start, Transaction.transaction_time < end)
        )
        or 0
    )


# ---------------------------------------------------------------- parallel scoring

CHUNK = 200
_worker: dict = {}


def _init_worker(expected_version: str) -> None:
    """Runs once in each worker process: its own DB session and model."""
    db = SessionLocal()
    bundle = ModelStore().load_active(db)
    if bundle is None or bundle.version != expected_version:
        raise RuntimeError(
            f"worker loaded {bundle and bundle.version}, expected {expected_version}"
        )
    _worker.update(db=db, bundle=bundle)


def _score_chunk(ids: list[int]) -> list[tuple[bool, bool]]:
    preds = score_ids(_worker["db"], _worker["bundle"], ids, chunk_size=len(ids), verbose=False)
    return [(p.is_alert, p.is_unusual) for p in preds]


def make_pool(workers: int, version: str) -> Executor:
    # "spawn": fresh processes, no database connection copied from the parent.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(version,)
    )


def stop_workers_on_signal(pool: Executor) -> None:
    """On kill / Ctrl+C, end the worker processes too (they would otherwise keep scoring)."""

    def handler(signum: int, _frame: object) -> None:
        for proc in list(getattr(pool, "_processes", {}).values()):
            proc.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def score_pending(
    db: Session, bundle: ScoringBundle, day: dt.date, pool: Executor | None = None
) -> list[tuple[bool, bool]]:
    """Score the day's rows that have no prediction yet. Returns (is_alert, is_unusual) each."""
    start, end = day_bounds(day)
    ids = pending_ids(db, bundle.model_version_id, start, end)
    if pool is None:
        return [(p.is_alert, p.is_unusual) for p in score_ids(db, bundle, ids)]
    chunks = [ids[i : i + CHUNK] for i in range(0, len(ids), CHUNK)]
    return [flags for part in pool.map(_score_chunk, chunks) for flags in part]


def replay_day(
    db: Session,
    bundle: ScoringBundle,
    day: dt.date,
    frame: pd.DataFrame,
    hooks: tuple[DayHook, ...] = (),
    pool: Executor | None = None,
) -> DayResult:
    """Replay one day; `frame` holds that day's rows from the processed Parquet file."""
    t0 = time.perf_counter()
    loaded = load_frame(db.connection(), frame, "replay") if len(frame) else 0
    db.commit()

    cases_before = count_cases(db, day)
    preds = score_pending(db, bundle, day, pool)
    cases = count_cases(db, day) - cases_before

    revealed = reveal_labels(db, day)
    if loaded or preds or revealed:
        audit.record(
            db,
            "replay.day",
            entity_type="day",
            entity_id=day.isoformat(),
            after={"loaded": loaded, "scored": len(preds), "cases": cases, "revealed": revealed},
        )
    db.commit()

    for hook in hooks:
        hook(db, bundle, day)
        db.commit()

    return DayResult(
        day=day,
        rows=len(frame),
        loaded=loaded,
        scored=len(preds),
        alerts=sum(alert for alert, _ in preds),
        unusual=sum(unusual for _, unusual in preds),
        cases=cases,
        revealed=revealed,
        seconds=time.perf_counter() - t0,
    )


def previous_day_loaded(db: Session, day: dt.date) -> bool:
    start, _ = day_bounds(day - dt.timedelta(days=1))
    return bool(
        db.scalar(
            select(func.count())
            .select_from(Transaction)
            .where(
                Transaction.transaction_time >= start,
                Transaction.transaction_time < start + dt.timedelta(days=1),
            )
        )
    )


def days_in(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days)]


def default_hooks() -> tuple[DayHook, ...]:
    """Monitoring hooks, if the monitoring service exists (it is added after the replay)."""
    try:
        from app.services.monitoring import compute_day  # noqa: PLC0415
    except ImportError:
        return ()
    return (compute_day,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--no-monitoring", action="store_true", help="skip the monitoring hooks")
    parser.add_argument("--workers", type=int, default=1, help="scoring processes per day")
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--to must be after --from")
    if args.start < TEST_START or args.end > TEST_END:
        parser.error(f"replay days must be inside the test period [{TEST_START}, {TEST_END})")

    hooks = () if args.no_monitoring else default_hooks()
    with SessionLocal() as db:
        bundle = ModelStore().load_active(db)
        if bundle is None:
            print("error: no active model (run register_model --activate)", file=sys.stderr)
            sys.exit(1)
        if not previous_day_loaded(db, args.start):
            print(
                f"error: {args.start - dt.timedelta(days=1)} is not in the database; replay days "
                "in order (customer history must be complete)",
                file=sys.stderr,
            )
            sys.exit(1)

        t0 = time.perf_counter()
        frame = read_period(args.start.isoformat(), args.end.isoformat())
        by_day = frame.groupby(frame[config.DATETIME_COL].dt.date)
        print(
            f"model {bundle.version}: replaying {len(frame):,} rows, "
            f"{args.start} -> {args.end} (excluded); monitoring hooks: {len(hooks)}; "
            f"scoring workers: {args.workers}"
        )
        pool = make_pool(args.workers, bundle.version) if args.workers > 1 else None
        if pool is not None:
            stop_workers_on_signal(pool)
        try:
            for day in days_in(args.start, args.end):
                part = by_day.get_group(day) if day in by_day.groups else frame.iloc[0:0]
                print(replay_day(db, bundle, day, part, hooks, pool).line(), flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
