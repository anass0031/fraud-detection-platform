"""Compute (or recompute) monitoring for a date range: daily metrics + drift, active model.

The replay runs this automatically after each day; use the script to recompute, e.g. after
changing the cost settings or adding a drift reference.

Example (inside the backend container):
    python -m app.scripts.compute_monitoring --from 2018-04-28 --to 2018-05-26

Dates are [from, to). Needs artifacts/monitoring/drift_reference-<model>.json for drift
(ml/scripts/build_drift_reference.py); without it only the metrics are computed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass

from sqlalchemy import select

from app.db.models import ModelVersion
from app.db.session import SessionLocal
from app.scripts.replay import days_in
from app.services.monitoring import compute_day, load_drift_reference, reference_path


@dataclass
class ActiveModel:
    model_version_id: int
    version: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--to", dest="end", required=True, type=dt.date.fromisoformat)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--to must be after --from")

    with SessionLocal() as db:
        row = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
        if row is None:
            print("error: no active model", file=sys.stderr)
            sys.exit(1)
        model = ActiveModel(row.id, row.version)
        if load_drift_reference(model.version) is None:
            print(f"warning: no drift reference at {reference_path(model.version)}: metrics only")
        t0 = time.perf_counter()
        for day in days_in(args.start, args.end):
            stored, drift_rows = compute_day(db, model, day)
            db.commit()
            status = "metrics" if stored else "no revealed labels"
            print(f"{day}: {status} · drift rows {drift_rows}", flush=True)
        print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
