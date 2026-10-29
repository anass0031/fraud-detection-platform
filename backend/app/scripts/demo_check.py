"""Pre-demo check: one PASS/FAIL line for everything the live demo needs.

Run it on the morning of the demo (and after restoring the demo snapshot):
    docker compose exec backend python -m app.scripts.demo_check

Checks: database, active model + its artifacts and drift reference, the candidate model,
the three demo users can log in (passwords from .env), open cases, monitoring up to the day
before the demo day, the demo day not replayed yet, the API health and the frontend.
Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from fraudml import config
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.db.models import Case, MetricSnapshot, ModelVersion, Transaction, User
from app.db.session import SessionLocal
from app.scripts.seed_users import ANALYSTS
from app.services.monitoring import reference_path

DEMO_DAY = dt.date(2018, 5, 26)  # first day of the last test week, kept for the live demo


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def demo_credentials(env: Mapping[str, str] = os.environ) -> dict[str, str]:
    """E-mail -> password of the seeded demo users (same source as seed_users)."""
    admin = (env.get("SEED_ADMIN_EMAIL") or "admin@fraud.local").strip().lower()
    creds = {admin: env.get("SEED_ADMIN_PASSWORD", "")}
    creds.update({email: env.get("SEED_ANALYST_PASSWORD", "") for email, _ in ANALYSTS})
    return creds


def database_checks(
    db: Session,
    credentials: dict[str, str],
    demo_day: dt.date = DEMO_DAY,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
) -> list[Check]:
    checks: list[Check] = []
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:  # connection refused, wrong password, ...
        return [Check("database", False, f"{type(exc).__name__}: {exc}")]
    checks.append(Check("database", True, "reachable"))

    active = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
    if active is None:
        checks.append(Check("active model", False, "no active row in model_versions"))
    else:
        missing = [
            p
            for p in (
                artifacts_dir / active.artifact_path,
                artifacts_dir / "models" / (active.anomaly_version or "(no anomaly model)"),
            )
            if not p.is_dir()
        ]
        checks.append(
            Check(
                "active model",
                not missing,
                f"{active.version} + {active.anomaly_version}"
                + (f"; missing {', '.join(map(str, missing))}" if missing else ""),
            )
        )
        ref = artifacts_dir / reference_path(active.version).relative_to(config.ARTIFACTS_DIR)
        checks.append(Check("drift reference", ref.is_file(), ref.name))

    candidates = db.scalars(
        select(ModelVersion.version).where(ModelVersion.status == "candidate")
    ).all()
    checks.append(
        Check(
            "candidate model",
            bool(candidates),
            ", ".join(candidates) if candidates else "none (the promotion step cannot be shown)",
        )
    )

    users = {u.email: u for u in db.scalars(select(User).where(User.email.in_(credentials)))}
    for email, password in credentials.items():
        user = users.get(email)
        if user is None:
            ok, detail = False, "missing (run seed_users)"
        elif not user.is_active:
            ok, detail = False, "deactivated"
        elif not password or not verify_password(password, user.password_hash):
            ok, detail = False, "password in .env does not match"
        else:
            ok, detail = True, f"{user.role}, can log in"
        checks.append(Check(f"user {email}", ok, detail))

    open_cases = db.scalar(select(func.count()).where(Case.status != "closed")) or 0
    checks.append(Check("open cases", open_cases > 0, f"{open_cases:,} in the queue"))

    if active is not None:
        days, last = db.execute(
            select(func.count(), func.max(MetricSnapshot.day)).where(
                MetricSnapshot.model_version_id == active.id
            )
        ).one()
        expected = demo_day - dt.timedelta(days=1)
        checks.append(
            Check(
                "monitoring",
                last == expected,
                f"{days} day(s), last {last} (expected {expected})",
            )
        )

    start = dt.datetime.combine(demo_day, dt.time())
    loaded = db.scalar(
        select(func.count()).where(
            Transaction.transaction_time >= start,
            Transaction.transaction_time < start + dt.timedelta(days=1),
        )
    )
    checks.append(
        Check(
            f"demo day {demo_day} not replayed",
            loaded == 0,
            "free" if loaded == 0 else f"{loaded:,} rows already loaded (restore the snapshot)",
        )
    )
    return checks


def http_checks(
    api_url: str,
    frontend_url: str,
    expected_model: str | None,
    get: Callable[..., httpx.Response] = httpx.get,
) -> list[Check]:
    checks: list[Check] = []
    try:
        r = get(f"{api_url}/api/v1/health", timeout=5)
        body = r.json()
        loaded = body.get("model", {}).get("version")
        ok = r.status_code == 200 and body.get("status") == "ok" and loaded == expected_model
        detail = f"HTTP {r.status_code}, {body.get('status')}, model {loaded}"
        if loaded != expected_model:
            detail += f" (database says {expected_model}: restart the backend)"
        checks.append(Check("API health", ok, detail))
    except (httpx.HTTPError, ValueError) as exc:
        checks.append(Check("API health", False, f"{type(exc).__name__}: {exc}"))

    # Vite only answers to known host names, so ask as the browser does (localhost).
    for name, path in (("frontend", "/"), ("frontend API proxy", "/api/v1/health")):
        try:
            r = get(f"{frontend_url}{path}", headers={"Host": "localhost"}, timeout=5)
            checks.append(Check(name, r.status_code == 200, f"HTTP {r.status_code}"))
        except httpx.HTTPError as exc:
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
    return checks


def print_report(checks: list[Check]) -> bool:
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"  {'PASS' if c.ok else 'FAIL'}  {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if not c.ok]
    print(f"RESULT: {'PASS' if not failed else f'FAIL ({len(failed)} of {len(checks)})'}")
    return not failed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--demo-day", type=dt.date.fromisoformat, default=DEMO_DAY)
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--frontend-url", default="http://frontend:5173")
    args = p.parse_args(argv)

    with SessionLocal() as db:
        checks = database_checks(db, demo_credentials(), args.demo_day)
        try:
            active = db.scalar(select(ModelVersion.version).where(ModelVersion.status == "active"))
        except Exception:
            active = None
    checks += http_checks(args.api_url, args.frontend_url, active)
    sys.exit(0 if print_report(checks) else 1)


if __name__ == "__main__":
    main()
