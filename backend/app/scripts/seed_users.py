"""Create the demo users: 1 admin + 2 analysts. Passwords come from .env, never from the code.

    docker compose exec backend python -m app.scripts.seed_users

Environment: SEED_ADMIN_EMAIL (default admin@fraud.local), SEED_ADMIN_PASSWORD,
SEED_ANALYST_PASSWORD (shared by both analysts). Existing users are skipped (passwords are
never reset by this script).
"""

from __future__ import annotations

import os
import sys

from app.db.session import SessionLocal
from app.services import audit
from app.services.users import UserError, create_user, normalise_email
from app.services.users import list_users as all_users

ANALYSTS = (("analyst1@fraud.local", "Analyst One"), ("analyst2@fraud.local", "Analyst Two"))


def _password(var: str) -> str:
    value = os.environ.get(var, "")
    if not value or value.startswith("change-me"):
        sys.exit(f"error: set {var} in .env (10+ characters), then: docker compose up -d backend")
    return value


def main() -> None:
    admin_email = normalise_email(os.environ.get("SEED_ADMIN_EMAIL") or "admin@fraud.local")
    wanted = [(admin_email, "Admin", "admin", _password("SEED_ADMIN_PASSWORD"))]
    analyst_password = _password("SEED_ANALYST_PASSWORD")
    wanted += [(email, name, "analyst", analyst_password) for email, name in ANALYSTS]

    with SessionLocal() as db:
        existing = {u.email for u in all_users(db)}
        created = 0
        for email, name, role, password in wanted:
            if email in existing:
                print(f"  exists   {email}")
                continue
            try:
                user = create_user(db, email, name, role, password)
            except UserError as exc:
                db.rollback()
                sys.exit(f"error: {email}: {exc}")
            audit.record(
                db,
                "user.create",
                entity_type="user",
                entity_id=user.id,
                after={**audit.user_snapshot(user), "via": "seed_users script"},
            )
            created += 1
            print(f"  created  {email} ({role})")
        db.commit()
        print(f"{created} user(s) created; {len(all_users(db))} in total")


if __name__ == "__main__":
    main()
