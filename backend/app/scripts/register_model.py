"""Register a trained model artifact (artifacts/models/<version>/) in model_versions.

Example (inside the backend container):
    python -m app.scripts.register_model lgbm-2026.10.27-1 --anomaly iforest-2026.10.27-1 --activate
    python -m app.scripts.register_model lgbm-2026.10.27-1 --activate   # already registered

The running API loads the active model at startup: restart the backend after activating.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.db.models import ModelVersion
from app.db.session import SessionLocal
from app.services.model_store import RegistryError, activate_version, register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="artifact folder name, e.g. lgbm-2026.10.27-1")
    parser.add_argument("--anomaly", help="Isolation Forest artifact linked to this model")
    parser.add_argument("--activate", action="store_true", help="make it the active model")
    args = parser.parse_args()

    with SessionLocal() as db:
        exists = db.scalar(select(ModelVersion).where(ModelVersion.version == args.version))
        try:
            if exists is None:
                row = register(db, args.version, args.anomaly, activate=args.activate)
                print(f"registered {row.version} (anomaly: {row.anomaly_version})")
            elif args.activate:
                if args.anomaly and args.anomaly != exists.anomaly_version:
                    raise RegistryError(
                        f"{args.version} is registered with anomaly {exists.anomaly_version}"
                    )
                row = activate_version(db, args.version)
            else:
                raise RegistryError(f"{args.version} is already registered")
            db.commit()
        except RegistryError as exc:
            db.rollback()
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

        for r in db.scalars(select(ModelVersion).order_by(ModelVersion.id)):
            print(
                f"  {r.version:<24} {r.status:<10} anomaly={r.anomaly_version} "
                f"policy={r.policy} threshold={r.threshold:.2f}"
            )
    if args.activate:
        print("restart the backend to load it: docker compose restart backend")


if __name__ == "__main__":
    main()
