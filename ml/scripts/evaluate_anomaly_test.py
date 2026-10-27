"""Check the anomaly safety net on the test period, ONCE.

Uses the saved artifacts as they are: the production model (alert rule fixed on validation)
and the Isolation Forest (cut-off fixed on train). Nothing is fitted or tuned on test.

A marker (reports/anomaly_test_results.json) is written BEFORE scoring; later runs are refused
unless --force is given (which must be logged as a decision). `--dry-run` runs the same code on
the validation period and writes to a separate folder (code check only).

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/evaluate_anomaly_test.py [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.evaluate import CostConfig, pr_auc
from fraudml.models import data as model_data
from fraudml.models.anomaly import safety_net, safety_net_metrics
from fraudml.registry import MODELS_DIR, load_anomaly, load_model

MARKER = config.REPORTS_DIR / "anomaly_test_results.json"
REPORT = config.REPORTS_DIR / "anomaly_test_results.md"


def latest(prefix: str) -> str:
    return sorted(p.name for p in MODELS_DIR.glob(f"{prefix}*") if p.is_dir())[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-run despite the marker")
    parser.add_argument("--dry-run", action="store_true", help="run on validation (code check)")
    parser.add_argument("--dry-run-dir", default=None, help="output folder for --dry-run")
    args = parser.parse_args()

    period, marker, report = "test", MARKER, REPORT
    if args.dry_run:
        out = Path(args.dry_run_dir or config.REPORTS_DIR / "dryrun")
        out.mkdir(parents=True, exist_ok=True)
        period, marker, report = (
            "validation",
            out / "anomaly_dryrun.json",
            out / "anomaly_dryrun.md",
        )
    elif MARKER.exists() and not args.force:
        raise SystemExit(f"safety net already checked on test ({MARKER}); use --force and log why")

    model_version, anomaly_version = latest("lgbm-"), latest("iforest-")
    started = {
        "status": "started",
        "period": period,
        "model": model_version,
        "anomaly": anomaly_version,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    marker.write_text(json.dumps(started), "utf-8")

    prod, detector = load_model(model_version), load_anomaly(anomaly_version)
    cfg = CostConfig(**prod.metadata["cost_config"])
    data = model_data.load_period(period, allow_test=True)
    y = data.y.to_numpy()
    amount = data.meta["amount"].to_numpy()
    day = data.meta["day"].to_numpy()
    flag = data.meta[config.EVAL_FLAG_COL].to_numpy()

    main_alert = prod.is_alert(prod.predict_proba(data.X), amount)
    score = detector.score(data.X)
    extra = safety_net(main_alert, score, detector.cutoff)
    assert not (extra & main_alert).any()

    rows = {
        pop: safety_net_metrics(y[k], amount[k], day[k], main_alert[k], extra[k], cfg)
        for pop, k in (("overall", np.ones_like(flag, dtype=bool)), ("excl_fraud_uids", ~flag))
    }
    alone = {
        "pr_auc": pr_auc(y, score),
        "flagged_share": float((score >= detector.cutoff).mean()),
        "precision_flagged": float(y[score >= detector.cutoff].mean()),
    }
    result = {
        **started,
        "status": "done",
        "cutoff": detector.cutoff,
        "safety_net": rows,
        "isolation_forest_alone": alone,
        "validation_metrics": detector.metadata.get("validation_metrics"),
    }
    marker.write_text(json.dumps(result, indent=2, default=float), "utf-8")

    table = pd.DataFrame(rows).T
    cols = [
        "extra_alerts_per_day",
        "extra_alerts_max_day",
        "extra_frauds",
        "extra_precision",
        "extra_lift",
        "share_of_missed_frauds_caught",
        "main_fraud_value_caught",
        "combined_fraud_value_caught",
        "delta_net_savings",
    ]
    report.write_text(
        f"# Anomaly safety net: {period}-period check ({anomaly_version} with {model_version}, "
        f"run once on {result['started_at'][:10]})\n\n"
        + table[cols].to_markdown(floatfmt=".4f")
        + f"\n\nIsolation Forest alone: PR-AUC {alone['pr_auc']:.4f}; "
        f"{alone['flagged_share']:.2%} of rows above the cut-off, "
        f"precision {alone['precision_flagged']:.4f} (fraud rate {y.mean():.4f}).\n",
        "utf-8",
    )
    o = rows["overall"]
    print(
        f"{period.upper()}: safety net +{o['extra_alerts_per_day']:.1f} alerts/day "
        f"(max {o['extra_alerts_max_day']}) · {o['extra_frauds']} extra frauds · precision "
        f"{o['extra_precision']:.3f} (lift {o['extra_lift']:.2f}) · fraud $ "
        f"{o['main_fraud_value_caught']:.1%} -> {o['combined_fraud_value_caught']:.1%} · "
        f"net savings change ${o['delta_net_savings']:,.0f} · IF alone PR-AUC "
        f"{alone['pr_auc']:.4f} ({alone['flagged_share']:.2%} flagged)"
    )


if __name__ == "__main__":
    main()
