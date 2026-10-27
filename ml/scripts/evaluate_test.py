"""Evaluate the saved production model on the test period, ONCE.

Uses the artifact as saved (calibrator, alert policy and threshold all fixed on validation):
nothing is re-fitted or re-tuned on test. Baselines (amount rule, logistic regression,
XGBoost defaults) are re-trained on the train period for the PR-curve comparison only and get
threshold-free metrics.

A marker (reports/test_results.json) is written BEFORE scoring; any later run is refused
unless --force is given (which must be logged as a decision in PROGRESS.md).

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/evaluate_test.py [--version lgbm-...] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.evaluate import (
    CostConfig,
    brier,
    daily_topk_mask,
    evaluate_report,
    evaluate_scores,
    fraud_value_caught,
    pr_auc,
    precision_at_capacity,
    roc_auc,
)
from fraudml.models import data as model_data
from fraudml.models.baselines import AmountRule, make_logistic
from fraudml.models.xgb import gpu_available, make_xgb, predict_proba_cpu
from fraudml.registry import MODELS_DIR, load_model

SEED = 42
MARKER = config.REPORTS_DIR / "test_results.json"
OUT = {
    "pr": config.FIGURES_DIR / "test_pr_curves.png",
    "weekly": config.FIGURES_DIR / "test_weekly.png",
    "md": config.REPORTS_DIR / "test_results.md",
}


def ranking_metrics(y, score, day, amount, k: int) -> dict:
    """Threshold-free metrics (fair for models whose threshold was not fixed on validation)."""
    top = daily_topk_mask(score, day, k)
    return {
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "precision_at_capacity": precision_at_capacity(y, score, day, k),
        "fraud_value_caught_at_capacity": fraud_value_caught(y, amount, top),
    }


def weekly_table(y, prob, alert_score, threshold, amount, day, cfg) -> pd.DataFrame:
    week = (day - day.min()) // 7
    rows = []
    for w in np.unique(week):
        s = week == w
        r = evaluate_scores(y[s], alert_score[s], amount[s], day[s], cfg, threshold, False)
        rows.append(
            {
                "week": int(w) + 1,
                "first_day": str(
                    pd.Timestamp(config.REFERENCE_DATETIME) + pd.Timedelta(days=int(day[s].min()))
                )[:10],
                "n": int(s.sum()),
                "fraud_rate": float(y[s].mean()),
                "pr_auc_prob": pr_auc(y[s], prob[s]),
                "alerts_per_day": r["alerts_per_day"],
                "precision_at_threshold": r["precision_at_threshold"],
                "fraud_value_caught_at_threshold": r["fraud_value_caught_at_threshold"],
                "net_savings": r["net_savings"],
                "no_model_cost": r["no_model_cost"],
            }
        )
    return pd.DataFrame(rows)


def plots(y, scores: dict, weekly: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, lw=1.5, label=f"{name} (PR-AUC {pr_auc(y, s):.3f})")
    ax.axhline(y.mean(), color="grey", ls=":", lw=1, label=f"random ({y.mean():.3f})")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Test period (evaluated once): precision-recall curves",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT["pr"], dpi=130)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    x = weekly["week"]
    ax1.plot(x, weekly["pr_auc_prob"], marker="o", label="PR-AUC")
    ax1.plot(x, weekly["precision_at_threshold"], marker="o", label="precision of alerts")
    ax1.plot(x, weekly["fraud_value_caught_at_threshold"], marker="o", label="fraud $ caught")
    ax1.plot(x, weekly["fraud_rate"] * 10, ls="--", color="grey", label="fraud rate × 10")
    ax1.set(
        xlabel="Test week",
        ylabel="Value",
        ylim=(0, 1),
        title="Test period: weekly stability of the production model",
    )
    ax1.set_xticks(x, [f"W{w}\n{d[5:]}" for w, d in zip(x, weekly["first_day"], strict=True)])
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT["weekly"], dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="artifact version (default: latest)")
    parser.add_argument("--force", action="store_true", help="re-run despite the marker")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run on the VALIDATION period, outputs in the scratch dir (code check only)",
    )
    parser.add_argument("--dry-run-dir", default=None, help="output folder for --dry-run")
    args = parser.parse_args()

    period = "test"
    marker = MARKER
    if args.dry_run:
        from pathlib import Path

        out = Path(args.dry_run_dir or config.REPORTS_DIR / "dryrun")
        out.mkdir(parents=True, exist_ok=True)
        period, marker = "validation", out / "dryrun_results.json"
        OUT.update(pr=out / "pr.png", weekly=out / "weekly.png", md=out / "dryrun.md")
    elif MARKER.exists() and not args.force:
        raise SystemExit(f"test period already evaluated ({MARKER}); use --force and log why")
    version = args.version or sorted(p.name for p in MODELS_DIR.glob("lgbm-*") if p.is_dir())[-1]
    marker.write_text(
        json.dumps(
            {
                "status": "started",
                "version": version,
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
        ),
        "utf-8",
    )

    model = load_model(version)
    meta = model.metadata
    cfg = CostConfig(**meta["cost_config"])
    test = model_data.load_period(period, allow_test=True)
    y = test.y.to_numpy()
    amount = test.meta["amount"].to_numpy()
    day = test.meta["day"].to_numpy()
    flag = test.meta[config.EVAL_FLAG_COL].to_numpy()
    k = cfg.capacity_per_day

    prob = model.predict_proba(test.X)
    raw = model.raw_score(test.X)
    alert_score = model.alert_score(prob, amount)
    threshold = float(meta["threshold"])

    policy = evaluate_report(y, alert_score, amount, day, flag, cfg, threshold, False)
    prob_metrics = {
        "overall": {
            **ranking_metrics(y, prob, day, amount, k),
            "brier": brier(y, prob),
            "pr_auc_uncalibrated": pr_auc(y, raw),
            "mean_prob": float(prob.mean()),
            "fraud_rate": float(y.mean()),
        },
        "excl_fraud_uids": {
            **ranking_metrics(y[~flag], prob[~flag], day[~flag], amount[~flag], k),
            "brier": brier(y[~flag], prob[~flag]),
        },
    }

    # Baselines, re-trained on train (threshold-free comparison only).
    train = model_data.load_period("train")
    rule = AmountRule().fit(train.meta["amount"])
    logreg = make_logistic(train.X, SEED).fit(train.X, train.y)
    device = "cuda" if gpu_available() else "cpu"
    xgbm = make_xgb(SEED, device).fit(train.X, train.y)
    scores = {
        f"production LightGBM ({version})": prob,
        "XGBoost defaults": predict_proba_cpu(xgbm, test.X),
        "logistic regression": logreg.predict_proba(test.X)[:, 1],
        "rule: amount > train p99": rule.score(amount),
    }
    comparison = {
        name: {
            "overall": ranking_metrics(y, s, day, amount, k),
            "excl_fraud_uids": ranking_metrics(y[~flag], s[~flag], day[~flag], amount[~flag], k),
        }
        for name, s in scores.items()
    }

    weekly = weekly_table(y, prob, alert_score, threshold, amount, day, cfg)
    plots(y, scores, weekly)

    result = {
        "status": "done",
        "version": version,
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "test_window": config.SPLIT_DATES["test"],
        "n": int(len(y)),
        "policy": meta["policy"],
        "threshold": threshold,
        "policy_metrics": {idx: row.to_dict() for idx, row in policy.iterrows()},
        "probability_metrics": prob_metrics,
        "comparison_threshold_free": comparison,
        "weekly": weekly.to_dict(orient="records"),
        "validation_metrics": meta.get("validation_metrics"),
    }
    result["period"] = period
    marker.write_text(json.dumps(result, indent=2, default=float), "utf-8")

    cols = [
        "population",
        "pr_auc",
        "alerts_per_day",
        "max_alerts_per_day",
        "precision_at_threshold",
        "recall_at_threshold",
        "fraud_value_caught_at_threshold",
        "total_cost",
        "no_model_cost",
        "net_savings",
    ]
    comp = pd.DataFrame(
        [{"model": n, "population": p, **v} for n, d in comparison.items() for p, v in d.items()]
    )
    md = [
        f"# Test-period results ({version}, evaluated once on {result['evaluated_at'][:10]})",
        "",
        f"Alert policy `{meta['policy']}` with threshold {threshold:.4g}, fixed on validation.",
        "`pr_auc` in the first table is computed on the alert score (expected loss).",
        "",
        policy.reset_index()[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Calibrated probability (production model)",
        "",
        pd.DataFrame(prob_metrics).T.to_markdown(floatfmt=".4f"),
        "",
        "## Threshold-free comparison (all models trained on the train period)",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Weekly stability",
        "",
        weekly.to_markdown(index=False, floatfmt=".4f"),
    ]
    OUT["md"].write_text("\n".join(md) + "\n", "utf-8")
    o = policy.loc["overall"]
    print(
        f"TEST {version}: alerts/day {o.alerts_per_day:.1f} (max {o.max_alerts_per_day:.0f}) · "
        f"precision {o.precision_at_threshold:.3f} · fraud $ caught "
        f"{o.fraud_value_caught_at_threshold:.1%} · net savings ${o.net_savings:,.0f} of "
        f"${o.no_model_cost:,.0f} · PR-AUC(p) {prob_metrics['overall']['pr_auc']:.4f} · "
        f"excl {prob_metrics['excl_fraud_uids']['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
