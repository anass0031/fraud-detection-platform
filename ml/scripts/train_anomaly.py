"""Isolation Forest safety net: fit on train (no labels), fix the cut-off, measure on validation.

1. Fit the detector on the train period (labels never used) and fix the cut-off at the 99.5th
   percentile of train anomaly scores.
2. On validation, apply the production model's alert rule, then the safety net
   (not alerted by the main rule AND anomaly score >= cut-off), and measure what it adds.
3. `--save`: write the detector artifact (artifacts/models/iforest-<date>-<n>/).

The test period is never read by this script.

Writes reports/anomaly_validation.{csv,md} and reports/figures/anomaly_{score_distribution,
capture}.png.

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/train_anomaly.py [--model lgbm-...] [--save]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.evaluate import CostConfig, pr_auc
from fraudml.features.pipeline import FEATURES_CURATED
from fraudml.models import data as model_data
from fraudml.models.anomaly import (
    TOP_SHARE,
    anomaly_score,
    fit_cutoff,
    make_iforest,
    safety_net,
    safety_net_metrics,
)
from fraudml.registry import MODELS_DIR, load_anomaly, load_model, save_anomaly

SEED = 42
SENSITIVITY = (0.0025, 0.005, 0.01)  # for information only; the rule uses TOP_SHARE


def latest(prefix: str) -> str:
    names = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not names:
        raise SystemExit(f"no artifact starting with {prefix!r} in {MODELS_DIR}")
    return names[-1]


def plot_distribution(train_scores, cutoff, val_scores, y_val, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(
        min(train_scores.min(), val_scores.min()), max(train_scores.max(), val_scores.max()), 80
    )
    ax.hist(train_scores, bins=bins, density=True, alpha=0.35, color="grey", label="train (all)")
    ax.hist(
        val_scores[y_val == 0],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label="validation legit",
    )
    ax.hist(
        val_scores[y_val == 1],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label="validation fraud",
    )
    ax.axvline(
        cutoff, color="tab:red", ls="--", lw=1.2, label=f"cut-off (top {TOP_SHARE:.1%} of train)"
    )
    ax.set(
        yscale="log",
        xlabel="Anomaly score (higher = more unusual)",
        ylabel="Density (log)",
        title="Isolation Forest anomaly scores",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_capture(row: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = row["main_fraud_value_caught"]
    extra = row["extra_fraud_value_caught"]
    missed = 1 - main - extra
    fig, ax = plt.subplots(figsize=(7, 2.6))
    left = 0.0
    for val, lab, col in (
        (main, "main model", "tab:blue"),
        (extra, "safety net (extra)", "tab:red"),
        (missed, "missed", "lightgrey"),
    ):
        ax.barh([0], [val], left=left, color=col, label=f"{lab}: {val:.1%}")
        left += val
    ax.set(
        xlim=(0, 1),
        yticks=[],
        title="Share of validation fraud value ($) caught: main model vs safety net",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="production model version (default: latest lgbm-*)")
    parser.add_argument("--save", action="store_true", help="save the detector artifact")
    args = parser.parse_args()
    t0 = time.perf_counter()

    # 1. Fit on train without labels; fix the cut-off from train scores.
    train = model_data.load_period("train")
    detector = make_iforest(train.X, SEED).fit(train.X)
    s_train = anomaly_score(detector, train.X)
    cutoff = fit_cutoff(s_train, TOP_SHARE)
    flagged_train = s_train >= cutoff
    print(
        f"fit on {len(train.X):,} train rows in {time.perf_counter() - t0:.0f}s · cut-off "
        f"{cutoff:.4f} flags {flagged_train.mean():.3%} of train "
        f"(fraud rate among them {train.y[flagged_train].mean():.2%} vs {train.y.mean():.2%})"
    )

    # 2. Validation: main rule + safety net.
    version = args.model or latest("lgbm-")
    prod = load_model(version)
    cfg = CostConfig(**prod.metadata["cost_config"])
    val = model_data.load_period("validation")
    y = val.y.to_numpy()
    amount = val.meta["amount"].to_numpy()
    day = val.meta["day"].to_numpy()
    flag = val.meta[config.EVAL_FLAG_COL].to_numpy()
    prob = prod.predict_proba(val.X)
    main_alert = prod.is_alert(prob, amount)
    s_val = anomaly_score(detector, val.X)
    extra = safety_net(main_alert, s_val, cutoff)
    assert not (extra & main_alert).any(), "safety net must only flag non-alerted rows"

    rows = []
    for pop, keep in (("overall", np.ones_like(flag, dtype=bool)), ("excl_fraud_uids", ~flag)):
        r = safety_net_metrics(y[keep], amount[keep], day[keep], main_alert[keep], extra[keep], cfg)
        rows.append({"top_share": TOP_SHARE, "population": pop, **r})
    for share in SENSITIVITY:
        if share == TOP_SHARE:
            continue
        ex = safety_net(main_alert, s_val, fit_cutoff(s_train, share))
        rows.append(
            {
                "top_share": share,
                "population": "overall",
                **safety_net_metrics(y, amount, day, main_alert, ex, cfg),
            }
        )
    table = pd.DataFrame(rows)
    o = table.iloc[0]
    alone = {
        "pr_auc": pr_auc(y, s_val),
        "precision_top_share": float(y[s_val >= cutoff].mean()),
        "flagged_share_val": float((s_val >= cutoff).mean()),
    }
    keep_rule = bool(o.extra_lift > 2 or o.delta_net_savings >= 0)
    decision = "keep" if keep_rule else "drop"

    table.to_csv(config.REPORTS_DIR / "anomaly_validation.csv", index=False)
    cols = [
        "top_share",
        "population",
        "extra_alerts_per_day",
        "extra_alerts_max_day",
        "extra_frauds",
        "extra_precision",
        "extra_lift",
        "share_of_missed_frauds_caught",
        "extra_fraud_value_caught",
        "combined_fraud_value_caught",
        "delta_net_savings",
    ]
    (config.REPORTS_DIR / "anomaly_validation.md").write_text(
        "# Anomaly safety net: validation results (generated by train_anomaly.py)\n\n"
        f"Production model `{version}` (alert rule: {prod.metadata['policy']} >= "
        f"{prod.metadata['threshold']:.4g}). Safety net: not alerted by the main rule AND "
        f"anomaly score >= cut-off {cutoff:.4f} (top {TOP_SHARE:.1%} of train scores).\n\n"
        + table[cols].to_markdown(index=False, floatfmt=".4f")
        + f"\n\nIsolation Forest alone on validation: PR-AUC {alone['pr_auc']:.4f}, "
        f"precision in the flagged rows {alone['precision_top_share']:.4f} "
        f"({alone['flagged_share_val']:.2%} of rows flagged; fraud rate {y.mean():.4f}).\n\n"
        f"Keep rule (lift > 2 or net savings change >= 0): **{decision}**.\n",
        "utf-8",
    )
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_distribution(
        s_train, cutoff, s_val, y, config.FIGURES_DIR / "anomaly_score_distribution.png"
    )
    plot_capture(o.to_dict(), config.FIGURES_DIR / "anomaly_capture.png")

    print(
        f"validation: main {o.main_alerts_per_day:.1f} alerts/day · safety net "
        f"+{o.extra_alerts_per_day:.1f}/day (max {o.extra_alerts_max_day}) · "
        f"{o.extra_frauds} extra frauds · precision {o.extra_precision:.3f} "
        f"(lift {o.extra_lift:.2f}) · {o.share_of_missed_frauds_caught:.1%} of missed frauds · "
        f"fraud $ {o.main_fraud_value_caught:.1%} -> {o.combined_fraud_value_caught:.1%} · "
        f"net savings change ${o.delta_net_savings:,.0f} · "
        f"IF alone PR-AUC {alone['pr_auc']:.4f} · decision: {decision}"
    )

    if args.save:
        folder = save_anomaly(
            detector,
            {
                "features": list(FEATURES_CURATED),
                "cutoff": cutoff,
                "top_share": TOP_SHARE,
                "params": {"n_estimators": 200, "random_state": SEED},
                "training_data_window": config.SPLIT_DATES["train"],
                "main_model": version,
                "keep_rule_passed": keep_rule,
                "validation_metrics": {
                    k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                    for k, v in o.to_dict().items()
                },
            },
        )
        loaded = load_anomaly(folder.name, folder.parent)
        assert np.array_equal(loaded.score(val.X), s_val), "anomaly artifact round trip failed"
        print(f"saved {folder} · round trip identical")

    assert not [q for q in model_data.READ_LOG if "features_test" in q], "test period was read"
    print(
        f"wrote reports/anomaly_validation.csv/.md · test set not read · "
        f"{time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
