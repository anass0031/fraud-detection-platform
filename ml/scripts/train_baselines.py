"""L04: train every candidate on the train period, evaluate on the validation period.

Writes reports/l04_comparison.csv (+ .md). The test period is never read (checked at the end).

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/train_baselines.py [--only rule,logreg,...]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.evaluate import CostConfig, cost_curve, evaluate_report, optimise_threshold, pr_auc
from fraudml.models import data as model_data
from fraudml.models.baselines import AmountRule, make_logistic
from fraudml.models.imbalance import correct_probability, undersample
from fraudml.models.lgbm import make_lgbm

SEED = 42


@dataclass
class Result:
    score: np.ndarray  # validation scores
    is_probability: bool
    threshold: float | None = None  # fixed alert threshold (None = cost-optimal on validation)
    note: str = ""


def run_rule(train: model_data.ModelData, val: model_data.ModelData) -> Result:
    rule = AmountRule().fit(train.meta["amount"])
    return Result(
        rule.score(val.meta["amount"]),
        is_probability=False,
        threshold=AmountRule.RULE_THRESHOLD,
        note=f"amount > {rule.cutoff_:.2f} (train p99)",
    )


def run_logreg(train: model_data.ModelData, val: model_data.ModelData) -> Result:
    model = make_logistic(train.X, SEED).fit(train.X, train.y)
    return Result(model.predict_proba(val.X)[:, 1], True, note="class_weight=balanced")


def run_lgbm_none(train: model_data.ModelData, val: model_data.ModelData) -> Result:
    model = make_lgbm(seed=SEED).fit(train.X, train.y)
    return Result(model.predict_proba(val.X)[:, 1], True, note="no imbalance handling")


def run_lgbm_weights(train: model_data.ModelData, val: model_data.ModelData) -> Result:
    spw = float((train.y == 0).sum() / train.y.sum())
    model = make_lgbm(scale_pos_weight=spw, seed=SEED).fit(train.X, train.y)
    return Result(model.predict_proba(val.X)[:, 1], True, note=f"scale_pos_weight={spw:.1f}")


def run_lgbm_under(train: model_data.ModelData, val: model_data.ModelData) -> Result:
    Xs, ys, beta = undersample(train.X, train.y, legit_per_fraud=5, seed=SEED)
    model = make_lgbm(seed=SEED).fit(Xs, ys)
    p = correct_probability(model.predict_proba(val.X)[:, 1], beta)
    note = f"1 fraud : 5 legit, beta={beta:.4f}, {len(ys):,} rows, mean p={p.mean():.4f}"
    return Result(p, True, note=note)


CANDIDATES: dict[str, Callable[[model_data.ModelData, model_data.ModelData], Result]] = {
    "rule_amount_p99": run_rule,
    "logreg_balanced": run_logreg,
    "lgbm_none": run_lgbm_none,
    "lgbm_class_weights": run_lgbm_weights,
    "lgbm_undersample": run_lgbm_under,
}


def bootstrap_pr_auc(
    val: model_data.ModelData, scores: dict[str, np.ndarray], best: str, n_boot: int = 200
) -> pd.DataFrame:
    """Day-block bootstrap: resample whole days with replacement (days are not independent rows).

    Gives a 95% interval for each model's PR-AUC and for its difference to the best model.
    If the difference interval contains 0, the gap may be noise.
    """
    y = val.y.to_numpy()
    day = val.meta["day"].to_numpy()
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in scores}
    for _ in range(n_boot):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, size=len(days))])
        for name, s in scores.items():
            samples[name].append(pr_auc(y[idx], s[idx]))
    out = []
    ref = np.array(samples[best])
    for name, vals in samples.items():
        v = np.array(vals)
        diff = v - ref
        out.append(
            {
                "model": name,
                "pr_auc_low": np.quantile(v, 0.025),
                "pr_auc_high": np.quantile(v, 0.975),
                "diff_vs_best": diff.mean(),
                "diff_low": np.quantile(diff, 0.025),
                "diff_high": np.quantile(diff, 0.975),
                "share_better_than_best": float((diff > 0).mean()),
            }
        )
    return pd.DataFrame(out)


def undersample_seed_check(
    train: model_data.ModelData, val: model_data.ModelData, seeds=(43, 44)
) -> list[float]:
    """PR-AUC of the undersampled model for other random draws (its only source of randomness)."""
    out = []
    for seed in seeds:
        Xs, ys, beta = undersample(train.X, train.y, legit_per_fraud=5, seed=seed)
        p = make_lgbm(seed=seed).fit(Xs, ys).predict_proba(val.X)[:, 1]
        out.append(pr_auc(val.y, p))
    return out


def plot_pr_curves(val: model_data.ModelData, scores: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(val.y, s)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {pr_auc(val.y, s):.3f})", lw=1.5)
    ax.axhline(val.y.mean(), color="grey", ls=":", lw=1, label=f"random ({val.y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Validation period: precision-recall curves (L04)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "l04_pr_curves.png", dpi=130)
    plt.close(fig)


def plot_cost_curve(
    val: model_data.ModelData, score: np.ndarray, name: str, cfg: CostConfig
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amount, day = val.meta["amount"], val.meta["day"]
    curve = cost_curve(val.y, score, amount, day, cfg)
    curve = curve[curve.alerts_per_day <= 600]
    best = optimise_threshold(val.y, score, amount, day, cfg)
    no_model = float(amount[val.y.to_numpy().astype(bool)].sum())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.alerts_per_day, curve.total_cost / 1e3, lw=1.8, label="total cost")
    ax.axhline(no_model / 1e3, color="grey", ls=":", label=f"no model (${no_model / 1e3:,.0f}k)")
    ax.axvline(cfg.capacity_per_day, color="tab:red", ls="--", lw=1, label="capacity 150/day")
    ax.scatter([best.alerts_per_day], [best.total_cost / 1e3], color="tab:red", zorder=3)
    ax.annotate(
        f"chosen: t={best.threshold:.3f}\n{best.alerts_per_day:.0f} alerts/day, "
        f"${best.total_cost / 1e3:,.0f}k",
        (best.alerts_per_day, best.total_cost / 1e3),
        xytext=(15, 25),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(xlabel="Mean alerts per day", ylabel="Total cost on validation ($k)")
    ax.set_title(f"Cost vs alert volume: {name} (validation, 33 days)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "l04_cost_curve.png", dpi=130)
    plt.close(fig)


def write_markdown(
    table: pd.DataFrame,
    boot: pd.DataFrame,
    seeds: list[float] | None,
    best: str,
    cfg: CostConfig,
) -> None:
    cols = [
        "model", "population", "pr_auc", "roc_auc", "recall_at_p50", "precision_at_capacity",
        "fraud_value_caught_at_capacity", "alerts_per_day", "max_alerts_per_day",
        "fraud_value_caught_at_threshold", "net_savings", "brier",
    ]  # fmt: skip
    parts = [
        "# L04 comparison (validation period, generated by `ml/scripts/train_baselines.py`)",
        "",
        f"Costs: review ${cfg.review_cost:g}/alert, friction ${cfg.friction_cost:g}/false alarm, "
        f"missed fraud = amount, capacity {cfg.capacity_per_day}/day. Threshold = cost-optimal "
        "on validation under the capacity (rule: fixed at its own cut-off). "
        "`excl_fraud_uids` reuses the overall threshold.",
        "",
        table[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Day-block bootstrap of PR-AUC (200 resamples, reference = {best})",
        "",
        boot.to_markdown(index=False, floatfmt=".4f"),
    ]
    if seeds is not None:
        parts += ["", f"Undersampling, other random draws (seeds 43, 44): PR-AUC {seeds}"]
    (config.REPORTS_DIR / "l04_comparison.md").write_text("\n".join(parts) + "\n", "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated candidate names")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else list(CANDIDATES)

    cfg = CostConfig()
    t0 = time.perf_counter()
    train = model_data.load_period("train")
    val = model_data.load_period("validation")
    print(
        f"train {len(train.y):,} rows ({train.y.mean():.2%} fraud) · "
        f"validation {len(val.y):,} rows ({val.y.mean():.2%} fraud) · "
        f"{val.meta['day'].nunique()} days · load {time.perf_counter() - t0:.1f}s"
    )

    rows = []
    scores: dict[str, np.ndarray] = {}
    for name in names:
        t = time.perf_counter()
        res = CANDIDATES[name](train, val)
        secs = time.perf_counter() - t
        scores[name] = res.score
        rep = evaluate_report(
            val.y, res.score, val.meta["amount"], val.meta["day"],
            val.meta[config.EVAL_FLAG_COL], cfg, res.threshold, res.is_probability,
        )  # fmt: skip
        rep.insert(0, "model", name)
        rep["seconds"] = round(secs, 1)
        rep["note"] = res.note
        rows.append(rep.reset_index())
        o = rep.loc["overall"]
        print(
            f"{name:<22} PR-AUC {o.pr_auc:.4f} · prec@cap {o.precision_at_capacity:.3f} · "
            f"$caught@thr {o.fraud_value_caught_at_threshold:.1%} · "
            f"savings ${o.net_savings:,.0f} · {secs:.0f}s"
        )

    table = pd.concat(rows, ignore_index=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "l04_comparison.csv", index=False)

    best = table[table.population == "overall"].sort_values("pr_auc").iloc[-1]["model"]
    boot = bootstrap_pr_auc(val, scores, best)
    boot.to_csv(config.REPORTS_DIR / "l04_bootstrap.csv", index=False)
    seeds = undersample_seed_check(train, val) if "lgbm_undersample" in scores else None
    plot_pr_curves(val, scores)
    plot_cost_curve(val, scores[best], best, cfg)
    write_markdown(table, boot, seeds, best, cfg)

    leaked = [p for p in model_data.READ_LOG if "features_test" in p]
    assert not leaked, f"test period was read: {leaked}"
    total = time.perf_counter() - t0
    print(f"wrote reports/l04_comparison.csv/.md · test set not read · {total:.0f}s")


if __name__ == "__main__":
    main()
