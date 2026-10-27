"""Model selection: validation comparison, calibration and alert policy of the production model.

1. Comparison (train -> validation, uncalibrated): defaults, Optuna-tuned, XGBoost defaults,
   tuned on full features.
2. Production model = LightGBM defaults (decision 2026-10-27: tuned params rejected, their gain
   came from re-identifying customers already caught in train). Isotonic calibration fitted on
   validation; alert policy A (probability >= t) vs B (expected loss = p x amount >= t), both
   under the 150/day capacity; the cheaper one is kept.
3. `--save`: writes the model artifact (fraudml.registry) used by the explanation and scoring
   steps and by evaluate_test.py.

The test period is never read by this script.

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/train_final.py [--skip-full] [--save]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fraudml import config
from fraudml.evaluate import CostConfig, brier, cost_curve, evaluate_report, pr_auc
from fraudml.features.pipeline import FEATURES_CURATED, FEATURES_FULL
from fraudml.models import data as model_data
from fraudml.models.calibration import IsotonicCalibrator, expected_loss
from fraudml.models.lgbm import LGBM_DEFAULTS, make_lgbm
from fraudml.models.tune import model_params
from fraudml.models.xgb import gpu_available, make_xgb, predict_proba_cpu
from fraudml.registry import load_model, save_model

SEED = 42
POLICY_NAMES = {"A": "probability", "B": "expected_loss"}


def load_best_params() -> dict:
    path = config.REPORTS_DIR / "tuning_best_params.json"
    if not path.exists():
        raise SystemExit("run ml/scripts/tune_lgbm.py first (reports/tuning_best_params.json)")
    return json.loads(path.read_text("utf-8"))


def tuned_lgbm(best: dict) -> LGBMClassifier:
    return LGBMClassifier(**model_params(best["best_params"], best["final_n_estimators"], SEED))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr(y, scores: dict[str, np.ndarray], path, title: str) -> None:
    from sklearn.metrics import precision_recall_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, lw=1.5, label=f"{name} (PR-AUC {pr_auc(y, s):.3f})")
    ax.axhline(np.mean(y), color="grey", ls=":", lw=1, label=f"random ({np.mean(y):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(y, raw, cal, path) -> None:
    """Mean predicted vs observed fraud rate in 20 bins of predicted value (equal counts)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in (("uncalibrated", raw), ("isotonic", cal)):
        df = pd.DataFrame({"p": p, "y": y})
        df["bin"] = pd.qcut(df["p"].rank(method="first"), 20, labels=False)
        g = df.groupby("bin").mean()
        ax.plot(
            g["p"].clip(lower=1e-4),
            g["y"].clip(lower=1e-4),
            marker="o",
            ms=3,
            label=f"{name} (Brier {brier(y, p):.4f})",
        )
    ax.plot([1e-4, 1], [1e-4, 1], color="grey", ls=":", lw=1, label="perfect")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Mean predicted probability",
        ylabel="Observed fraud rate",
        title="Reliability on validation (20 equal-size bins)",
    )
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_policy_costs(curves: dict[str, pd.DataFrame], no_model: float, cfg, path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        c = c[c.alerts_per_day <= 600]
        ax.plot(c.alerts_per_day, c.total_cost / 1e3, lw=1.8, label=name)
    ax.axhline(no_model / 1e3, color="grey", ls=":", label=f"no model (${no_model / 1e3:,.0f}k)")
    ax.axvline(cfg.capacity_per_day, color="tab:red", ls="--", lw=1, label="capacity 150/day")
    ax.set(
        xlabel="Mean alerts per day",
        ylabel="Total cost on validation ($k)",
        title="Alert policies, calibrated LightGBM (validation)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-full", action="store_true", help="skip the full-feature run")
    parser.add_argument("--save", action="store_true", help="save the production model artifact")
    args = parser.parse_args()

    cfg = CostConfig()
    best = load_best_params()
    t0 = time.perf_counter()
    train = model_data.load_period("train")
    val = model_data.load_period("validation")
    m = val.meta
    y, amount, day, flag = val.y, m["amount"], m["day"], m[config.EVAL_FLAG_COL]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. comparison (uncalibrated) ----------------
    rows: list[pd.DataFrame] = []
    scores: dict[str, np.ndarray] = {}

    def add_rows(name: str, score, note: str, secs: float = 0.0, **kw) -> pd.DataFrame:
        rep = evaluate_report(y, score, amount, day, flag, cfg, **kw)
        rep.insert(0, "model", name)
        rep["seconds"] = round(secs, 1)
        rep["note"] = note
        rows.append(rep.reset_index())
        o = rep.loc["overall"]
        print(
            f"{name:<22} PR-AUC {o.pr_auc:.4f} · excl {rep.loc['excl_fraud_uids', 'pr_auc']:.4f}"
            f" · prec@thr {o.precision_at_threshold:.3f} · $caught@thr"
            f" {o.fraud_value_caught_at_threshold:.1%} · savings ${o.net_savings:,.0f}",
            flush=True,
        )
        return rep

    def fit_eval(name: str, fit_predict, note: str) -> np.ndarray:
        t = time.perf_counter()
        score = fit_predict()
        add_rows(name, score, note, time.perf_counter() - t)
        scores[name] = score
        return score

    prod = make_lgbm(seed=SEED)
    fit_eval(
        "lgbm_default",
        lambda: prod.fit(train.X, train.y).predict_proba(val.X)[:, 1],
        "PRODUCTION: defaults, 500 trees",
    )
    fit_eval(
        "lgbm_tuned",
        lambda: tuned_lgbm(best).fit(train.X, train.y).predict_proba(val.X)[:, 1],
        f"Optuna best, {best['final_n_estimators']} trees (rejected)",
    )
    device = "cuda" if gpu_available() else "cpu"
    fit_eval(
        "xgb_default",
        lambda: predict_proba_cpu(make_xgb(SEED, device).fit(train.X, train.y), val.X),
        f"library defaults, device={device}",
    )

    # ---------------- 2. calibration + alert policy (production model) ----------------
    raw = scores["lgbm_default"]
    cal = IsotonicCalibrator().fit(raw, y)
    p = cal.predict(raw)
    top = raw >= np.quantile(raw, 0.95)
    print(
        f"calibration: Brier {brier(y, raw):.4f} -> {brier(y, p):.4f} · PR-AUC {pr_auc(y, raw):.4f}"
        f" -> {pr_auc(y, p):.4f} · {len(cal.x)} breakpoints · {len(np.unique(p[top]))} distinct"
        f" values in the top 5% · mean p {p.mean():.4f} (rate {y.mean():.4f})"
    )
    plot_reliability(y.to_numpy(), raw, p, config.FIGURES_DIR / "model_reliability.png")

    policy_scores = {"A": p, "B": expected_loss(p, amount)}
    policy_reps, curves = {}, {}
    for key, s in policy_scores.items():
        name = f"policy_{key}_{POLICY_NAMES[key]}"
        policy_reps[key] = add_rows(name, s, "calibrated defaults", is_probability=(key == "A"))
        curves[f"{key}: {POLICY_NAMES[key]}"] = cost_curve(y, s, amount, day, cfg)
    no_model = float(amount[y.to_numpy().astype(bool)].sum())
    plot_policy_costs(curves, no_model, cfg, config.FIGURES_DIR / "model_cost_policies.png")
    chosen = min(policy_reps, key=lambda k: policy_reps[k].loc["overall", "total_cost"])
    rep = policy_reps[chosen]
    threshold = float(rep.loc["overall", "threshold"])
    print(f"chosen policy {chosen} ({POLICY_NAMES[chosen]}), threshold {threshold:.6g}")

    # ---------------- 3. full features (memory: drop curated train first) ----------------
    if not args.skip_full:
        del train
        train_f = model_data.load_period("train", FEATURES_FULL)
        val_f = model_data.load_period("validation", FEATURES_FULL)
        fit_eval(
            "lgbm_tuned_full",
            lambda: tuned_lgbm(best).fit(train_f.X, train_f.y).predict_proba(val_f.X)[:, 1],
            f"Optuna params, {len(FEATURES_FULL)} features",
        )
        fit_eval(
            "lgbm_default_full",
            lambda: make_lgbm(seed=SEED).fit(train_f.X, train_f.y).predict_proba(val_f.X)[:, 1],
            f"defaults, {len(FEATURES_FULL)} features",
        )
        del train_f, val_f

    # ---------------- reports ----------------
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(config.REPORTS_DIR / "model_validation.csv", index=False)
    cols = [
        "model",
        "population",
        "pr_auc",
        "recall_at_p50",
        "precision_at_capacity",
        "alerts_per_day",
        "max_alerts_per_day",
        "precision_at_threshold",
        "recall_at_threshold",
        "fraud_value_caught_at_threshold",
        "net_savings",
        "brier",
    ]
    (config.REPORTS_DIR / "model_validation.md").write_text(
        "# Model validation results (generated by train_final.py)\n\n"
        "Rows `lgbm_*` / `xgb_*`: uncalibrated scores, cost-optimal threshold under capacity. "
        "Rows `policy_*`: production model after isotonic calibration (fitted on validation).\n\n"
        + table[cols].to_markdown(index=False, floatfmt=".4f")
        + f"\n\nChosen alert policy: **{chosen} ({POLICY_NAMES[chosen]})**, "
        f"threshold {threshold:.6g}.\n",
        "utf-8",
    )
    plot_pr(
        y,
        scores,
        config.FIGURES_DIR / "model_val_pr_curves.png",
        "Validation period: LightGBM variants vs XGBoost",
    )

    # ---------------- artifact ----------------
    if args.save:
        o, e = rep.loc["overall"], rep.loc["excl_fraud_uids"]
        keep = [
            "pr_auc",
            "precision_at_threshold",
            "recall_at_threshold",
            "alerts_per_day",
            "fraud_value_caught_at_threshold",
            "net_savings",
            "total_cost",
        ]
        folder = save_model(
            prod,
            cal,
            {
                "features": list(FEATURES_CURATED),
                "categorical_features": [
                    c for c in FEATURES_CURATED if isinstance(val.X[c].dtype, pd.CategoricalDtype)
                ],
                "params": {**LGBM_DEFAULTS, "random_state": SEED},
                "policy": POLICY_NAMES[chosen],
                "threshold": threshold,
                "cost_config": cfg.to_dict(),
                "training_data_window": config.SPLIT_DATES["train"],
                "calibration_window": config.SPLIT_DATES["validation"],
                "validation_metrics": {
                    "overall": {k: float(o[k]) for k in keep},
                    "excl_fraud_uids": {k: float(e[k]) for k in keep},
                    "pr_auc_uncalibrated": pr_auc(y, raw),
                    "brier_uncalibrated": brier(y, raw),
                    "brier_calibrated": brier(y, p),
                },
                "notes": "Default params; Optuna-tuned params rejected (gain came from "
                "re-identifying fraud-in-train customers). Test metrics: see evaluate_test.py.",
            },
        )
        # Round trip: the reloaded artifact must reproduce the in-memory scores exactly.
        loaded = load_model(folder.name, folder.parent)
        diff_raw = float(np.max(np.abs(loaded.raw_score(val.X) - raw)))
        same_p = bool(
            np.array_equal(loaded.predict_proba(val.X), cal.predict(loaded.raw_score(val.X)))
        )
        assert diff_raw < 1e-12 and same_p, f"artifact round trip failed ({diff_raw})"
        print(f"saved artifact {folder} · round trip max |diff| {diff_raw:.1e}")

    assert not [q for q in model_data.READ_LOG if "features_test" in q], "test period was read"
    print(
        f"wrote reports/model_validation.csv/.md · test set not read · "
        f"{time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
