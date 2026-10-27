"""SHAP explanations of the production model: checks, global figures, example cases.

1. Checks on real validation rows: additivity (base + contributions = raw log-odds), equality
   with `shap.TreeExplainer`, and the time to explain one transaction.
2. Global figures on a 5,000-row validation sample: mean |SHAP| bar chart and beeswarm
   (masked families grouped).
3. Ten example cases (random, seed 42): 4 caught frauds, 3 false alarms, 2 missed frauds and
   1 ordinary legitimate transaction, with their top-5 factors.

Validation period only; the test period is not read.

Writes reports/shap_examples.md, reports/figures/shap_{importance,beeswarm}.png.

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/explain_model.py [--model lgbm-...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.explain.contributions import BASE, GROUPS, contributions, group_contributions
from fraudml.explain.explain import explain_batch, explain_transaction
from fraudml.explain.templates import TEMPLATES, money
from fraudml.models import data as model_data
from fraudml.registry import MODELS_DIR, load_model

SEED = 42
SAMPLE = 5000


def checks(model, X: pd.DataFrame) -> dict:
    sample = X.sample(1000, random_state=SEED)
    c = contributions(model.booster, sample)
    raw = model.booster.predict(sample[model.features], raw_score=True)
    additivity = float(np.max(np.abs(c.sum(axis=1).to_numpy() - raw)))
    try:
        import warnings

        import shap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = shap.TreeExplainer(model.booster).shap_values(sample[model.features])
        sv = sv[1] if isinstance(sv, list) else sv
        shap_diff = float(np.max(np.abs(c.drop(columns=BASE).to_numpy() - sv)))
    except ImportError:
        shap_diff = float("nan")
    one = X.iloc[[0]]
    explain_transaction(model.booster, one, calibrator=model.calibrator)  # warm-up
    times = []
    for i in range(50):
        t = time.perf_counter()
        explain_transaction(model.booster, X.iloc[[i]], calibrator=model.calibrator)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "additivity_max_error": additivity,
        "shap_max_diff": shap_diff,
        "ms_per_explanation_median": float(np.median(times)),
    }


def label_of(key: str) -> str:
    return TEMPLATES[key][0]


def global_figures(model, X: pd.DataFrame) -> pd.Series:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = X.sample(SAMPLE, random_state=SEED)
    g = group_contributions(contributions(model.booster, sample)).drop(columns=BASE)
    importance = g.abs().mean().sort_values(ascending=False)

    top = importance.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    colors = ["tab:grey" if k in GROUPS else "tab:blue" for k in top.index]
    ax.barh([label_of(k) for k in top.index], top.values, color=colors)
    ax.set(
        xlabel="Mean |SHAP contribution| (log-odds)",
        title="What drives the risk score (validation sample, top 20)",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.text(
        0.98,
        0.02,
        "grey = masked feature family",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="tab:grey",
    )
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)

    # Beeswarm: one dot per transaction, colour = feature value rank (grey for groups/categories).
    keys = list(importance.head(15).index)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    rng = np.random.default_rng(SEED)
    for row, key in enumerate(reversed(keys)):
        vals = g[key].to_numpy()
        jitter = rng.normal(0, 0.12, len(vals))
        if key in GROUPS or isinstance(sample[key].dtype, pd.CategoricalDtype):
            ax.scatter(vals, row + jitter, s=3, color="tab:grey", alpha=0.4, linewidths=0)
        else:
            feat = sample[key].astype("float64")
            col = feat.rank(pct=True).to_numpy()
            nan = np.isnan(feat.to_numpy())
            ax.scatter(
                vals[nan], row + jitter[nan], s=3, color="lightgrey", alpha=0.4, linewidths=0
            )
            sc = ax.scatter(
                vals[~nan],
                row + jitter[~nan],
                s=3,
                c=col[~nan],
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                alpha=0.6,
                linewidths=0,
            )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(keys)), [label_of(k) for k in reversed(keys)])
    ax.set(
        xlabel="SHAP contribution (log-odds; right = raises risk)",
        title="How feature values move the risk score (validation sample)",
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Feature value (low → high rank); grey = category, group or missing")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_beeswarm.png", dpi=130)
    plt.close(fig)
    return importance


def examples(model, val, prob, amount, alert) -> str:
    y = val.y.to_numpy().astype(bool)
    rng = np.random.default_rng(SEED)
    picks = []
    for kind, mask, n in (
        ("caught fraud", y & alert, 4),
        ("false alarm", ~y & alert, 3),
        ("missed fraud", y & ~alert, 2),
        ("ordinary legit", ~y & ~alert, 1),
    ):
        idx = rng.choice(np.flatnonzero(mask), size=n, replace=False)
        picks += [(kind, int(i)) for i in idx]
    rows = val.X.iloc[[i for _, i in picks]]
    items = explain_batch(model.booster, rows, k=5, calibrator=model.calibrator)
    lines = [
        "# SHAP example cases (validation period, generated by explain_model.py)",
        "",
        f"Production model `{model.metadata['version']}`. Alert rule: expected loss "
        f"(probability x amount) >= {money(model.metadata['threshold'])}. Cases drawn at random "
        "(seed 42) within each group. Factors are SHAP contributions to the model's risk score "
        "(log-odds): they show what pushed the score up or down, not causes.",
        "",
    ]
    for n, ((kind, i), item) in enumerate(zip(picks, items, strict=True), 1):
        el = prob[i] * amount[i]
        lines += [
            f"## {n}. {kind.capitalize()}: transaction {int(val.meta['TransactionID'].iloc[i])}",
            "",
            f"Amount {money(amount[i])} · fraud probability {prob[i]:.1%} · expected loss "
            f"{money(el)} · {'ALERT' if alert[i] else 'no alert'} · true label: "
            f"{'fraud' if y[i] else 'legitimate'}",
            "",
            *[f"- {f['text']}" for f in item["factors"]],
            f"- *(all other factors together: {item['other_contribution']:+.2f} log-odds)*",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="model version (default: latest lgbm-*)")
    args = parser.parse_args()
    version = args.model or sorted(p.name for p in MODELS_DIR.glob("lgbm-*") if p.is_dir())[-1]
    model = load_model(version)
    val = model_data.load_period("validation")

    c = checks(model, val.X)
    print(
        f"additivity max error {c['additivity_max_error']:.1e} · shap.TreeExplainer max diff "
        f"{c['shap_max_diff']:.1e} · {c['ms_per_explanation_median']:.1f} ms per explanation"
    )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = global_figures(model, val.X)
    print(
        "top 8 by mean |SHAP|:",
        ", ".join(f"{label_of(k)} {v:.3f}" for k, v in importance.head(8).items()),
    )

    amount = val.meta["amount"].to_numpy()
    prob = model.predict_proba(val.X)
    alert = model.is_alert(prob, amount)
    (config.REPORTS_DIR / "shap_examples.md").write_text(
        examples(model, val, prob, amount, alert) + "\n", "utf-8"
    )
    assert not [q for q in model_data.READ_LOG if "features_test" in q], "test period was read"
    print("wrote reports/shap_examples.md and figures · test set not read")


if __name__ == "__main__":
    main()
