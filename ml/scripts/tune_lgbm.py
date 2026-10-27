"""L05 S1: tune LightGBM with Optuna on forward-chaining folds inside the train period.

The study is stored in SQLite, so an interrupted run resumes where it stopped.
Writes reports/l05_best_params.json and reports/l05_tuning_trials.csv.

Usage (repo root):
    ml/.venv/Scripts/python ml/scripts/tune_lgbm.py [--trials 30]
"""

from __future__ import annotations

import argparse
import json
import time

import optuna

from fraudml import config
from fraudml.models import data as model_data
from fraudml.models.cv import forward_folds
from fraudml.models.tune import DEFAULT_PARAMS, final_n_estimators, make_objective

SEED = 42
STUDY = "lgbm-curated-l05"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=30, help="finished trials wanted")
    args = parser.parse_args()

    train = model_data.load_period("train")  # validation and test are not used for tuning
    folds = forward_folds(train.meta["day"].to_numpy(), n_folds=3, val_days=15, gap_days=7)
    for i, f in enumerate(folds):
        print(f"fold {i}: train days {f.train_days} ({len(f.train_idx):,} rows) · "
              f"val days {f.val_days} ({len(f.val_idx):,} rows)")  # fmt: skip

    store = config.ARTIFACTS_DIR / "optuna"
    store.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY,
        storage=f"sqlite:///{(store / 'lgbm_l05.db').as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True,
    )
    for t in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)  # left over from an interrupt
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULT_PARAMS)  # trial 0 = library defaults

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = args.trials - len(study.get_trials(states=finished))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def log(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        if tr.state == optuna.trial.TrialState.PRUNED:
            step = tr.last_step
            val = f"{tr.intermediate_values[step]:.4f} (mean of folds 0..{step} only)"
        else:
            val = f"{tr.value:.4f}" if tr.value is not None else "-"
        secs = tr.user_attrs.get("seconds", "-")
        print(f"trial {tr.number:>2} {tr.state.name:<8} cv PR-AUC {val} · "
              f"best {st.best_value:.4f} · {secs}s", flush=True)  # fmt: skip

    t0 = time.perf_counter()
    if remaining > 0:
        study.optimize(make_objective(train.X, train.y, folds, SEED), n_trials=remaining,
                       callbacks=[log])  # fmt: skip

    best = study.best_trial
    default = study.trials[0]
    out = {
        "study": STUDY,
        "n_trials": len(study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "folds": [{"train_days": f.train_days, "val_days": f.val_days} for f in folds],
        "best_params": best.params,
        "best_cv_pr_auc": best.value,
        "best_fold_scores": best.user_attrs["fold_scores"],
        "best_iterations": best.user_attrs["best_iterations"],
        "final_n_estimators": final_n_estimators(best.user_attrs["best_iterations"]),
        "default_cv_pr_auc": default.value,
        "default_fold_scores": default.user_attrs.get("fold_scores"),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "l05_best_params.json").write_text(json.dumps(out, indent=2), "utf-8")
    study.trials_dataframe().to_csv(config.REPORTS_DIR / "l05_tuning_trials.csv", index=False)
    print(
        f"best trial {best.number}: cv PR-AUC {best.value:.4f} (defaults {default.value:.4f}) · "
        f"final trees {out['final_n_estimators']} · {time.perf_counter() - t0:.0f}s this run"
    )
    assert not [p for p in model_data.READ_LOG if "features_test" in p], "test period was read"


if __name__ == "__main__":
    main()
