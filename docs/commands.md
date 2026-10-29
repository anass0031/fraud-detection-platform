# Commands (all verified)

Run from the repository root (Windows: Git Bash or PowerShell; on Linux/macOS use `ml/.venv/bin/...` instead of `ml/.venv/Scripts/...`). The Docker stack must be running for `docker compose exec` commands.

Run from the repository root.

| Task | Command | Expected result |
|---|---|---|
| Verify raw data | `ml/.venv/Scripts/python ml/scripts/verify_raw_data.py` | 5 files OK with official row counts; exit code 0 |
| CSV → Parquet | `ml/.venv/Scripts/python ml/scripts/make_parquet.py` | `data/processed/train.parquet` (590,540 × 435) and `test.parquet` (506,691 × 434) |
| Build feature datasets | `ml/.venv/Scripts/python ml/scripts/build_features.py` | `data/processed/features_{train,validation,test}.parquet` (353,716 / 97,344 / 99,985 rows, 59 features) and `artifacts/feature_encoder.json` |
| Batch-vs-online check | `ml/.venv/Scripts/python ml/scripts/check_skew.py` | `mismatches: 0`, future invariance identical, `RESULT: PASS` (about 2–3 min) |
| Baseline comparison | `ml/.venv/Scripts/python ml/scripts/train_baselines.py` | 5 models trained on train, evaluated on validation; `reports/baselines_comparison.{csv,md}`, `baselines_bootstrap.csv`, `figures/baselines_pr_curves.png`, `figures/baselines_cost_curve.png`; prints `test set not read` (about 2 min) |
| Tune LightGBM | `ml/.venv/Scripts/python ml/scripts/tune_lgbm.py` | 30 Optuna trials (resumable SQLite study in `artifacts/optuna/`), `reports/tuning_best_params.json` (about 25 min) |
| Final model | `ml/.venv/Scripts/python ml/scripts/train_final.py --save` | validation comparison + calibration + alert policy → `reports/model_validation.{csv,md}`, figures, artifact `artifacts/models/lgbm-<date>-<n>/` (round trip checked; about 2 min) |
| Test evaluation (once) | `ml/.venv/Scripts/python ml/scripts/evaluate_test.py` | `reports/test_results.{json,md}` + figures; refuses a second run (`--dry-run` checks the code on validation) |
| Anomaly safety net | `ml/.venv/Scripts/python ml/scripts/train_anomaly.py --save` | Isolation Forest fitted on train (no labels), cut-off = top 0.5% of train; `reports/anomaly_validation.{csv,md}`, figures, artifact `artifacts/models/iforest-<date>-<n>/` (about 15 s) |
| Safety-net test check (once) | `ml/.venv/Scripts/python ml/scripts/evaluate_anomaly_test.py` | `reports/anomaly_test_results.{json,md}`; refuses a second run (`--dry-run` checks the code on validation) |
| SHAP explanations | `ml/.venv/Scripts/python ml/scripts/explain_model.py` | additivity and `shap` cross-check, `reports/figures/shap_{importance,beeswarm}.png`, `reports/shap_examples.md` (about 30 s) |
| Drift reference | `ml/.venv/Scripts/python ml/scripts/build_drift_reference.py --model lgbm-2026.10.27-1` | `artifacts/monitoring/drift_reference-<model>.json` (59 features from train, score from validation; self-check PSI 0; about 6 s) |
| Retrain (challenger) | `docker compose exec -T backend python -m app.scripts.export_training_data --from 2018-04-28 --to 2018-05-26 > data/processed/replayed_labelled.parquet`, then `ml/.venv/Scripts/python ml/scripts/retrain.py`, then `docker compose exec backend python -m app.scripts.register_model <version> --anomaly iforest-2026.10.27-1` | 82,233 exported rows; candidate artifact + drift reference + `reports/retrain_comparison.md` (about 20 s); registered as `candidate` |
| ML tests | `ml/.venv/Scripts/python -m pytest ml/tests -q` (coverage: add `--cov=fraudml --cov-config=ml/pyproject.toml`) | 182 tests pass, 96% coverage (about 100 s) |
| ML lint | `ml/.venv/Scripts/ruff check ml` | no issues |
| Run the EDA notebook | `cd ml/notebooks && ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 01_eda.ipynb` | runs end to end; figures written to `reports/figures/eda/` |
| Load transactions | `docker compose exec backend python -m app.scripts.load_transactions --until 2018-04-28 --source history` | `inserted 490,555 new rows`; a re-run inserts 0 (about 30 s) |
| Register the model | `docker compose exec backend python -m app.scripts.register_model lgbm-2026.10.27-1 --anomaly iforest-2026.10.27-1 --activate` | one `active` row in `model_versions`; after a restart `/api/v1/health` shows both versions |
| Seed demo users | `docker compose exec backend python -m app.scripts.seed_users` | 3 users created (admin + 2 analysts); a re-run creates 0; refuses placeholder passwords |
| Log in | `curl -c cookies.txt -X POST localhost:8000/api/v1/auth/login -d "username=<e-mail>&password=<password>"` | 200 with `access_token`; refresh token stored as an httpOnly cookie |
| Score one transaction | `curl -X POST localhost:8000/api/v1/score -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" -d @txn.json` | 200 with calibrated score, expected loss, alert, risk level, unusual flag and top 10 factors |
| Batch scoring | `docker compose exec backend python -m app.scripts.score_batch --from 2018-04-28 --to 2018-04-29` | 3,060 rows, 154 alerts; a re-run scores 0 (about 7 min); opens one review case per alert / unusual flag |
| Open review cases (backfill) | `docker compose exec backend python -m app.scripts.open_cases --from 2018-04-28 --to 2018-04-29` | for predictions stored before the review queue existed: 156 cases (154 model alerts + 2 unusual); a re-run opens 0 |
| Review queue | `curl -H "Authorization: Bearer <access_token>" "localhost:8000/api/v1/cases?status=new&limit=10"` | the 10 highest-priority open cases and the total |
| Export cases | `curl -H "Authorization: Bearer <access_token>" -o cases.csv "localhost:8000/api/v1/reports/cases.csv?status=closed"` | CSV with one row per case and its final decision (no ground truth) |
| Replay test days | `docker compose exec backend python -m app.scripts.replay --from 2018-05-26 --to 2018-05-27 --workers 4` | per day, in order: load → score (opens cases) → reveal labels → monitoring; re-running a day changes nothing; about 5–6 min per day |
| Recompute monitoring | `docker compose exec backend python -m app.scripts.compute_monitoring --from 2018-04-28 --to 2018-05-26` | daily metrics + drift for every day (about 20 s for 28 days) |
| Fairness report | `docker compose exec backend python -m app.scripts.fairness_report --from 2018-04-28 --to 2018-05-26` | alert rate, precision, recall and false-alarm rate by product, card type and card network on replayed days (Markdown; used in the model card) |
| Pre-demo check | `docker compose exec backend python -m app.scripts.demo_check` | one line per check (database, active model + artifacts, drift reference, candidate, the 3 demo users can log in, open cases, monitoring up to the day before the demo day, demo day not replayed, API health, frontend + proxy), `RESULT: PASS`, exit code 0 |
| Demo DB snapshot / restore | see [demo_script.md](demo_script.md#demo-database-snapshot) | `pg_dump -Fc` in `backups/` (ignored, 76 MB, about 15 s); restore about 1 min, row counts identical |
| Skew check vs PostgreSQL | `docker compose exec backend python -m app.scripts.check_skew_db` | 1,000 rows: 0 feature mismatches, probability difference 0, 0 alert mismatches, `RESULT: PASS` (about 3–5 min; uses the model's own threshold, whatever the admin settings) |
| Backend tests | `docker compose exec backend pytest -q` (coverage: add `--cov`) | 215 tests pass (separate `fraud_test` database); services 97.6% covered |
| Backend lint | `docker compose exec backend ruff check .` | no issues |
| Frontend checks | `docker compose exec frontend npm run lint` / `npm run format:check` / `npm run build` | no errors (the build also type-checks against the generated API types) |
| Frontend tests | `docker compose exec frontend npm test` (coverage: `npm run coverage`) | 145 Vitest tests pass, 96% of lines covered |
| Regenerate API types | `docker compose exec frontend npm run gen:api` | rewrites `frontend/src/api/schema.d.ts` from the running backend; no diff when the API is unchanged |

To open the notebook interactively: `ml/.venv/Scripts/jupyter lab`. You can also register the environment as a kernel with `ml/.venv/Scripts/python -m ipykernel install --user --name fraudml`.
