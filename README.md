# Fraud Detection and Explainable AI Platform

An end-of-studies (PFA) project to build a platform that **scores payment transactions for fraud risk**, **explains each score** in plain language, and **ranks suspicious cases** for human review. It is built on the IEEE-CIS Fraud Detection dataset (real e-commerce transactions from Vesta Corporation).

> **Status: in progress.** Data ingestion and exploratory analysis, the time-safe feature pipeline, the evaluation framework, the production model (calibrated LightGBM with a cost-based alert rule, evaluated once on the test period), the anomaly safety net, plain-language SHAP explanations the **scoring API** (`POST /api/v1/score` and a batch scoring script, with the model registry in PostgreSQL) and **authentication with an append-only audit trail** are complete. The review queue and the analyst dashboard come next. See [Project status](#project-status).

---

## Contents
- [The problem](#the-problem)
- [Dataset](#dataset)
- [Architecture and repository structure](#architecture-and-repository-structure)
- [Setup](#setup)
- [Usage: verified commands](#usage-verified-commands)
- [Completed work: data ingestion and EDA](#completed-work-data-ingestion-and-eda)
- [Completed work: feature pipeline](#completed-work-feature-pipeline)
- [Completed work: evaluation and baselines](#completed-work-evaluation-and-baselines)
- [Completed work: production model and test results](#completed-work-production-model-and-test-results)
- [Completed work: anomaly safety net](#completed-work-anomaly-safety-net)
- [Completed work: SHAP explanations](#completed-work-shap-explanations)
- [Completed work: scoring API](#completed-work-scoring-api)
- [Completed work: authentication and audit trail](#completed-work-authentication-and-audit-trail)
- [Key findings](#key-findings)
- [Chronological split](#chronological-split)
- [Leakage safeguards](#leakage-safeguards)
- [Project status](#project-status)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)

---

## The problem

Fraud is rare (about 3.5% of transactions in this dataset, often much less in production), but every missed fraud costs money and every false alarm costs analyst time and customer goodwill. Accuracy is therefore a misleading metric: a model that never flags anything is 96.5% "accurate". The platform is designed around:

- **Imbalance-aware evaluation:** PR-AUC, recall at a fixed precision, and precision within a daily review capacity.
- **Business cost:** missed fraud amount vs. review cost, used to choose the alert threshold.
- **Explainability:** SHAP-based, analyst-readable risk factors for each scored transaction.
- **Time-aware validation and leakage control:** models are always evaluated on a later period than they were trained on.

## Dataset

| | |
|---|---|
| Source | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle competition, data provided by Vesta Corporation) |
| Files used | `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`, `sample_submission.csv` |
| Labelled data | 590,540 transactions, 394 transaction columns + 41 identity columns (joined on `TransactionID`), target `isFraud` |
| Unlabelled data | 506,691 transactions (Kaggle test period), used only for drift analysis |
| Expected location | `data/raw/` (raw CSVs) → `data/processed/` (generated Parquet) |

**The dataset is not included in this repository.** It is distributed by Kaggle under the competition's rules, which you must accept before downloading. The two transaction files are also far larger than GitHub's per-file limit (652 MiB and 585 MiB). To reproduce the results:

1. Sign in to Kaggle, open the competition page and accept the rules.
2. Download the data from the **Data** tab and place the five CSV files in `data/raw/`.
3. Run the verification script (below). It checks that all files are present with the exact official row counts.

## Architecture and repository structure

The target system is a web platform: a **FastAPI** backend with **PostgreSQL**, a **React + TypeScript** dashboard, and an ML package shared between offline training and online scoring, all orchestrated with **Docker Compose**. The parts implemented so far are marked below.

```
.
├── docker-compose.yml        # PostgreSQL 16, FastAPI backend, React frontend, Adminer   [implemented]
├── .env.example              # environment template (copy to .env)                        [implemented]
├── backend/                  # FastAPI app (installs the fraudml package; no copy of feature code)
│   ├── alembic/              # database migrations (applied on container start)            [implemented]
│   ├── app/
│   │   ├── api/v1/           # health, auth (login/refresh/logout/me), users, POST /score  [implemented]
│   │   ├── db/models.py      # transactions, model_versions, predictions                  [implemented]
│   │   ├── schemas/          # request/response models (raw IEEE-CIS field names)         [implemented]
│   │   ├── core/security.py  # bcrypt passwords, JWT access tokens, refresh-token hashing [implemented]
│   │   ├── services/         # history, model registry, scoring, loader, auth, users, audit [implemented]
│   │   └── scripts/          # load_transactions, register_model, score_batch, check_skew_db [implemented]
│   └── tests/                # API, registry, history, loader, migration tests (separate test DB)
├── frontend/                 # React 19 + TypeScript + Vite + Mantine: system-status page [skeleton]
│   └── src/
├── ml/                       # "fraudml" Python package (Python 3.11)
│   ├── fraudml/
│   │   ├── config.py         # paths, dataset constants, split dates                      [implemented]
│   │   ├── data/ingest.py    # raw-file verification, merge, downcast, Parquet export     [implemented]
│   │   ├── data/split.py     # chronological train / gap / validation / gap / test split  [implemented]
│   │   ├── features/         # time-safe feature pipeline (batch + online), encoder        [implemented]
│   │   ├── evaluate/         # metrics, cost model, capacity-constrained threshold        [implemented]
│   │   ├── models/           # loader, baselines, LightGBM/XGBoost, CV folds, tuning, calibration, anomaly [implemented]
│   │   ├── explain/          # SHAP contributions, masked-feature groups, plain-language factors [implemented]
│   │   └── registry.py       # file-based model artifacts (model.txt + calibrator/metadata JSON) [implemented]
│   ├── scripts/
│   │   ├── verify_raw_data.py
│   │   ├── make_parquet.py
│   │   ├── build_features.py # feature datasets + fitted encoder
│   │   ├── check_skew.py     # batch-vs-online consistency check on real data
│   │   ├── train_baselines.py # baseline model comparison on the validation period
│   │   ├── tune_lgbm.py      # Optuna tuning (time-ordered folds inside train)
│   │   ├── train_final.py    # comparison, calibration, alert policy, model artifact
│   │   ├── evaluate_test.py  # single evaluation on the test period
│   │   ├── train_anomaly.py  # Isolation Forest safety net: fit, cut-off, validation analysis
│   │   ├── evaluate_anomaly_test.py # single test check of the safety net
│   │   └── explain_model.py  # SHAP checks, global figures, example explanations
│   ├── notebooks/01_eda.ipynb  # exploratory data analysis                                 [implemented]
│   └── tests/                # unit, leakage and training/serving consistency tests
├── docs/eda_findings.md      # one-page summary of the EDA results
├── docs/features.md          # feature dictionary (meaning of every model feature)
├── reports/                  # model comparison tables, findings and figures (figures/eda: EDA notebook)
├── data/                     # raw + processed data (not versioned)
└── artifacts/                # fitted encoder, future trained models (not versioned)
```

## Setup

### Prerequisites
- **Python 3.11** (the ML package and the backend image both target 3.11)
- **Docker Desktop** (with WSL2 on Windows) for the application stack
- The IEEE-CIS CSV files in `data/raw/` (see [Dataset](#dataset))

### ML environment (host)
Commands below are for Windows (Git Bash or PowerShell), where they were verified. On Linux/macOS use `python3.11 -m venv` and `ml/.venv/bin/...` instead of `ml/.venv/Scripts/...`.

```bash
py -3.11 -m venv ml/.venv
ml/.venv/Scripts/python -m pip install -e "ml[eda,train,dev]"
```

Main dependencies (from `ml/pyproject.toml`): pandas, pyarrow, numpy, scipy; extras `eda` (matplotlib, seaborn, JupyterLab, nbconvert) and `dev` (pytest, ruff).

### Application stack (Docker)
```bash
cp .env.example .env            # then set POSTGRES_PASSWORD, JWT_SECRET and the SEED_* passwords
docker compose up -d --build
```

| Service | URL |
|---|---|
| Frontend (system status page) | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/v1/health |
| Adminer (database browser) | http://localhost:8080 |
| PostgreSQL from the host | `localhost:5433` |

Database migrations run automatically when the backend starts. To serve the model (after the ML steps produced `data/processed/` and `artifacts/`), run once inside the backend container:

```bash
# 1. Customer history: every labelled transaction before the test period (490,555 rows, ~30 s)
docker compose exec backend python -m app.scripts.load_transactions --until 2018-04-28 --source history
# 2. Register and activate the model bundle, then restart the API so it loads it
docker compose exec backend python -m app.scripts.register_model lgbm-2026.10.27-1 --anomaly iforest-2026.10.27-1 --activate
docker compose restart backend
# 3. Optional: load and score the first test day (3,060 rows, ~7 min)
docker compose exec backend python -m app.scripts.load_transactions --from 2018-04-28 --to 2018-04-29 --source batch
docker compose exec backend python -m app.scripts.score_batch --from 2018-04-28 --to 2018-04-29
# 4. Demo users: 1 admin + 2 analysts, passwords from .env (re-running skips existing users)
docker compose exec backend python -m app.scripts.seed_users
```

Every endpoint except the health check and login needs a logged-in user. In Swagger (http://localhost:8000/docs), click **Authorize** and log in with an e-mail and password.

## Usage: verified commands

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
| ML tests | `ml/.venv/Scripts/python -m pytest ml/tests -q` | 152 tests pass (about 45 s) |
| ML lint | `ml/.venv/Scripts/ruff check ml` | no issues |
| Run the EDA notebook | `cd ml/notebooks && ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 01_eda.ipynb` | runs end to end; figures written to `reports/figures/eda/` |
| Load transactions | `docker compose exec backend python -m app.scripts.load_transactions --until 2018-04-28 --source history` | `inserted 490,555 new rows`; a re-run inserts 0 (about 30 s) |
| Register the model | `docker compose exec backend python -m app.scripts.register_model lgbm-2026.10.27-1 --anomaly iforest-2026.10.27-1 --activate` | one `active` row in `model_versions`; after a restart `/api/v1/health` shows both versions |
| Seed demo users | `docker compose exec backend python -m app.scripts.seed_users` | 3 users created (admin + 2 analysts); a re-run creates 0; refuses placeholder passwords |
| Log in | `curl -c cookies.txt -X POST localhost:8000/api/v1/auth/login -d "username=<e-mail>&password=<password>"` | 200 with `access_token`; refresh token stored as an httpOnly cookie |
| Score one transaction | `curl -X POST localhost:8000/api/v1/score -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" -d @txn.json` | 200 with calibrated score, expected loss, alert, risk level, unusual flag and top 10 factors |
| Batch scoring | `docker compose exec backend python -m app.scripts.score_batch --from 2018-04-28 --to 2018-04-29` | 3,060 rows, 154 alerts; a re-run scores 0 (about 7 min) |
| Skew check vs PostgreSQL | `docker compose exec backend python -m app.scripts.check_skew_db` | 1,000 rows: 0 feature mismatches, probability difference 0, `RESULT: PASS` (about 3 min) |
| Backend tests | `docker compose exec backend pytest -q` | 86 tests pass (separate `fraud_test` database) |
| Backend lint | `docker compose exec backend ruff check .` | no issues |
| Frontend checks | `docker compose exec frontend npm run lint` / `npm run build` | no errors |

To open the notebook interactively: `ml/.venv/Scripts/jupyter lab`. You can also register the environment as a kernel with `ml/.venv/Scripts/python -m ipykernel install --user --name fraudml`.

## Completed work: data ingestion and EDA

**Data ingestion** (`ml/fraudml/data/ingest.py`):
- verifies the five raw files and their exact row counts;
- renames the test identity columns (`id-01` → `id_01`) so train and test share one schema;
- left-joins identity data onto transactions (`TransactionID`, one-to-one check);
- adds `TransactionDatetime` from the `TransactionDT` seconds offset (reference date 2017-12-01, a common convention);
- downcasts types (float64 → float32, smallest integer types, text → category) while keeping `TransactionAmt` in float64 so cents stay exact;
- writes compressed Parquet. Memory drops from ~1.98 GB to ~0.93 GB for the training data, and loading takes about 1 second. Values were checked to be identical to the CSVs, missing-value patterns included.

**Exploratory analysis** (`ml/notebooks/01_eda.ipynb`, summary in [`docs/eda_findings.md`](docs/eda_findings.md)):
size and column groups, target imbalance, missing values, readable features vs. fraud, time coverage, chronological split, drift (PSI), a pseudo customer key, and leakage checks.

## Completed work: feature pipeline

`ml/fraudml/features/` turns raw transactions into **59 curated features** (full list with plain-language meanings in [`docs/features.md`](docs/features.md)):
- **row-level:** hour, weekday, amount (log, cents, round flag), product, card network/type, e-mail domains and a same-domain flag, device type, browser and OS *family*, missing-data flags, masked `C`/`M` columns, and drift-free `day − D` versions of key `D` columns;
- **customer behaviour, past only:** earlier transactions, last-24 h count and amount, usual amount, amount vs. usual amount, how unusual the amount is, time since the previous transaction, first time on a device / e-mail domain;
- **encoding fitted on the train period only:** rare or unseen categories become `other`; card, region, e-mail and device frequencies are learned from train.

The **same code** computes features for a whole dataset (training) and for **one transaction plus its history** (future scoring API). Tests prove:
- **training/serving consistency:** online features equal batch features for every synthetic row, and for 1,992 sampled real validation/test transactions (0 mismatches, max difference 3e-11), including same-second transactions, customers without a key, long histories and unseen categories;
- **no look-ahead:** removing all later transactions never changes a feature (synthetic data and 48,672 real validation rows);
- **no label use:** flipping or removing `isFraud` leaves every feature unchanged;
- **no identifiers:** `TransactionID`, `uid`, timestamps and evaluation flags can never appear in a feature list.

Feature datasets keep `TransactionID`, `TransactionDT`, `period`, `isFraud` and the evaluation-only flag `uid_fraud_in_train` (validation 965 rows, test 663 rows) as separate columns, never as features.

## Completed work: evaluation and baselines

`ml/fraudml/evaluate/` measures models the way the business sees them: PR-AUC, recall at precision ≥ 0.5, precision within each day's top 150 alerts (analyst capacity), share of fraud **value** caught, and a cost model ($5 per review, $2 friction per false alarm, a missed fraud costs its amount). The alert threshold is chosen to minimise total cost while staying within 150 alerts per day on average. Every result is reported overall and excluding customers who already had a fraud in train.

Validation period results (models trained on the train period; test period untouched):

| Model | PR-AUC | Precision in daily top 150 | Net savings vs no model |
|---|---|---|---|
| Rule: amount > train p99 | 0.048 | 0.084 | $73k |
| Logistic regression | 0.450 | 0.369 | $223k |
| **LightGBM (defaults, no resampling)** | **0.565** | **0.453** | **$349k** |
| LightGBM + class weights | 0.542 | 0.428 | $325k |
| LightGBM + undersampling (probabilities corrected) | 0.555 | 0.444 | $352k |

A day-level bootstrap shows LightGBM without resampling is best on ranking, and class weights are clearly worse. Review capacity, not the model, limits savings: cost keeps falling up to about 500 alerts/day. Details: [`reports/baselines_findings.md`](reports/baselines_findings.md).

![PR curves](reports/figures/baselines_pr_curves.png)

## Completed work: production model and test results

The production model is **LightGBM with default settings** on the 59 curated features, with isotonic calibration and a cost-based alert rule. Alert a transaction when **expected loss = fraud probability × amount ≥ $21.82**, capped by the analysts' capacity of 150 alerts/day.

| Test period (evaluated once, 2018-04-28 → 06-01) | Result |
|---|---|
| Fraud value caught | **65.5%** (target ≥ 50%) |
| Net savings vs no model | **$303k of $516k** fraud losses (59%) |
| Alerts per day (mean / max) | 153 / 210 |
| Precision of alerts | 0.25 |
| PR-AUC (overall / excluding repeat-fraud customers) | 0.454 / 0.455 (validation 0.559) |
| PR-AUC of alternatives on test | XGBoost 0.440 · logistic regression 0.372 · amount rule 0.036 |

What the experiments showed:
- **Tuning was rejected.** Optuna's gain (+0.009 PR-AUC, not significant) came only from re-identifying customers who were already caught in train; for everyone else it was significantly worse (−0.015). A higher headline score can hide leakage.
- **Expected-loss alerting beats probability alerting.** It saves 27% more on validation ($434k vs $342k) at the same alert volume.
- **Readable features cost little.** Using all 409 raw columns raises PR-AUC (0.595 vs 0.565) but not savings ($343k vs $349k).
- **Drift is visible.** Test performance is lower than validation, but stable week to week (62–70% of fraud value caught each week). This motivates the monitoring and retraining steps.

Details: [`reports/model_findings.md`](reports/model_findings.md).

![Weekly stability on test](reports/figures/test_weekly.png)

## Completed work: anomaly safety net

An **Isolation Forest** (unsupervised, fitted on the train period without labels) flags unusual transactions. Transactions that the main model did not alert but that are among the 0.5% most unusual of the train period are flagged as an "unusual pattern".

| | Validation | Test (checked once) |
|---|---|---|
| Extra alerts per day | 2.9 | 2.8 |
| Extra frauds caught | 9 (precision 9.4%) | 1 (precision 1.0%) |
| Net savings change | +$117 | −$516 |

**Result:** the safety net adds almost nothing here. Most unusual transactions are already caught by the main model, and on test the remaining extra alerts are less precise than random review. It is kept only as a **low-priority "unusual pattern" flag** (about 3 per day, reviewed when capacity is left). A one-off burst of unusual transactions in December 2017 also makes the train-based cut-off stricter than intended, which argues for refitting it on recent traffic. Details: [`reports/anomaly_findings.md`](reports/anomaly_findings.md).

## Completed work: SHAP explanations

Every transaction gets its **top risk factors in plain language**, computed with LightGBM's built-in exact SHAP values (identical to the `shap` library, about 18 ms per transaction). Anonymised columns are grouped into four families and never named.

> Amount $554: moderately raises risk
> 24.5 h since this customer's previous transaction: moderately raises risk
> Card type is credit: strongly raises risk

The explanation object (top 10 factors with direction, strength and a `masked` flag) is ready to be stored with each prediction. The anonymised families drive the score most, so the case page will list readable reasons first. Factors describe what moved the model's score, not causes. Details and 10 reviewed examples: [`reports/shap_findings.md`](reports/shap_findings.md), [`reports/shap_examples.md`](reports/shap_examples.md).

![SHAP importance](reports/figures/shap_importance.png)

## Completed work: scoring API

`POST /api/v1/score` takes one raw transaction (IEEE-CIS field names; only id, time and amount are required). It returns a calibrated fraud probability, the expected loss (probability × amount), the alert decision (expected loss ≥ $21.82), a risk level (low < 5% · medium < 20% · high < 50% · critical), the "unusual pattern" flag and the top 10 plain-language factors. The transaction and its prediction are stored in PostgreSQL.

- **One feature pipeline:** the backend installs the `fraudml` package and calls its online feature path. Only the history lookup is new (the customer's earlier transactions, read from PostgreSQL).
- **Model registry:** `model_versions` holds each scoring bundle (LightGBM model, calibrator, alert policy, linked Isolation Forest). Registration refuses artifacts trained with another feature encoder or other library versions. At most one bundle is active, and the API loads it at startup.
- **Safe to retry:** the same transaction sent twice returns the stored prediction; different data under the same id is rejected (409). Ground truth can never be sent (`isFraud` → 422) and is never returned.
- **Batch scoring** uses the same service for a date range of stored transactions.

**Training–serving consistency, re-checked on PostgreSQL:** for 1,000 random validation and test transactions, features computed by the API path equal the offline training features (0 mismatches over 59 features, largest difference 2e-12), and probabilities are identical. The first test day scored in batch (3,060 transactions) reproduces offline scoring exactly: same probabilities, same 154 alerts, same 2 unusual flags.

Scoring takes about 140 ms per transaction (features 66 ms, Isolation Forest 30 ms, SHAP 24 ms, LightGBM 10 ms, history query 4 ms).

## Completed work: authentication and audit trail

- **Login** (`POST /auth/login`, OAuth2 password form) returns a 15-minute JWT access token, sent as `Authorization: Bearer …`. A 7-day refresh token is set as an **httpOnly cookie** (SameSite=Strict, limited to `/api/v1/auth`), so page scripts cannot read it. `POST /auth/refresh` swaps it for a new one each time. If an already-used refresh token is presented again, it is treated as stolen and all of that user's sessions are revoked. `POST /auth/logout` revokes it.
- **Stored safely:** passwords are hashed with bcrypt (10 to 72 bytes). Only a SHA-256 hash of each refresh token is stored. A wrong password and an unknown e-mail get the same answer, so the login does not reveal which accounts exist.
- **Two roles:** analyst and admin. `GET/POST /users` is admin only. The user is re-read on every request, so a deactivated account or a changed role takes effect immediately. `POST /score` needs any logged-in user.
- **Demo users** come from a seed script, with passwords taken only from `.env`.
- **Audit trail:** every state-changing action writes one row to `audit_logs` in the same transaction as the change: `auth.login`, `auth.login_failed` (with a reason, never the password), `auth.logout`, `auth.token_reuse`, `user.create`, `model.register` and `model.activate`. **PostgreSQL itself refuses UPDATE, DELETE and TRUNCATE** on this table (trigger), and a user with audit history cannot be deleted. The audit service also refuses any field that looks like a password or token.

## Key findings

- **Fraud rate: about 3.50%** in the labelled data (20,663 of 590,540; 1 fraud for every 27.6 legitimate transactions), about 113 frauds per day. The weekly rate varies between 2.3% and 4.9%.
- **Mostly masked features:** about 415 of 434 columns are anonymised (`V`, `C`, `D`, `M`, `id_`). Analyst-readable explanations will rely on engineered features built from the readable columns.
- **Missing values carry signal:** identity data exists for 24.4% of transactions, and those transactions are 3.8× riskier (7.85% vs 2.09%).
- **Strong readable signals:** product C (11.7% fraud vs 2.0% for W), credit vs debit cards (6.7% vs 2.4%), mobile devices (10.2%), low-volume hours of the day (up to 10.6%), certain e-mail domains. Raw amount alone is a weak signal.
- **Drift:** browser version (`id_31`) and some time-delta columns (`D15`) drift strongly over time; amount and card features are stable.
- **No single-column leak:** the best single numeric column reaches ROC-AUC 0.69.

![Daily volume and fraud rate](reports/figures/eda/02_volume_and_fraud_rate_over_time.png)

## Chronological split

The model will predict *future* transactions, so data is split by time, never randomly. There is a **7-day gap** before validation and before test, so look-back features and late-arriving labels cannot leak across periods.

| Period | Dates (inclusive) | Rows | Share | Fraud rate |
|---|---|---|---|---|
| Train | 2017-12-02 → 2018-03-11 | 353,716 | 59.9% | 3.38% |
| Gap | 2018-03-12 → 2018-03-18 | 20,946 | 3.5% | – |
| Validation | 2018-03-19 → 2018-04-20 | 97,344 | 16.5% | 3.93% |
| Gap | 2018-04-21 → 2018-04-27 | 18,549 | 3.1% | – |
| Test | 2018-04-28 → 2018-06-01 | 99,985 | 16.9% | 3.43% |

Gap rows are not used for training or scoring. They may be used as past history when computing behavioural features.

![Time split](reports/figures/eda/05_time_split.png)

## Leakage safeguards

The EDA confirmed strong **label propagation**: when a pseudo customer (`uid = card1 + addr1 + (day − D1)`) has a fraudulent transaction, their next transaction is fraudulent 87.2% of the time (vs 0.36% otherwise). Therefore:

1. **No features built from past labels** of the same customer or card. In practice, labels (chargebacks) arrive weeks later.
2. **No raw identifiers as features:** neither `uid` nor `TransactionID`. Only aggregates over *past* transactions are allowed.
3. **Test metrics are reported twice:** overall, and excluding customers who already had a fraud in the training period. That small group (0.7% of test rows) accounts for about 10% of test frauds and would otherwise inflate results.
4. Encoders, scalers and any resampling are fitted on training data only; behavioural aggregates use strictly earlier transactions.
5. Rare high-cardinality categories (e.g. specific device models) will be grouped or frequency-encoded to avoid memorisation.

## Project status

| Stage | Status |
|---|---|
| Infrastructure skeleton (Docker Compose, PostgreSQL, FastAPI health endpoint, React status page) | ✅ done |
| **Data ingestion and exploratory analysis** | ✅ done |
| **Feature engineering, chronological split, training/serving consistency test** | ✅ done |
| **Evaluation framework, baselines and imbalance handling** | ✅ done |
| **Model selection (LightGBM/XGBoost), calibration, cost-based threshold, test evaluation** | ✅ done |
| **Anomaly detection safety net (Isolation Forest)** | ✅ done |
| **SHAP explanations (local factors, plain-language reasons, global figures)** | ✅ done |
| **Scoring API: database schema, model registry, `POST /score`, batch scoring** | ✅ done |
| **Authentication and roles, audit trail** | ✅ done |
| **Review queue and case workflow** | ⏭ next |
| Analyst dashboard, monitoring (drift, performance), retraining workflow | planned |

**Next stage:** the review queue. Cases created from alerts, then filters, assignment and decisions (confirm fraud / false positive / escalate), all audited.

## Limitations

- Activating another model requires restarting the backend. The frontend is still a skeleton.
- No login rate limiting or account lock-out yet (failed logins are audited). Behind the Vite proxy, the audit log records the proxy's IP, not the browser's. Users cannot be edited, deactivated or have their password reset through the API yet (database only). The audit log is readable through the database (Adminer), not through an API page.
- Performance drops from validation to test (PR-AUC 0.559 → 0.454) because of drift; probabilities are slightly too high on test (mean 4.2% vs 3.4% fraud rate).
- Scoring takes about 140 ms per transaction (shared pandas feature code, chosen to guarantee consistency with training). That is fine for single calls; replaying the whole test period (~100k transactions) would take a few hours.
- Most dataset features are anonymised, which limits how readable explanations can be.
- The pseudo customer key is an approximation: 11.3% of rows have none, and a few keys group many transactions.
- The time zone is unknown, so "hour of day" is relative to the reference date.
- Commands were verified on Windows 11 with Python 3.11.9 and Docker Desktop.

## Reproducibility

- **Data access:** a Kaggle account and acceptance of the IEEE-CIS competition rules are required. The data is not redistributed here.
- **Integrity check:** `verify_raw_data.py` fails if any file is missing or has an unexpected row count.
- **Deterministic pipeline:** ingestion, the EDA notebook and the feature build involve no randomness. Re-running `make_parquet.py`, the notebook and `build_features.py` reproduces the numbers and figures above. Model experiments use a fixed seed (42), so `train_baselines.py` and `train_final.py` reproduce the baseline and model tables (checked by re-running). The Optuna study is stored in SQLite and resumable.
- **Environment:** Python 3.11; dependencies declared in `ml/pyproject.toml`, `backend/pyproject.toml` and `frontend/package.json` (+ `package-lock.json`). Docker images pin `postgres:16`, `python:3.11-slim` and `node:22-alpine`.
