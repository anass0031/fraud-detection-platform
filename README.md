# Fraud Detection and Explainable AI Platform

An end-of-studies (PFA) project to build a platform that **scores payment transactions for fraud risk**, **explains each score** in plain language, and **ranks suspicious cases** for human review. It is built on the IEEE-CIS Fraud Detection dataset (real e-commerce transactions from Vesta Corporation).

> **Status: early stage.** The infrastructure skeleton and the data-ingestion and exploratory-analysis stage (**L02**) are complete. **No model has been trained yet**, and the API currently exposes only a health check. See [Project status](#project-status).

---

## Contents
- [The problem](#the-problem)
- [Dataset](#dataset)
- [Architecture and repository structure](#architecture-and-repository-structure)
- [Setup](#setup)
- [Usage: verified commands](#usage-verified-commands)
- [Completed work: data ingestion and EDA (L02)](#completed-work-data-ingestion-and-eda-l02)
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
├── backend/                  # FastAPI app: settings, DB session, GET /api/v1/health      [skeleton]
│   ├── app/
│   └── tests/
├── frontend/                 # React 19 + TypeScript + Vite + Mantine: system-status page [skeleton]
│   └── src/
├── ml/                       # "fraudml" Python package (Python 3.11)
│   ├── fraudml/
│   │   ├── config.py         # paths and dataset constants                                [implemented]
│   │   └── data/ingest.py    # raw-file verification, merge, downcast, Parquet export     [implemented]
│   ├── scripts/
│   │   ├── verify_raw_data.py
│   │   └── make_parquet.py
│   ├── notebooks/01_eda.ipynb  # exploratory data analysis                                 [implemented]
│   └── tests/test_ingest.py
├── docs/eda_findings.md      # one-page summary of the EDA results
├── reports/figures/eda/      # figures produced by the EDA notebook
├── data/                     # raw + processed data (not versioned)
└── artifacts/                # future trained models (not versioned)
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
ml/.venv/Scripts/python -m pip install -e "ml[eda,dev]"
```

Main dependencies (from `ml/pyproject.toml`): pandas, pyarrow, numpy, scipy; extras `eda` (matplotlib, seaborn, JupyterLab, nbconvert) and `dev` (pytest, ruff).

### Application stack (Docker)
```bash
cp .env.example .env            # then set a local password in .env
docker compose up -d --build
```

| Service | URL |
|---|---|
| Frontend (system status page) | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/v1/health |
| Adminer (database browser) | http://localhost:8080 |
| PostgreSQL from the host | `localhost:5433` |

## Usage: verified commands

Run from the repository root.

| Task | Command | Expected result |
|---|---|---|
| Verify raw data | `ml/.venv/Scripts/python ml/scripts/verify_raw_data.py` | 5 files OK with official row counts; exit code 0 |
| CSV → Parquet | `ml/.venv/Scripts/python ml/scripts/make_parquet.py` | `data/processed/train.parquet` (590,540 × 435) and `test.parquet` (506,691 × 434) |
| ML tests | `ml/.venv/Scripts/python -m pytest ml/tests -q` | 7 tests pass |
| ML lint | `ml/.venv/Scripts/ruff check ml` | no issues |
| Run the EDA notebook | `cd ml/notebooks && ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 01_eda.ipynb` | runs end to end; figures written to `reports/figures/eda/` |
| Backend tests | `docker compose exec backend pytest -q` | 2 tests pass |
| Frontend checks | `docker compose exec frontend npm run lint` / `npm run build` | no errors |

To open the notebook interactively: `ml/.venv/Scripts/jupyter lab`. You can also register the environment as a kernel with `ml/.venv/Scripts/python -m ipykernel install --user --name fraudml`.

## Completed work: data ingestion and EDA (L02)

**Data ingestion** (`ml/fraudml/data/ingest.py`):
- verifies the five raw files and their exact row counts;
- renames the test identity columns (`id-01` → `id_01`) so train and test share one schema;
- left-joins identity data onto transactions (`TransactionID`, one-to-one check);
- adds `TransactionDatetime` from the `TransactionDT` seconds offset (reference date 2017-12-01, a common convention);
- downcasts types (float64 → float32, smallest integer types, text → category) while keeping `TransactionAmt` in float64 so cents stay exact;
- writes compressed Parquet. Memory drops from ~1.98 GB to ~0.93 GB for the training data, and loading takes about 1 second. Values were checked to be identical to the CSVs, missing-value patterns included.

**Exploratory analysis** (`ml/notebooks/01_eda.ipynb`, summary in [`docs/eda_findings.md`](docs/eda_findings.md)):
size and column groups, target imbalance, missing values, readable features vs. fraud, time coverage, chronological split, drift (PSI), a pseudo customer key, and leakage checks.

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
| **L02: data ingestion and exploratory analysis** | ✅ done |
| **L03: feature engineering, chronological split, training/serving consistency test** | ⏭ next |
| Model training and evaluation (baselines, imbalance handling, LightGBM/XGBoost, cost-based threshold) | planned |
| Anomaly detection (Isolation Forest) and SHAP explanations | planned |
| Scoring API, authentication and roles, review queue with audit trail | planned |
| Analyst dashboard, monitoring (drift, performance), retraining workflow | planned |

**Next stage (L03):** implement the chronological split above in the `fraudml` package; build readable, time-safe features (amount relative to the card's history, hour of day, product, card type, device, e-mail domains, short-window velocity and novelty per pseudo customer); and add a test proving that features computed in batch (training) and for a single transaction (scoring) are identical.

## Limitations

- No model, scoring endpoint or explanation exists yet. The backend and frontend are skeletons.
- Most dataset features are anonymised, which limits how readable explanations can be.
- The pseudo customer key is an approximation: 11.3% of rows have none, and a few keys group many transactions.
- The time zone is unknown, so "hour of day" is relative to the reference date.
- Commands were verified on Windows 11 with Python 3.11.9 and Docker Desktop.

## Reproducibility

- **Data access:** a Kaggle account and acceptance of the IEEE-CIS competition rules are required. The data is not redistributed here.
- **Integrity check:** `verify_raw_data.py` fails if any file is missing or has an unexpected row count.
- **Deterministic pipeline:** ingestion and the EDA notebook involve no randomness. Re-running `make_parquet.py` and the notebook reproduces the numbers and figures above.
- **Environment:** Python 3.11; dependencies declared in `ml/pyproject.toml`, `backend/pyproject.toml` and `frontend/package.json` (+ `package-lock.json`). Docker images pin `postgres:16`, `python:3.11-slim` and `node:22-alpine`.
