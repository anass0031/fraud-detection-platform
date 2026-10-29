# ML pipeline: data, features, models, explanations

How the model was built and evaluated, step by step. Every number here comes from the scripts in `ml/scripts/` and the reports in [`reports/`](../reports/). For the model's summary sheet see the [model card](model_card.md); for the application see [platform.md](platform.md).

## What the data looks like (EDA)

- **Fraud rate: about 3.50%** in the labelled data (20,663 of 590,540; 1 fraud for every 27.6 legitimate transactions), about 113 frauds per day. The weekly rate varies between 2.3% and 4.9%.
- **Mostly masked features:** about 415 of 434 columns are anonymised (`V`, `C`, `D`, `M`, `id_`). Analyst-readable explanations therefore rely on engineered features built from the readable columns.
- **Missing values carry signal:** identity data exists for 24.4% of transactions, and those transactions are 3.8× riskier (7.85% vs 2.09%).
- **Strong readable signals:** product C (11.7% fraud vs 2.0% for W), credit vs debit cards (6.7% vs 2.4%), mobile devices (10.2%), low-volume hours of the day (up to 10.6%), certain e-mail domains. Raw amount alone is a weak signal.
- **Drift:** browser version (`id_31`) and some time-delta columns (`D15`) drift strongly over time; amount and card features are stable.
- **No single-column leak:** the best single numeric column reaches ROC-AUC 0.69.

![Daily volume and fraud rate](../reports/figures/eda/02_volume_and_fraud_rate_over_time.png)

## Chronological split

The model predicts *future* transactions, so data is split by time, never randomly. There is a **7-day gap** before validation and before test, so look-back features and late-arriving labels cannot leak across periods.

| Period | Dates (inclusive) | Rows | Share | Fraud rate |
|---|---|---|---|---|
| Train | 2017-12-02 → 2018-03-11 | 353,716 | 59.9% | 3.38% |
| Gap | 2018-03-12 → 2018-03-18 | 20,946 | 3.5% | – |
| Validation | 2018-03-19 → 2018-04-20 | 97,344 | 16.5% | 3.93% |
| Gap | 2018-04-21 → 2018-04-27 | 18,549 | 3.1% | – |
| Test | 2018-04-28 → 2018-06-01 | 99,985 | 16.9% | 3.43% |

Gap rows are not used for training or scoring. They may be used as past history when computing behavioural features.

![Time split](../reports/figures/eda/05_time_split.png)

## Leakage safeguards

The EDA confirmed strong **label propagation**: when a pseudo customer (`uid = card1 + addr1 + (day − D1)`) has a fraudulent transaction, their next transaction is fraudulent 87.2% of the time (vs 0.36% otherwise). Therefore:

1. **No features built from past labels** of the same customer or card. In practice, labels (chargebacks) arrive weeks later.
2. **No raw identifiers as features:** neither `uid` nor `TransactionID`. Only aggregates over *past* transactions are allowed.
3. **Test metrics are reported twice:** overall, and excluding customers who already had a fraud in the training period. That small group (0.7% of test rows) accounts for about 10% of test frauds and would otherwise inflate results.
4. Encoders, scalers and any resampling are fitted on training data only; behavioural aggregates use strictly earlier transactions.
5. Rare high-cardinality categories (e.g. specific device models) are grouped into `other` or frequency-encoded to avoid memorisation.

## Data ingestion and exploratory analysis

**Data ingestion** (`ml/fraudml/data/ingest.py`):
- verifies the five raw files and their exact row counts;
- renames the test identity columns (`id-01` → `id_01`) so train and test share one schema;
- left-joins identity data onto transactions (`TransactionID`, one-to-one check);
- adds `TransactionDatetime` from the `TransactionDT` seconds offset (reference date 2017-12-01, a common convention);
- downcasts types (float64 → float32, smallest integer types, text → category) while keeping `TransactionAmt` in float64 so cents stay exact;
- writes compressed Parquet. Memory drops from ~1.98 GB to ~0.93 GB for the training data, and loading takes about 1 second. Values were checked to be identical to the CSVs, missing-value patterns included.

**Exploratory analysis** (`ml/notebooks/01_eda.ipynb`, summary in [`eda_findings.md`](eda_findings.md)):
size and column groups, target imbalance, missing values, readable features vs. fraud, time coverage, chronological split, drift (PSI), a pseudo customer key, and leakage checks.

## Feature pipeline

`ml/fraudml/features/` turns raw transactions into **59 curated features** (full list with plain-language meanings in [`features.md`](features.md)):
- **row-level:** hour, weekday, amount (log, cents, round flag), product, card network/type, e-mail domains and a same-domain flag, device type, browser and OS *family*, missing-data flags, masked `C`/`M` columns, and drift-free `day − D` versions of key `D` columns;
- **customer behaviour, past only:** earlier transactions, last-24 h count and amount, usual amount, amount vs. usual amount, how unusual the amount is, time since the previous transaction, first time on a device / e-mail domain;
- **encoding fitted on the train period only:** rare or unseen categories become `other`; card, region, e-mail and device frequencies are learned from train.

The **same code** computes features for a whole dataset (training) and for **one transaction plus its history** (the scoring API). Tests prove:
- **training/serving consistency:** online features equal batch features for every synthetic row, and for 1,992 sampled real validation/test transactions (0 mismatches, max difference 3e-11), including same-second transactions, customers without a key, long histories and unseen categories;
- **no look-ahead:** removing all later transactions never changes a feature (synthetic data and 48,672 real validation rows);
- **no label use:** flipping or removing `isFraud` leaves every feature unchanged;
- **no identifiers:** `TransactionID`, `uid`, timestamps and evaluation flags can never appear in a feature list.

Feature datasets keep `TransactionID`, `TransactionDT`, `period`, `isFraud` and the evaluation-only flag `uid_fraud_in_train` (validation 965 rows, test 663 rows) as separate columns, never as features.

## Evaluation and baselines

`ml/fraudml/evaluate/` measures models the way the business sees them: PR-AUC, recall at precision ≥ 0.5, precision within each day's top 150 alerts (analyst capacity), share of fraud **value** caught, and a cost model ($5 per review, $2 friction per false alarm, a missed fraud costs its amount). The alert threshold is chosen to minimise total cost while staying within 150 alerts per day on average. Every result is reported overall and excluding customers who already had a fraud in train.

Validation period results (models trained on the train period; test period untouched):

| Model | PR-AUC | Precision in daily top 150 | Net savings vs no model |
|---|---|---|---|
| Rule: amount > train p99 | 0.048 | 0.084 | $73k |
| Logistic regression | 0.450 | 0.369 | $223k |
| **LightGBM (defaults, no resampling)** | **0.565** | **0.453** | **$349k** |
| LightGBM + class weights | 0.542 | 0.428 | $325k |
| LightGBM + undersampling (probabilities corrected) | 0.555 | 0.444 | $352k |

A day-level bootstrap shows LightGBM without resampling is best on ranking, and class weights are clearly worse. Review capacity, not the model, limits savings: cost keeps falling up to about 500 alerts/day. Details: [`reports/baselines_findings.md`](../reports/baselines_findings.md).

![PR curves](../reports/figures/baselines_pr_curves.png)

## Production model and test results

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

Details: [`reports/model_findings.md`](../reports/model_findings.md).

![Weekly stability on test](../reports/figures/test_weekly.png)

## Anomaly safety net

An **Isolation Forest** (unsupervised, fitted on the train period without labels) flags unusual transactions. Transactions that the main model did not alert but that are among the 0.5% most unusual of the train period are flagged as an "unusual pattern".

| | Validation | Test (checked once) |
|---|---|---|
| Extra alerts per day | 2.9 | 2.8 |
| Extra frauds caught | 9 (precision 9.4%) | 1 (precision 1.0%) |
| Net savings change | +$117 | −$516 |

**Result:** the safety net adds almost nothing here. Most unusual transactions are already caught by the main model, and on test the remaining extra alerts are less precise than random review. It is kept only as a **low-priority "unusual pattern" flag** (about 3 per day, reviewed when capacity is left). A one-off burst of unusual transactions in December 2017 also makes the train-based cut-off stricter than intended, which argues for refitting it on recent traffic. Details: [`reports/anomaly_findings.md`](../reports/anomaly_findings.md).

## SHAP explanations

Every transaction gets its **top risk factors in plain language**, computed with LightGBM's built-in exact SHAP values (identical to the `shap` library, about 18 ms per transaction). Anonymised columns are grouped into four families and never named.

> Amount $554: moderately raises risk
> 24.5 h since this customer's previous transaction: moderately raises risk
> Card type is credit: strongly raises risk

The explanation object (top 10 factors with direction, strength and a `masked` flag) is stored with each prediction. The anonymised families drive the score most, so the case page lists readable reasons first. Factors describe what moved the model's score, not causes. Details and 10 reviewed examples: [`reports/shap_findings.md`](../reports/shap_findings.md), [`reports/shap_examples.md`](../reports/shap_examples.md).

![SHAP importance](../reports/figures/shap_importance.png)
