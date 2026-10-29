# Model card — fraud scoring model `lgbm-2026.10.27-1`

One-page summary of the production model: what it is for, what it was trained on, how well it works, how it explains itself, and where it falls short. A retrained candidate (`lgbm-2026.10.28-1`) is described at the end.

## Intended use

- **Task:** score card-not-present e-commerce transactions for fraud risk and decide which ones an analyst should review.
- **Users:** fraud analysts (review queue, case explanations) and administrators (threshold, costs, models). The model **assists** a human decision; it does not block payments by itself.
- **Not intended for:** automatic refusal of payments, credit decisions, or any decision about a person without human review; data from other merchants, countries or time periods without re-validation.

## Data

- **Source:** IEEE-CIS Fraud Detection (Kaggle, Vesta Corporation): 590,540 labelled transactions, 2017-12-02 → 2018-06-01, fraud rate 3.5%.
- **Split by time** with 7-day gaps: train 2017-12-02 → 03-11 (353,716 rows), validation 03-19 → 04-20 (97,344; calibration and threshold), test 04-28 → 06-01 (99,985; evaluated **once**).
- **Labels** are chargeback-based "fraud" flags from the merchant's side; their exact definition is not public.
- **No personal data in clear:** about 415 of 434 columns are anonymised by the provider; there are no names, no demographic attributes, no protected characteristics.

## Model

- **Algorithm:** LightGBM, library default settings (500 trees, learning rate 0.05), seed 42. Tuned settings (Optuna) were **rejected**: their small gain came only from recognising customers already caught in training.
- **Inputs:** 59 curated features: time and amount, product and card type, e-mail and device families, missing-data flags, provider match/count/timing signals, and the customer's **past** behaviour (earlier transactions, 24 h activity, usual amount, novelty of device and e-mail). Computed by the same code in training and in the API (verified identical on real data).
- **Output:** a probability calibrated with isotonic regression (fitted on validation), then **expected loss = probability × amount**.
- **Decision rule:** alert when expected loss ≥ **$21.82**, the threshold that minimises total cost on validation (missed fraud amount + $5 per review + $2 friction per false alarm) at about 150 alerts per day, the analysts' capacity. Admins can override the threshold (audited).
- **Risk level shown to analysts:** low < 5% · medium < 20% · high < 50% · critical (calibrated probability; adjustable).
- **Safety net:** an Isolation Forest (`iforest-2026.10.27-1`, unsupervised) flags the 0.5% most unusual transactions the model did not alert on, as a low-priority "unusual pattern" case.

## Performance

| | Validation (calibration period) | **Test** (once, 5 weeks) | Replay (first 4 test weeks, through the platform) |
|---|---:|---:|---:|
| PR-AUC (probability) | 0.559 | **0.454** (0.455 excluding customers with a fraud in train) | 0.440 |
| Precision of alerts | 37.0% | **25.0%** | 24.0% |
| Recall of alerts | 47.1% | **39.0%** | 37.9% |
| Fraud dollars caught | 71.0% | **65.5%** | 63.9% |
| Alerts per day (mean / max) | 147.7 / — | **153 / 210** | 151.3 |
| Net savings vs no model | $434k | **$303k** of $516k fraud losses | $219k (4 weeks) |

- **Baselines on validation (PR-AUC):** amount rule 0.048, logistic regression 0.450, XGBoost (defaults) 0.529; LightGBM 0.565 before calibration.
- **Stability:** on test, 62–70% of fraud dollars are caught every week; the drop from validation is due to drift (see below). Calibrated probabilities are slightly high on test (mean 4.2% vs 3.4% fraud rate).
- **The replay** loads, scores and labels the test days through the API, database and monitoring code; its weekly results are identical to the offline test evaluation, so the deployed system behaves exactly like the evaluated model.

## Explanations

- Each score comes with its **top 10 SHAP contributions** (LightGBM's exact TreeSHAP, identical to the `shap` library), turned into sentences: *"Amount $400: strongly raises risk"*, *"Card type is credit: moderately raises risk"*.
- Anonymised provider columns are grouped into four families (counts, timing, match checks, distances) and never named; readable factors are listed first.
- The waterfall chart starts at the model's average, adds each factor and ends exactly at the transaction's raw score (checked for all 82,233 replayed transactions).
- **Caveat:** factors explain what moved the *model's* score, not why a person committed fraud. The most influential factors are often anonymised families, which limits how much an analyst can learn from them.

## Fairness

The data has no protected attributes, but some features can act as proxies for groups of customers. Error rates of the model on the replayed weeks (82,233 labelled transactions), from `python -m app.scripts.fairness_report --from 2018-04-28 --to 2018-05-26`:

**By card type**

| Group | Transactions | Fraud rate | Alert rate | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| debit | 62,383 | 2.5% | 3.0% | 25.1% | 30.5% | 2.3% |
| credit | 19,107 | 5.8% | 12.1% | 23.1% | 47.7% | 9.9% |
| (missing) | 743 | 2.0% | 4.8% | 27.8% | 66.7% | 3.6% |

**By product code**

| Group | Transactions | Fraud rate | Alert rate | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| W | 64,359 | 1.8% | 4.4% | 11.6% | 28.8% | 4.0% |
| C | 8,768 | 12.7% | 7.2% | 69.0% | 39.3% | 2.6% |
| R | 3,962 | 4.9% | 9.7% | 40.2% | 79.5% | 6.1% |
| S | 2,791 | 3.3% | 3.1% | 11.5% | 10.8% | 2.9% |
| H | 2,353 | 6.1% | 12.5% | 29.6% | 60.4% | 9.4% |

**Reading:** errors are **not evenly spread**. Legitimate **credit-card** customers are about **4 times more likely to be flagged** than debit-card customers (false-alarm rate 9.9% vs 2.3%), partly because credit cards carry more fraud (5.8% vs 2.5%) and the expected-loss rule favours larger amounts. Products H and R have the highest false-alarm rates, product S the lowest recall. A false alarm here costs a review and some friction, not a refused payment, but if card type correlated with a protected group in a real deployment, this would need review (e.g. per-group thresholds or monitoring of these rates). By card network, the small Discover group (840 transactions) has a 19.5% false-alarm rate.

## Monitoring and retraining

- Daily monitoring (on replayed days): precision, recall, PR-AUC, fraud dollars caught, net savings, alert volume vs capacity, and drift (PSI per feature and on the score against the training data).
- **Observed drift:** the normalised "days since" features drift strongly every day (PSI 1.3–2.4, they grow with the calendar) and the provider match flags on most days; the **score distribution stays stable on 27 of 28 days** (the exception, 2018-05-21, is visible on the dashboard).
- **Retrained candidate `lgbm-2026.10.28-1`:** same recipe, trained with the validation period and the first replayed week added; calibrated and thresholded ($17.92) on a later week; compared on a week **never used for fitting** (2018-05-19 → 05-25, 19,941 transactions):

| | Current | Candidate |
|---|---:|---:|
| PR-AUC (excluding customers with a fraud in training) | 0.432 (0.454) | **0.532 (0.519)** |
| Precision / recall of alerts | 25.7% / 36.7% | 35.2% / 45.7% |
| Fraud dollars caught | 64.5% | 71.0% |
| Net savings / alerts per day | $64,265 / 149 | $72,307 / 136 |

  The PR-AUC gain is +0.100 (95% interval [+0.064, +0.140], resampling whole days). Activating it is an audited admin decision on the models page.

## Limitations

- Trained on one merchant's data from 6 months of 2017–2018; patterns drift within weeks (validation → test PR-AUC 0.559 → 0.454).
- The pseudo customer key (card + address + account start) is an approximation: 11.3% of transactions have none, and some keys group several people.
- Labels are revealed at once in the replay; real chargebacks arrive weeks later, so a live system would see performance with a delay.
- The anomaly safety net added almost nothing on test (1 extra fraud in about 99 extra alerts) and is kept only as a low-priority flag.
- Most features are anonymised: some of the model's strongest signals cannot be explained to an analyst in words.
