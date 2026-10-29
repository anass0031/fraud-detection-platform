# Defense material

Slide-by-slide outline for the defense, with the figure and the numbers to use on each slide, followed by the shot list for the backup screen recording. Every number comes from a file in `reports/` or `docs/` (source given per slide), so it can be checked or regenerated.

Suggested timing: **≈ 12 min of slides + ≈ 12 min live demo** ([demo_script.md](demo_script.md)) + questions. Slides 1–9 come before the demo, 10–12 after it.

## Slides

### 1. Title
Fraud Detection and Explainable AI Platform: scoring card transactions, explaining every score, routing cases to analysts. Name, programme, date.

### 2. The problem in numbers
- 590,540 real e-commerce transactions (IEEE-CIS / Vesta), 182 days; **3.50% fraud** (1 in 28), 3.87% of the money ($3.08 M of $79.7 M).
- **Why accuracy fails:** "never fraud" is 96.5% accurate and catches nothing. What matters is money caught within what analysts can review.
- **Cost model:** a missed fraud costs its amount; each alert costs $5 of review; a false alarm adds $2 of customer friction; capacity **150 reviews per day** (about 3,050 transactions and 113 frauds per day).
- Figure: `reports/figures/eda/02_volume_and_fraud_rate_over_time.png`
- Source: [eda_findings.md](eda_findings.md)

### 3. What the system does (architecture)
- Analyst queue ranked by expected loss → explained case → decision; admin monitoring, thresholds, model promotion; scoring API; append-only audit trail; day-by-day replay; offline retraining.
- Figure: the Mermaid diagram from the README ("How it works"), exported as an image.
- Key message: **one feature pipeline** (the API installs the same `fraudml` package as training; 0 mismatches between training and serving on real data).

### 4. Rigour 1: time split and leakage
- Split by time with 7-day gaps: train 12-02 → 03-11 (353,716) · validation 03-19 → 04-20 (97,344) · test 04-28 → 06-01 (99,985), evaluated **once**.
- **Label propagation found in the EDA:** after a fraud, a customer's next transaction is fraud **87.2%** of the time (0.36% otherwise). So: no feature uses past labels, no raw ids, and every test metric is also reported **excluding customers with a fraud in train**.
- Figure: `reports/figures/eda/05_time_split.png`
- Source: [eda_findings.md](eda_findings.md), [ml_pipeline.md](ml_pipeline.md)

### 5. Rigour 2: models and imbalance (validation)
| Model | PR-AUC |
|---|---:|
| Rule: amount > p99 | 0.048 |
| Logistic regression (balanced) | 0.450 |
| XGBoost (defaults) | 0.529 |
| LightGBM, no resampling | **0.565** |
| LightGBM, class weights | 0.542 |
| LightGBM, undersampling | 0.555 |
- No imbalance correction beats none (day-block bootstrap: class weights −0.025 [−0.035, −0.016]), and the model without resampling stays calibrated.
- Figure: `reports/figures/baselines_pr_curves.png`
- Source: `reports/baselines_comparison.md`, `reports/model_findings.md`

### 6. Rigour 3: two honest trade-offs
- **Tuning rejected:** Optuna raised validation PR-AUC (0.565 → 0.574) but *lowered* it on customers not seen as fraudsters (0.564 → 0.548, interval [−0.026, −0.005]): bigger trees re-identified known fraudsters. A higher score can hide leakage.
- **Curated vs full features:** 59 readable features vs 409: PR-AUC 0.565 vs 0.595, but net savings $349k vs $343k. Readability costs some ranking and almost no money.
- Source: `reports/model_findings.md`

### 7. Decision rule: expected loss, not probability
- Alert when **probability × amount ≥ $21.82** (cost-optimal on validation under 150/day).
- Against a probability threshold at the same 148 alerts/day: net savings **$434k vs $342k**, fraud dollars caught 71% vs 56.5% (precision lower, 0.37 vs 0.48: more false alarms, on larger amounts).
- Figures: `reports/figures/baselines_cost_curve.png`, `reports/figures/model_cost_policies.png`
- Source: `reports/model_findings.md`

### 8. Results on the test period (evaluated once)
| | Validation | **Test** |
|---|---:|---:|
| PR-AUC | 0.559 | **0.454** (0.455 excluding known fraudsters) |
| Precision of alerts | 37.0% | **25.0%** |
| Fraud $ caught | 71.0% | **65.5%** (target ≥ 50%) |
| Alerts per day (mean / max) | 148 | **153 / 210** |
| Net savings | $434k | **$303k of $516k** |
- 62–70% of fraud dollars caught **every week**; model ranking holds on test (LightGBM 0.454 > XGBoost 0.440 > logistic 0.372 > rule 0.036).
- The drop is drift, not overfitting (slide 11).
- Figures: `reports/figures/test_weekly.png`, `reports/figures/test_pr_curves.png`
- Source: `reports/test_results.md`, [model_card.md](model_card.md)

### 9. Explanations
- Top 10 SHAP factors per transaction, turned into sentences ("Card type is credit: strongly raises risk"); waterfall adds up exactly to the score (checked).
- About 415 of 434 dataset columns are anonymised: readable factors are listed first, anonymised families after.
- Figures: `reports/figures/shap_importance.png` (or `shap_beeswarm.png`), plus a screenshot of a case page's waterfall.
- Source: `reports/shap_findings.md`, `reports/shap_examples.md`

→ **Live demo** (≈ 12 min, [demo_script.md](demo_script.md)).

### 10. The deployed system behaves like the evaluated model
- **Replay:** 28 test days loaded, scored and labelled through the API, database and monitoring code (82,233 transactions): weekly results **identical** to the offline evaluation (largest difference 3e-16).
- Tests: 182 ML · 215 backend · 145 frontend; backend services 97.6% of lines covered.
- Source: [platform.md](platform.md), [test_report.md](test_report.md)

### 11. Monitoring, retraining and fairness
- **Drift:** the "days since" features drift strongly (PSI 1.3–2.4, they grow with the calendar), yet the model's score distribution is stable on 27 of 28 days (mean PSI 0.03); 2018-05-21 stands out (score PSI 0.32).
- **Retraining** on the replayed weeks, compared on an untouched week: PR-AUC **0.432 → 0.532** (gain +0.100, interval [+0.064, +0.140]), precision 25.7% → 35.2%, fraud $ caught 64.5% → 71.0%, with fewer alerts (149 → 136/day). Promotion is manual and audited.
- **Fairness:** legitimate credit-card customers are flagged **4× more often** than debit (false-alarm rate 9.9% vs 2.3%); products H and R also high. Visible in the model card; would need review in a real deployment.
- Figure: screenshot of the monitoring dashboard (drift heatmap).
- Source: [platform.md](platform.md), `reports/retrain_comparison.md`, [model_card.md](model_card.md)

### 12. Limitations and future work
- **Limitations:** drift lowers performance (validation → test PR-AUC 0.559 → 0.454; probabilities slightly high); anonymised features limit explanations; pseudo customer key is approximate (11.3% of rows have none); labels revealed at once (real chargebacks take weeks, selection bias: only reviewed cases get analyst labels); the Isolation Forest safety net adds ~3 flags/day and caught **1** extra fraud on test (a cheap net, not a second detector); ~0.2 s per score (one day of replay ≈ 3–6 min); no login rate limiting.
- **Future work:** scheduled retraining with delayed labels; streaming ingestion (broker) instead of daily batches; graph features (shared devices, e-mails, addresses); per-group threshold monitoring; live threshold what-if simulator; audit-log and users pages.
- Source: README "Limitations", [model_card.md](model_card.md), `reports/anomaly_findings.md`

## Backup recording: shot list

Record after restoring the snapshot and a `demo_check` PASS, at 1920×1080, browser zoom 110%, one take per shot (cut together later). Total ≈ 10 min. Restore the snapshot afterwards.

| # | Shot | Length | Notes |
|---|---|---|---|
| 1 | Terminal: `demo_check` → `RESULT: PASS` | 15 s | shows the starting state |
| 2 | Terminal: start the replay of 2018-05-26 (`--to 2018-05-27`) | 15 s | keep it running in the background |
| 3 | Analyst: login → review queue (sort, filters, risk badges) | 45 s | |
| 4 | Top case → **Claim** → detail page: gauge, expected loss, waterfall, reasons, earlier transactions | 90 s | hover a waterfall bar |
| 5 | Decision **Confirmed fraud** + comment → dialog → closed; **Timeline** card | 45 s | |
| 6 | Queue: Reason = unusual pattern, Risk = low → open one | 40 s | |
| 7 | Admin: monitoring dashboard (KPIs, performance chart, alerts vs capacity, drift heatmap; point at 05-21) | 90 s | |
| 8 | Admin: threshold page: change to 30, **Save thresholds**, clear it again | 40 s | |
| 9 | Terminal: replay finished (205 new cases) → queue filtered on 2018-05-26 | 30 s | wait for the end of the replay |
| 10 | Admin: models page, compare, **Activate** candidate | 60 s | only after shot 9 |
| 11 | Swagger: `POST /score` with `docs/demo/score_example.json` → response | 60 s | |
| 12 | Adminer: `audit_logs` ordered by `id desc` | 30 s | claim, decision, settings, activation rows |
