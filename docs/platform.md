# Platform: API, security, review workflow, dashboard, monitoring, model management

How the application around the model works. For the ML side see [ml_pipeline.md](ml_pipeline.md); for every command see [commands.md](commands.md).

## Scoring API

`POST /api/v1/score` takes one raw transaction (IEEE-CIS field names; only id, time and amount are required). It returns a calibrated fraud probability, the expected loss (probability × amount), the alert decision (expected loss ≥ $21.82 by default; an admin can change it), a risk level (low < 5% · medium < 20% · high < 50% · critical), the "unusual pattern" flag and the top 10 plain-language factors. The transaction and its prediction are stored in PostgreSQL.

- **One feature pipeline:** the backend installs the `fraudml` package and calls its online feature path. Only the history lookup is new (the customer's earlier transactions, read from PostgreSQL).
- **Model registry:** `model_versions` holds each scoring bundle (LightGBM model, calibrator, alert policy, linked Isolation Forest). Registration refuses artifacts trained with another feature encoder or other library versions. At most one bundle is active, and the API loads it at startup.
- **Safe to retry:** the same transaction sent twice returns the stored prediction; different data under the same id is rejected (409). Ground truth can never be sent (`isFraud` → 422) and is never returned.
- **Batch scoring** uses the same service for a date range of stored transactions.

**Training–serving consistency, re-checked on PostgreSQL:** for 1,000 random validation and test transactions, features computed by the API path equal the offline training features (0 mismatches over 59 features, largest difference 2e-12), and probabilities are identical. The first test day scored in batch (3,060 transactions) reproduces offline scoring exactly: same probabilities, same 154 alerts, same 2 unusual flags.

Scoring takes about 140 ms per transaction (features 66 ms, Isolation Forest 30 ms, SHAP 24 ms, LightGBM 10 ms, history query 4 ms).

## Authentication, roles and audit trail

- **Login** (`POST /auth/login`, OAuth2 password form) returns a 15-minute JWT access token, sent as `Authorization: Bearer …`. A 7-day refresh token is set as an **httpOnly cookie** (SameSite=Strict, limited to `/api/v1/auth`), so page scripts cannot read it. `POST /auth/refresh` swaps it for a new one each time. If an already-used refresh token is presented again, it is treated as stolen and all of that user's sessions are revoked. `POST /auth/logout` revokes it.
- **Stored safely:** passwords are hashed with bcrypt (10 to 72 bytes). Only a SHA-256 hash of each refresh token is stored. A wrong password and an unknown e-mail get the same answer, so the login does not reveal which accounts exist.
- **Two roles:** analyst and admin. `GET/POST /users` is admin only. The user is re-read on every request, so a deactivated account or a changed role takes effect immediately. `POST /score` needs any logged-in user.
- **Demo users** come from a seed script, with passwords taken only from `.env`.
- **Audit trail:** every state-changing action writes one row to `audit_logs` in the same transaction as the change: `auth.login`, `auth.login_failed` (with a reason, never the password), `auth.logout`, `auth.token_reuse`, `user.create`, `model.register` and `model.activate`. **PostgreSQL itself refuses UPDATE, DELETE and TRUNCATE** on this table (trigger), and a user with audit history cannot be deleted. The audit service also refuses any field that looks like a password or token.

## Review queue and case workflow

Every alert becomes a **case** that an analyst reviews. All endpoints need a logged-in user; settings are admin only.

- **Cases are opened by scoring** (API and batch, same database transaction): an alert opens a tier-1 case (reason "model score"), an "unusual pattern" flag a tier-2 case. One case per transaction. A backfill script does the same for predictions stored earlier.
- **Queue order:** model alerts first, ranked by **expected loss** (probability × amount, the quantity the threshold was optimised on), then unusual-pattern cases. Case id breaks ties, so pages are stable. Days with more than 150 alerts are not cut: unreviewed cases stay in the queue.
- **`GET /cases`** filters by status, risk level, reason, assignee (`me`, `unassigned` or a user), amount and transaction date, sorts by priority, amount or time, and pages results with a total. **`GET /cases/{id}`** adds the transaction facts, the score and plain-language risk factors, the customer's 10 earlier transactions (with a "had a case" flag only) and the decisions.
- **Workflow:** `claim` (`new` → `assigned`), `assign` (admin), `decision`: *confirmed fraud* or *false positive* closes the case; *escalate* (comment required) unassigns it for an admin to claim and close. Only the assigned person can decide; closed cases cannot change. The case row is locked during a change, so two analysts cannot claim the same case.
- **Timeline:** `GET /cases/{id}/timeline` is read from the audit trail (`case.create` by the system, `case.claim`, `case.assign`, `case.decide` with the comment).
- **Settings (admin):** `GET/PUT /settings/thresholds` (alert threshold override, `null` = the active model's $21.82, and the risk-level bands) and `GET/PUT /settings/costs` (review cost, false-alarm cost, daily capacity). Changes are audited with before/after values and apply to transactions scored afterwards, not retroactively.
- **CSV export:** `GET /reports/cases.csv` with the queue filters, in queue order, with the final decision and who made it. Cells that a spreadsheet would run as a formula are escaped.
- **Ground truth is never returned** by any case, timeline or export endpoint (checked by tests).

Checked on the demo database: 156 cases for the first test day (154 model alerts + 2 unusual); an analyst claims and confirms the top case ($994, expected loss $481) while a second analyst is refused (409/403); an escalated case is closed by an admin; the timeline, settings changes and CSV export each match the audit rows written.

## Analyst dashboard

A React + TypeScript web app (Mantine UI, TanStack Query and Table) on top of the API. Analysts and admins log in with a seeded account.

- **Login and session:** the short-lived access token is kept in memory only (never in browser storage); the refresh token is an httpOnly cookie. A page reload restores the session with **one** refresh request, and requests that fail with 401 share a single refresh before retrying (the backend treats a refresh token used twice as stolen). Pages need a logged-in user; admin-only pages show a 403 page to analysts.
- **Review queue:** tabs *Open* / *My cases* / *All*; filters (risk level, reason, amount, transaction date, status); sortable columns (expected loss, amount, time, age) and pages, all done by the API and **kept in the address bar** (reload, back button and shared links keep the view); *Claim*, and *Assign* for admins.
- **Case detail:** score gauge (probability as a whole percent), risk level, expected loss, "unusual pattern" note, transaction facts, the customer's earlier transactions, and a **SHAP waterfall**: bars start at a typical transaction, each factor pushes the risk up (red) or down (green), and the last bar lands on this case's score. The plain-language reasons are listed next to it, readable ones first, anonymised signals after.
- **Decisions:** only the analyst holding the case can decide; closing a case asks for confirmation; escalating needs a comment and hands the case to the admins. The **timeline** turns the audit trail into sentences ("Escalated to the admins", "Decision: false positive", with comments).
- **Typed against the API:** TypeScript types are generated from the backend's OpenAPI description, so a backend change that breaks the UI fails the build (this caught a mismatch on the first build).
- **Ground truth is never shown:** the API never returns it, and the built app contains no reference to it.


## Monitoring

The test period is **replayed day by day through the real system**: each day's transactions are loaded, scored by the API's scoring service (which opens review cases), then that day's labels are revealed for monitoring only (the analyst UI never shows them). 28 days (2018-04-28 → 05-25, 82,233 transactions) are replayed; the last test week is kept for the live demo.

- **Exact check:** the weekly results computed from the replay equal the one-time offline test evaluation for all 4 weeks (transactions, alerts per day, precision, fraud $ caught, net savings, to 3e-16). This verifies loading, customer history, features, scoring, labels and metrics together on real data.
- **Performance per day** (same code as the offline evaluation): precision and recall of the alerts, PR-AUC, precision within the 150/day capacity, share of fraud dollars caught, total cost and net savings. Daily counts are stored, so weeks and periods are summed exactly.
- **Drift** against a reference built once: training-period features (10 quantile bins + a missing-value bin, or one bin per category) and validation-period scores. PSI decides the severity (< 0.1 stable, ≤ 0.25 moderate, above significant); KS is reported but not used, since with ~2,900 rows a day its p-value flags harmless differences. `weekday` is not monitored per day (a one-day window always holds one weekday).
- **Dashboard page** (`/monitoring`, analysts and admins): KPI cards (last 7 days vs the whole period), daily performance chart, alert volume with the capacity line and the open backlog, a drift heatmap (model score + the 15 most drifting features), analyst feedback.
- **Replay speed:** about 5–6 minutes per day with 4 worker processes (results identical to scoring one by one, checked by the weekly comparison).

| Replayed period (28 days) | Value |
|---|---|
| Alerts per day | 151.3 (capacity 150; 14 of 28 days above it) |
| Precision / recall of alerts | 24.0% / 37.9% |
| PR-AUC | 0.440 |
| Fraud $ caught | 63.9% |
| Net savings | $219,148 |

**What the drift shows:** the four normalised "days since" features (`D1n`, `D4n`, `D10n`, `D15n`) drift strongly every day (PSI 1.3–2.4) because they grow with the calendar, and the `M1`–`M9` match flags drift on most days (PSI ≈ 0.6). Yet **the model's score distribution stays stable on 27 of 28 days** (mean PSI 0.03), and precision holds. The exception is 2018-05-21 (score PSI 0.32, only 83 alerts): exactly the kind of day the dashboard is meant to surface.

## Model management and retraining

The feedback loop is closed with an **offline retraining script** and **manual champion/challenger promotion**.

- **Retraining** uses the labels now in the database (revealed labels of replayed days, else analyst decisions). The challenger is trained on the original train + validation periods + replayed week 1; week 2 is a gap; week 3 calibrates it and sets its cost-optimal threshold; **week 4 (2018-05-19 → 05-25) is never used for fitting** and serves to compare both models on the same transactions. Same features, encoder and LightGBM recipe as the current model.
- **Result** (`reports/retrain_comparison.md`, 19,941 transactions):

| Week 4 | Current model | Retrained candidate |
|---|---:|---:|
| PR-AUC (excluding customers with a fraud in training) | 0.432 (0.454) | **0.532 (0.519)** |
| Precision / recall of alerts | 25.7% / 36.7% | **35.2% / 45.7%** |
| Fraud $ caught | 64.5% | **71.0%** |
| Net savings | $64,265 | **$72,307** |
| Alerts per day | 149 | 136 |

  The PR-AUC gain is +0.100 with a 95% interval of [+0.064, +0.140] (bootstrap over whole days). Newer training data (closer to the drifted period) is the likely reason.
- **Model management page** (admin): the candidate next to the current model on the comparison week (all customers / excluding past-fraud customers, with the confidence of the gain), the versions table, and **Activate** / **Archive**. Activation is audited, archives the previous model, resets a custom threshold to the new model's own, and switches the API to the new model **without a restart** (it is loaded first; a broken artifact changes nothing). A previous model can be re-activated.
- **Threshold and costs page** (admin): use the model's threshold or a custom one, the risk-level bounds, the review / friction costs and the daily capacity, with the same validation as the API, next to the training cost curve that explains the threshold.
- **Export CSV** button on the review queue (current filters).
