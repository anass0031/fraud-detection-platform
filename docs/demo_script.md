# Demo script (about 15 minutes)

The live part of the defense, step by step: what to click or type, what to say, how long it takes, and what to do if a step fails. Every step was rehearsed through the API on a copy of the demo database (timings below are from that rehearsal).

- **Accounts:** admin = `SEED_ADMIN_EMAIL` from `.env`; analysts = `analyst1@fraud.local`, `analyst2@fraud.local` (password `SEED_ANALYST_PASSWORD`).
- **Demo state:** 28 test days replayed (2018-04-28 → 05-25), about 4,300 open cases, active model `lgbm-2026.10.27-1`, candidate `lgbm-2026.10.28-1` (retrained on the replayed weeks), **2018-05-26 not replayed** (it is replayed live).
- **Windows to open beforehand:** a terminal at the repo root, the app (http://localhost:5173) logged in as Analyst One, a second browser profile (or private window) logged in as admin, Swagger (http://localhost:8000/docs), and the backup recording ready to play.

## Before the defense (morning, 10 minutes)

| # | Action | Expected |
|---|---|---|
| 1 | Start Docker Desktop, then `docker compose up -d` | 4 containers up (`docker compose ps`) |
| 2 | Restore the demo snapshot (see [Demo database snapshot](#demo-database-snapshot)) | about 40 s |
| 3 | `docker compose exec backend python -m app.scripts.demo_check` | 13 lines `PASS`, `RESULT: PASS` |
| 4 | Log in to the app in both browser windows, open Swagger and click **Authorize** (admin) | queue and dashboard load |

If `demo_check` fails, its detail column says what to do (restore the snapshot, run `seed_users`, restart the backend). Do not start the demo on a `FAIL`.

## The live demo

| Time | Step | What to do | What to say | If it fails |
|---|---|---|---|---|
| 0:00 | **1. Start the next day** | Terminal: `docker compose exec backend python -m app.scripts.replay --from 2018-05-26 --to 2018-05-27 --workers 4` (`--to` is exclusive: this is one day) | "This replays the next day of the test period: load the day's transactions, score them, open review cases, then reveal the labels and update the monitoring. It takes about three minutes, so I'll show the rest meanwhile." | Leave it; show the replayed days that are already there. The recording shows a replay. |
| 0:30 | **2. Review queue** (analyst window) | Queue page: model alerts first, sorted by expected loss; show the risk badges and filters (status, risk, amount) | "Every alert becomes a case. The queue is ordered by expected loss, probability × amount, so a likely $2,000 fraud comes before a likely $20 one. Analysts never see the true label." | Refresh the page; log in again (token expired). |
| 1:30 | **3. Claim and explain the top case** | Click the top case → **Claim** → detail page: score gauge, expected loss, SHAP waterfall, plain-language reasons, card history | "The waterfall starts at a typical transaction; each factor pushes the risk up (red) or down (green) and the last bar is this case's score. The sentences under it are generated from the same SHAP values." | Open any other `critical` case. |
| 3:00 | **4. Decide** | Decision **Confirmed fraud**, short comment (e.g. "Amount far above the card's usual spend") → confirm in the dialog → case closed | "The decision and the comment are stored; every state change also writes a row in an append-only audit log, enforced by a database trigger." | Use **False positive** instead; any decision shows the same flow. |
| 3:45 | **5. A case the model missed** | Queue filter **Reason = unusual pattern**, **Risk = low** → open the top one | "This one has a low fraud score, but the Isolation Forest flagged it as unusual, so it is still reviewed, in a second tier after the model alerts. It is a cheap extra net: about 3 flags a day, and on the test period it caught only one extra fraud." | Any `unusual pattern` case. |
| 4:45 | **6. Monitoring** (admin window) | Dashboard: KPI cards, daily precision/recall, alert volume vs capacity (150/day), drift heatmap; point at 2018-05-21 | "Performance per day once the labels are revealed. Several input features drift strongly (the day counters grow with the calendar), but the model's score distribution stays stable on 27 of 28 days. 2018-05-21 is the exception: that is what this page is for." | Reload; the numbers are also in the model card. |
| 6:45 | **7. Threshold** | Settings: show the threshold in force ($21.82 expected loss) and the cost settings; change the alert threshold to e.g. 30, **Save thresholds**; then clear it again (back to the model's value) | "The threshold comes from the cost model: reviewing costs $5, a missed fraud costs its amount. An admin can override it; the change is audited." | Skip; show `GET /settings/thresholds` in Swagger. |
| 7:45 | **8. Back to the queue** | Analyst window: refresh the queue, filter on 2018-05-26 | "The day we started at the beginning is now scored: these are its new alerts." | If the replay is still running, come back after step 10. |
| 8:30 | **9. Promote the retrained model** | Models page: compare candidate `lgbm-2026.10.28-1` with the active model (metrics, training window) → **Activate** → confirm in the dialog | "The candidate was retrained offline on the replayed weeks with the revealed labels. Promotion is manual, swaps the model without a restart, and is audited. Promoting also resets any threshold override to the new model's own value." | Do this only **after the replay has finished**, so the day is scored by one model. Skip if late. |
| 9:45 | **10. Score one transaction live** | Swagger: `POST /api/v1/score` → **Try it out** → paste [`docs/demo/score_example.json`](demo/score_example.json) → **Execute** | "One raw transaction in, a calibrated probability, the expected loss, the alert decision and the top reasons out, in about 0.3 seconds. The features are computed by the same code as in training." | Sending the same id twice returns `already_scored: true` (the transaction is stored once): change `TransactionID`, keeping it above 4,200,000 (outside the dataset's ids, so it can never collide with a day the replay loads later). |
| 11:00 | **11. Audit trail** | **Timeline** card on the detail page of the case from step 4, or Adminer: `audit_logs` ordered by `id desc` | "Claim, decision, threshold change, promotion: each has a row with who, when, before and after." | Show the timeline only. |
| 12:00 | End of the live part | Back to the slides (limitations, future work) | | |

About 3 minutes of margin are left for questions during the demo. If anything blocks for more than 30 seconds, switch to the backup recording at the same step.

## Rehearsal through the API (2026-10-29)

Done on the demo database, then the database was restored from the snapshot (`demo_check` PASS after the restore).

| Step | Call | Result | Time |
|---|---|---|---|
| 1 | `replay --from 2018-05-26 --to 2018-05-27 --workers 4` | 3,032 rows loaded and scored, 203 alerts + 2 unusual → 205 new cases, labels revealed, monitoring updated | 2 min 42 s (allow up to 6 min: earlier days took 5–6 min) |
| 2 | `POST /auth/login` (analyst), `GET /cases?status=new` | 200, 4,318 new cases | < 0.1 s |
| 3 | `POST /cases/1614/claim`, `GET /cases/1614` | 200, 200 (score 85%, critical, 9 earlier transactions) | 0.05 s, 0.06 s |
| 4 | `POST /cases/1614/decision` (`confirmed_fraud` + comment) | 200, case closed, timeline has the claim and the decision | 0.03 s |
| 5 | `GET /cases?reason=anomaly&risk_level=low` | 64 cases (top one: $500, score 2.8%) | < 0.1 s |
| 6 | `GET /monitoring/performance`, `/drift`, `/alerts`, `/feedback` | 200 (28 days) | 1.1 s, < 0.1 s each |
| 7 | `GET /settings/thresholds`, `PUT` 30 | 200, threshold in force $30 (was $21.82), audited | < 0.1 s |
| 8 | `GET /cases?status=new&date_from=2018-05-26&date_to=2018-05-26` | 205 new cases from the replayed day | < 0.1 s |
| 9 | `POST /models/2/activate` | 200, `/health` serves `lgbm-2026.10.28-1` without a restart; `threshold_override_reset: true` (threshold back to the new model's $17.92) | 0.55 s |
| 10 | `POST /score` with `docs/demo/score_example.json` | 200: probability 22%, expected loss $322, alert, risk high, 5 readable reasons | 0.34 s |
| 11 | `audit_logs` (newest first) | `settings.update` (override reset), `model.activate`, `settings.update` (threshold 30), `replay.day`, `case.create` × 205, `case.decide`, `case.claim` | |
| after | `demo_check` on the used database | `FAIL (3 of 13)`: no candidate left, no monitoring for the new active model yet, demo day already loaded | |
| reset | restore from the snapshot | 39 s; row counts identical to the snapshot; `demo_check` PASS | |

Notes from the rehearsal:
- `--to` is exclusive: `--from 2018-05-26 --to 2018-05-26` is refused ("--to must be after --from").
- The decision body field is `comment` (not `note`).

## Demo database snapshot

The snapshot is a `pg_dump` in custom format (compressed) of the whole demo database: users, cases, decisions, audit log, predictions, monitoring, both models. It is kept in `backups/` (git-ignored: it contains the demo users' password hashes and is 76 MB). The model files themselves are in `artifacts/` (also ignored), which the snapshot does not include.

Run from the repository root. In **Git Bash**, put `MSYS_NO_PATHCONV=1 ` in front of each command (otherwise `/tmp` is rewritten into a Windows path); PowerShell needs nothing.

**Take a snapshot** (about 15 s):

```bash
docker compose exec db pg_dump -U fraud -d fraud -Fc -f /tmp/demo.dump
docker compose cp db:/tmp/demo.dump backups/demo-YYYY-MM-DD.dump
```

**Restore it** (about 40 s; replaces the demo database completely, including its audit log):

```bash
docker compose stop backend
docker compose cp backups/demo-YYYY-MM-DD.dump db:/tmp/demo.dump
docker compose exec db dropdb -U fraud --force fraud
docker compose exec db createdb -U fraud fraud
docker compose exec db pg_restore -U fraud -d fraud --no-owner --exit-on-error /tmp/demo.dump
docker compose start backend
docker compose exec backend python -m app.scripts.demo_check
```

The restore was tested into a scratch database: all 12 tables have the same row counts as the source and both `audit_logs` protection triggers are restored.
