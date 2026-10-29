# Test report

Measured on 2026-10-29 (Windows 11, Docker Desktop). All suites pass.

| Part | Tests | Coverage | Command |
|---|---:|---|---|
| ML package (`fraudml`) | **182** | **96.2%** of lines + branches | `ml/.venv/Scripts/python -m pytest ml/tests -q --cov=fraudml --cov-config=ml/pyproject.toml` |
| Backend (FastAPI) | **215** | **83.6%** overall · **services 97.6%** of lines (89.7% of branches) | `docker compose exec backend pytest -q --cov` |
| Frontend (React) | **145** | **96.0%** of lines · 88.3% of branches | `docker compose exec frontend npm run coverage` |

Target from the specification: at least 60% on the backend services. **Met: 97.6%.**

## Backend coverage by area

| Area | Lines | Branches | Notes |
|---|---:|---:|---|
| `api/` (endpoints) | 99.1% | 95.0% | |
| `services/` (business logic) | 97.6% | 89.7% | scoring, cases, audit, settings, monitoring, model store / promotion |
| `db/`, `schemas/` | 100% | 100% | |
| `core/` (config, security) | 97.4% | 83.3% | |
| `scripts/` (command-line tools) | 49.3% | 38.0% | see below |

**Why the scripts are lower:** each script is a thin command-line wrapper (argument parsing, printing) around functions that *are* tested (`replay_day`, `score_pending`, `compute_day`, `export_frame`, `open_cases`, `register`, the loader). The wrappers themselves were exercised on the real data many times and their results checked: the replay of 28 days (re-run = no change), the weekly comparison with the offline test evaluation (identical for all 4 weeks), the skew check against PostgreSQL (`RESULT: PASS`), the retraining export (82,233 rows). `check_skew_db` is itself a verification tool. `demo_check` is tested both ways: a ready database passes, and each broken precondition (missing artifacts or drift reference, wrong password, unknown user, demo day already loaded, stale model in the API, frontend down) fails with a hint.

## What the tests cover (highlights)

- **No leakage, one feature pipeline:** batch vs online features identical; features never use the future; forbidden columns refused; skew check against PostgreSQL.
- **Business metrics:** cost model, capacity-constrained threshold, PR-AUC, bootstrap of a PR-AUC difference.
- **Scoring:** calibrated probability, expected-loss alert rule, threshold from settings, risk bands, anomaly flag, SHAP factors that add up to the model's score.
- **Security:** JWT and refresh-token rotation with reuse detection, role checks on every admin endpoint, the audit log refusing UPDATE/DELETE (database trigger), no ground truth in any response, CSV formula-injection guard.
- **Review workflow:** claim/assign/decide transitions, row locking rules, escalation chain, timeline.
- **Monitoring:** PSI/KS, daily metrics equal to the offline cost model, exact sums over periods, replay idempotence, parallel scoring path.
- **Model management:** activation with hot reload (proven by `/score` returning the new version), a broken artifact changes nothing, rollback, startup without a model or without a database.
- **Frontend:** single-flight token refresh, route guards, URL ↔ filter mapping, the waterfall data and the drawn SVG, decision rules, settings validation identical to the API, CSV download, menus per role.

## Gaps filled while measuring

Coverage was measured first, then tests were added only where an untested path mattered:

- Backend: API startup (model loaded / no model / database down), drift-reference loading and the "no reference" path, the Parquet period reader used by the loader and the replay.
- ML: the plain-language helpers behind analyst sentences (durations, rarity), and a check that **every explanation template that can receive a missing value produces a readable sentence** (no "nan").
- Frontend: login form and logout (token cleared, cache emptied), the "no retry on 4xx" rule, the navigation per role, the debounced amount filter (one request, not one per keystroke), the CSV export button, every lazily loaded page.

## Not covered by automated tests

- The XGBoost comparison model on GPU (`fraudml/models/xgb.py`, 52%): used once in the model selection, not in production.
- Visual layout, dark mode and small screens: checked by hand in the browser (click-throughs at the end of each frontend step).
- End-to-end browser tests (Playwright): bonus item, not implemented.
