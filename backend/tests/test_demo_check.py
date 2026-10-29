import datetime as dt

import httpx

from app.db.models import Case, MetricSnapshot, ModelVersion
from app.scripts.demo_check import database_checks as run_db
from app.scripts.demo_check import demo_credentials, http_checks, print_report
from tests.conftest import TEST_PASSWORD

DAY = dt.date(2018, 5, 26)


def by_name(checks):
    return {c.name: c for c in checks}


def ready_demo_db(db, make_user, make_prediction, tmp_path):
    """Everything the demo needs; returns the credentials and the artifacts folder."""
    users = [make_user("admin"), make_user("analyst"), make_user("analyst")]
    pred = make_prediction(when=dt.datetime(2018, 5, 25, 10))
    active = pred.model_version
    active.anomaly_version = "iforest-x"
    active.artifact_path = "models/test-model"
    db.add(
        ModelVersion(
            version="cand",
            algorithm="lightgbm",
            artifact_path="models/cand",
            feature_list=["x"],
            policy="expected_loss",
            threshold=20.0,
            status="candidate",
        )
    )
    db.add(
        Case(
            transaction_id=pred.transaction_id,
            prediction_id=pred.id,
            reason="model_score",
            priority_tier=1,
            priority_score=30.0,
        )
    )
    zero = dict.fromkeys(["n", "n_fraud", "alerts", "tp", "fp", "fn", "unusual_flags"], 0)
    money = dict.fromkeys(
        ["fraud_value", "fraud_value_caught", "total_cost", "no_model_cost", "net_savings"], 0.0
    )
    db.add(MetricSnapshot(model_version_id=active.id, day=DAY - dt.timedelta(1), **zero, **money))
    db.flush()
    for d in ("models/test-model", "models/iforest-x", "monitoring"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "monitoring" / "drift_reference-test-model.json").write_text("{}")
    return {u.email: TEST_PASSWORD for u in users}


def test_ready_demo_db_passes(db_session, make_user, make_prediction, tmp_path):
    creds = ready_demo_db(db_session, make_user, make_prediction, tmp_path)
    checks = run_db(db_session, creds, DAY, tmp_path)
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert len(checks) == 7 + len(creds)


def test_each_broken_precondition_fails_clearly(db_session, make_user, make_prediction, tmp_path):
    creds = ready_demo_db(db_session, make_user, make_prediction, tmp_path)
    first = next(iter(creds))
    creds[first] = "wrong-password"
    creds["nobody@example.com"] = TEST_PASSWORD
    (tmp_path / "models" / "iforest-x").rmdir()
    (tmp_path / "monitoring" / "drift_reference-test-model.json").unlink()
    make_prediction(when=dt.datetime(2018, 5, 26, 9))  # demo day already (partly) replayed

    c = by_name(run_db(db_session, creds, DAY, tmp_path))
    assert not c["active model"].ok and "iforest-x" in c["active model"].detail
    assert not c["drift reference"].ok
    assert not c[f"user {first}"].ok and "does not match" in c[f"user {first}"].detail
    assert "missing" in c["user nobody@example.com"].detail
    day = c[f"demo day {DAY} not replayed"]
    assert not day.ok and "restore the snapshot" in day.detail
    assert c["candidate model"].ok and c["open cases"].ok and c["monitoring"].ok


def test_empty_database_fails(db_session):
    c = by_name(run_db(db_session, {"a@x.io": "pw"}, DAY))
    assert c["database"].ok
    assert not c["active model"].ok and not c["candidate model"].ok
    assert not c["open cases"].ok and not c["user a@x.io"].ok
    assert "monitoring" not in c  # needs an active model


def fake_get(status=200, body=None, fail=()):
    def get(url, headers=None, timeout=None):
        if any(part in url for part in fail):
            raise httpx.ConnectError("refused")
        if url.endswith("/api/v1/health"):
            return httpx.Response(status, json=body or {"status": "ok", "model": {"version": "m1"}})
        return httpx.Response(status, text="<html>")

    return get


def test_http_checks_pass_and_fail():
    ok = http_checks("http://api", "http://web", "m1", fake_get())
    assert all(c.ok for c in ok) and len(ok) == 3

    stale = by_name(http_checks("http://api", "http://web", "m2", fake_get()))
    assert not stale["API health"].ok and "restart the backend" in stale["API health"].detail

    down = by_name(http_checks("http://api", "http://web", "m1", fake_get(fail=("web",))))
    assert down["API health"].ok and not down["frontend"].ok


def test_credentials_and_report(capsys):
    env = {
        "SEED_ADMIN_EMAIL": " Boss@X.io ",
        "SEED_ADMIN_PASSWORD": "a",
        "SEED_ANALYST_PASSWORD": "b",
    }
    creds = demo_credentials(env)
    assert creds["boss@x.io"] == "a" and creds["analyst1@fraud.local"] == "b" and len(creds) == 3

    checks = http_checks("http://api", "http://web", "m1", fake_get(fail=("web",)))
    assert print_report(checks) is False
    out = capsys.readouterr().out
    assert "FAIL  frontend" in out and "RESULT: FAIL (2 of 3)" in out
