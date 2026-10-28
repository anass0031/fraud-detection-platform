import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Case
from app.scripts.open_cases import flagged_predictions, open_cases
from app.services.cases import TIER_MODEL_ALERT, TIER_UNUSUAL, open_case

DAY = (dt.date(2018, 4, 28), dt.date(2018, 4, 29))


def _count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


def test_alert_opens_a_tier_one_case(db_session: Session, make_prediction):
    pred = make_prediction(amount=200.0, score=0.4, alert=True)
    case = open_case(db_session, pred)
    assert case is not None
    assert case.reason == "model_score"
    assert case.priority_tier == TIER_MODEL_ALERT
    assert case.priority_score == 80.0  # expected loss = 0.4 x 200
    assert case.status == "new"
    assert case.assigned_to is None
    assert case.prediction_id == pred.id


def test_unusual_pattern_opens_a_tier_two_case(db_session: Session, make_prediction):
    case = open_case(db_session, make_prediction(alert=False, unusual=True))
    assert case is not None
    assert (case.reason, case.priority_tier) == ("anomaly", TIER_UNUSUAL)


def test_no_alert_no_case(db_session: Session, make_prediction):
    assert open_case(db_session, make_prediction(alert=False, unusual=False)) is None
    assert _count(db_session, Case) == 0


def test_one_case_per_transaction(db_session: Session, make_prediction):
    pred = make_prediction()
    assert open_case(db_session, pred) is not None
    assert open_case(db_session, pred) is None
    assert _count(db_session, Case) == 1


def test_case_creation_is_audited_as_system_action(db_session: Session, make_prediction):
    case = open_case(db_session, make_prediction(amount=50.0, score=0.5))
    row = db_session.scalar(select(AuditLog).where(AuditLog.action == "case.create"))
    assert row is not None
    assert row.actor_id is None
    assert (row.entity_type, row.entity_id) == ("case", str(case.id))
    assert row.after["reason"] == "model_score"
    assert row.after["priority_score"] == 25.0


def test_backfill_opens_missing_cases_once(db_session: Session, make_prediction):
    alert = make_prediction()
    make_prediction(alert=False, unusual=True)
    make_prediction(alert=False, unusual=False)
    make_prediction(when=dt.datetime(2018, 4, 29, 0, 0))  # outside [from, to)
    open_case(db_session, alert)  # already has a case

    model_id = alert.model_version_id
    assert len(flagged_predictions(db_session, model_id, *DAY)) == 1
    opened = open_cases(db_session, model_id, *DAY)
    assert dict(opened) == {"anomaly": 1}
    assert open_cases(db_session, model_id, *DAY) == {}
    assert _count(db_session, Case) == 2
