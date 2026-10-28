import datetime as dt

from sqlalchemy.orm import Session

from app.db.models import Decision
from app.scripts.export_training_data import export_frame
from app.services.cases import open_case

DAY = dt.date(2018, 5, 1)
NOW = dt.datetime.now(dt.UTC)


def test_label_rules_and_champion_columns(db_session: Session, make_prediction, make_user):
    analyst = make_user("analyst")
    when = dt.datetime(2018, 5, 1, 12)

    def txn(ground_truth, revealed, decisions=(), amount=100.0):
        p = make_prediction(amount=amount, when=when, ground_truth=ground_truth)
        p.transaction.features = {"amt_log": amount / 100, "card6": "debit"}
        if revealed:
            p.transaction.ground_truth_revealed_at = NOW
        if decisions:
            case = open_case(db_session, p)
            for d in decisions:
                db_session.add(Decision(case_id=case.id, analyst_id=analyst.id, decision=d))
        db_session.flush()
        return p

    revealed = txn(True, revealed=True, decisions=["false_positive"])  # truth wins
    decided = txn(True, revealed=False, decisions=["escalate", "false_positive"])  # final decision
    unlabelled = txn(False, revealed=False)
    no_features = make_prediction(when=when)  # never scored with features -> not exported
    assert no_features.transaction.features is None

    df = export_frame(db_session, DAY, DAY + dt.timedelta(days=1), revealed.model_version_id)
    by_id = df.set_index("TransactionID")
    assert set(by_id.index) == {
        revealed.transaction.external_id,
        decided.transaction.external_id,
        unlabelled.transaction.external_id,
    }
    r = by_id.loc[revealed.transaction.external_id]
    assert (r["label"], r["label_source"]) == (1, "ground_truth")
    d = by_id.loc[decided.transaction.external_id]
    assert (d["label"], d["label_source"]) == (0, "decision")
    u = by_id.loc[unlabelled.transaction.external_id]
    assert u["label_source"] == "none" and str(u["label"]) == "<NA>"
    # Features are expanded into columns; the champion's stored score and alert are kept.
    assert r["amt_log"] == 1.0 and r["card6"] == "debit"
    assert r["champion_score"] == revealed.fraud_score and bool(r["champion_alert"]) is True


def test_empty_range(db_session: Session):
    assert export_frame(db_session, DAY, DAY + dt.timedelta(days=1), None).empty
