import datetime as dt

import pytest
from sqlalchemy.orm import Session

from app.db.models import Decision
from app.services.cases import open_case
from tests.conftest import auth_header

URL = "/api/v1/cases"


@pytest.fixture
def queue(db_session: Session, make_prediction):
    """Five cases: 4 model alerts (expected loss 80, 45, 45, 10) + 1 unusual pattern (90)."""

    def case(**kw):
        return open_case(db_session, make_prediction(**kw))

    return {
        "el80": case(amount=200.0, score=0.4, risk="high", customer="a"),
        "el45a": case(amount=150.0, score=0.3, risk="high", customer="b", product="C"),
        "el45b": case(amount=150.0, score=0.3, risk="high", customer="c"),
        "el10": case(amount=20.0, score=0.5, risk="critical", when=dt.datetime(2018, 4, 29, 8, 0)),
        "unusual": case(amount=900.0, score=0.1, alert=False, unusual=True, risk="medium"),
    }


@pytest.fixture
def headers(api, make_user):
    return auth_header(api, make_user("analyst"))


def ids(resp) -> list[int]:
    assert resp.status_code == 200, resp.text
    return [item["id"] for item in resp.json()["items"]]


def test_requires_login(api):
    assert api.get(URL).status_code == 401


def test_default_order_is_tier_then_expected_loss_then_id(api, headers, queue):
    resp = api.get(URL, headers=headers)
    expected = ["el80", "el45a", "el45b", "el10", "unusual"]
    assert ids(resp) == [queue[k].id for k in expected]
    body = resp.json()
    assert body["total"] == 5
    first = body["items"][0]
    assert first["expected_loss"] == pytest.approx(80.0)
    assert first["reason"] == "model_score" and first["priority_tier"] == 1
    assert first["status"] == "new" and first["assignee"] is None
    assert first["final_decision"] is None


def test_ascending_priority_reverses_tiers(api, headers, queue):
    assert ids(api.get(URL, headers=headers, params={"order": "asc"}))[0] == queue["unusual"].id


def test_sort_by_amount(api, headers, queue):
    got = ids(api.get(URL, headers=headers, params={"sort": "amount"}))
    assert got[0] == queue["unusual"].id  # $900
    assert got[-1] == queue["el10"].id  # $20


def test_pagination_is_stable_and_total_counts_all(api, headers, queue):
    pages = [
        api.get(URL, headers=headers, params={"limit": 2, "offset": o}).json() for o in (0, 2, 4)
    ]
    assert [p["total"] for p in pages] == [5, 5, 5]
    seen = [item["id"] for p in pages for item in p["items"]]
    assert seen == ids(api.get(URL, headers=headers))


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"reason": "anomaly"}, ["unusual"]),
        ({"risk_level": ["critical", "medium"]}, ["el10", "unusual"]),
        ({"amount_min": 100, "amount_max": 200}, ["el80", "el45a", "el45b"]),
        ({"date_from": "2018-04-29"}, ["el10"]),
        ({"date_to": "2018-04-28"}, ["el80", "el45a", "el45b", "unusual"]),
        ({"status": ["assigned", "closed"]}, []),
        ({"status": "new", "reason": "model_score", "amount_min": 150}, ["el80", "el45a", "el45b"]),
    ],
)
def test_filters(api, headers, queue, params, expected):
    assert ids(api.get(URL, headers=headers, params=params)) == [queue[k].id for k in expected]


def test_assignee_filter(api, db_session, make_user, queue):
    me = make_user("analyst")
    other = make_user("analyst")
    queue["el80"].assigned_to, queue["el80"].status = me.id, "assigned"
    queue["el10"].assigned_to, queue["el10"].status = other.id, "assigned"
    db_session.flush()
    h = auth_header(api, me)
    assert ids(api.get(URL, headers=h, params={"assignee": "me"})) == [queue["el80"].id]
    assert ids(api.get(URL, headers=h, params={"assignee": str(other.id)})) == [queue["el10"].id]
    assert len(ids(api.get(URL, headers=h, params={"assignee": "unassigned"}))) == 3
    assert api.get(URL, headers=h, params={"assignee": "bob"}).status_code == 422


@pytest.mark.parametrize(
    "params",
    [{"status": "open"}, {"sort": "score"}, {"limit": 0}, {"limit": 201}, {"offset": -1}],
)
def test_invalid_parameters(api, headers, params):
    assert api.get(URL, headers=headers, params=params).status_code == 422


def test_detail(api, db_session, headers, make_prediction, queue):
    # Two earlier transactions of the same customer; the first one had a case.
    early = make_prediction(amount=5.0, customer="a", when=dt.datetime(2018, 4, 1))
    open_case(db_session, early)
    make_prediction(amount=7.0, alert=False, customer="a", when=dt.datetime(2018, 4, 2))
    make_prediction(amount=9.0, customer="a", when=dt.datetime(2018, 5, 1))  # later: excluded

    case = queue["el80"]
    resp = api.get(f"{URL}/{case.id}", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == case.id
    assert body["transaction"]["amount"] == 200.0
    assert body["transaction"]["product_code"] == "W"
    assert body["prediction"]["model_version"] == "test-model"
    assert body["prediction"]["fraud_score"] == pytest.approx(0.4)
    assert body["prediction"]["top_factors"] == []
    assert [(h["amount"], h["had_case"]) for h in body["card_history"]] == [
        (7.0, False),
        (5.0, True),
    ]
    assert body["decisions"] == []


def test_detail_shows_decisions_and_final_decision(api, db_session, make_user, queue):
    analyst, admin = make_user("analyst"), make_user("admin")
    case = queue["el45a"]
    db_session.add_all(
        [
            Decision(case_id=case.id, analyst_id=analyst.id, decision="escalate", comment="?"),
            Decision(case_id=case.id, analyst_id=admin.id, decision="false_positive"),
        ]
    )
    db_session.flush()
    h = auth_header(api, analyst)
    body = api.get(f"{URL}/{case.id}", headers=h).json()
    assert [d["decision"] for d in body["decisions"]] == ["escalate", "false_positive"]
    assert body["decisions"][0]["analyst"]["id"] == analyst.id
    assert body["final_decision"] == "false_positive"
    listed = {i["id"]: i for i in api.get(URL, headers=h).json()["items"]}
    assert listed[case.id]["final_decision"] == "false_positive"


def test_detail_not_found(api, headers):
    assert api.get(f"{URL}/999999", headers=headers).status_code == 404


def test_ground_truth_is_never_returned(api, db_session, headers, make_prediction):
    case = open_case(db_session, make_prediction(ground_truth=True))
    for resp in (api.get(URL, headers=headers), api.get(f"{URL}/{case.id}", headers=headers)):
        assert resp.status_code == 200
        assert "ground_truth" not in resp.text
        assert "isFraud" not in resp.text
