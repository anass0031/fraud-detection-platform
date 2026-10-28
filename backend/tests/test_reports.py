import csv
import io

import pytest
from sqlalchemy.orm import Session

from app.api.v1.reports import COLUMNS, cell
from app.db.models import Decision
from app.services.cases import open_case
from tests.conftest import auth_header

URL = "/api/v1/reports/cases.csv"


def rows(resp) -> list[dict]:
    assert resp.status_code == 200, resp.text
    return list(csv.DictReader(io.StringIO(resp.text)))


@pytest.fixture
def cases(db_session: Session, make_prediction):
    return [
        open_case(db_session, make_prediction(amount=200.0, score=0.4, ground_truth=True)),
        open_case(db_session, make_prediction(amount=50.0, score=0.5, ground_truth=False)),
        open_case(db_session, make_prediction(amount=900.0, alert=False, unusual=True)),
    ]


def test_csv_has_header_and_rows_in_queue_order(api, make_user, cases):
    resp = api.get(URL, headers=auth_header(api, make_user("analyst")))
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.text.splitlines()[0] == ",".join(COLUMNS)
    got = rows(resp)
    assert [int(r["case_id"]) for r in got] == [c.id for c in cases]  # EL 80, 25, then unusual
    first = got[0]
    assert (first["amount"], first["expected_loss"], first["fraud_score"]) == (
        "200.00",
        "80.00",
        "0.4000",
    )
    assert (first["status"], first["assignee"], first["final_decision"]) == ("new", "", "")


def test_csv_uses_queue_filters(api, make_user, cases):
    h = auth_header(api, make_user("analyst"))
    assert [
        int(r["case_id"]) for r in rows(api.get(URL, headers=h, params={"reason": "anomaly"}))
    ] == [cases[2].id]
    assert rows(api.get(URL, headers=h, params={"status": "closed"})) == []
    assert api.get(URL, headers=h, params={"assignee": "x"}).status_code == 422


def test_csv_shows_final_decision(api, db_session, make_user, cases):
    analyst = make_user("analyst", email="ana@example.com")
    case = cases[1]
    case.status, case.assigned_to = "closed", analyst.id
    db_session.add_all(
        [
            Decision(case_id=case.id, analyst_id=analyst.id, decision="escalate", comment="?"),
            Decision(case_id=case.id, analyst_id=analyst.id, decision="confirmed_fraud"),
        ]
    )
    db_session.flush()
    got = {int(r["case_id"]): r for r in rows(api.get(URL, headers=auth_header(api, analyst)))}
    r = got[case.id]
    assert (r["status"], r["assignee"], r["final_decision"], r["decided_by"]) == (
        "closed",
        "ana@example.com",
        "confirmed_fraud",
        "ana@example.com",
    )
    assert r["decided_at"] != ""


def test_csv_never_contains_ground_truth(api, make_user, cases):
    resp = api.get(URL, headers=auth_header(api, make_user("analyst")))
    assert "ground_truth" not in resp.text and "isFraud" not in resp.text
    assert all(set(r) == set(COLUMNS) for r in rows(resp))


def test_csv_requires_login(api):
    assert api.get(URL).status_code == 401


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=HYPERLINK(1)", "'=HYPERLINK(1)"),
        ("+1", "'+1"),
        ("-1", "'-1"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("visa", "visa"),
        (None, ""),
        (3, 3),
    ],
)
def test_formula_injection_guard(value, expected):
    assert cell(value) == expected


def test_product_code_with_formula_is_escaped(api, db_session, make_user, make_prediction):
    open_case(db_session, make_prediction(product="=cmd"))
    (r,) = rows(api.get(URL, headers=auth_header(api, make_user("analyst"))))
    assert r["product_code"] == "'=cmd"
