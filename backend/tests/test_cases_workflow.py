import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Case
from app.services.cases import open_case
from tests.conftest import auth_header

URL = "/api/v1/cases"


@pytest.fixture
def case(db_session: Session, make_prediction) -> Case:
    return open_case(db_session, make_prediction())


@pytest.fixture
def users(make_user):
    return {
        "analyst": make_user("analyst"),
        "analyst2": make_user("analyst"),
        "admin": make_user("admin"),
        "admin2": make_user("admin"),
    }


def post(api, user, path, **kw):
    return api.post(f"{URL}/{path}", headers=auth_header(api, user), **kw)


def audit_rows(db: Session, case: Case, action: str) -> list[AuditLog]:
    return list(
        db.scalars(
            select(AuditLog)
            .where(AuditLog.entity_id == str(case.id), AuditLog.action == action)
            .order_by(AuditLog.id)
        )
    )


def escalate(db: Session, case: Case) -> None:
    """Put a case in the escalated, unassigned state (the decision endpoint comes later)."""
    case.status, case.assigned_to = "escalated", None
    db.flush()


# ------------------------------------------------------------------ claim


def test_claim_new_case(api, db_session, case, users):
    resp = post(api, users["analyst"], f"{case.id}/claim")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "assigned"
    assert body["assignee"]["id"] == users["analyst"].id

    (row,) = audit_rows(db_session, case, "case.claim")
    assert row.actor_id == users["analyst"].id
    assert row.before == {"status": "new", "assigned_to": None}
    assert row.after == {"status": "assigned", "assigned_to": users["analyst"].id}
    assert row.ip_address is not None


def test_second_claim_by_someone_else_is_a_conflict(api, db_session, case, users):
    assert post(api, users["analyst"], f"{case.id}/claim").status_code == 200
    resp = post(api, users["analyst2"], f"{case.id}/claim")
    assert resp.status_code == 409
    db_session.refresh(case)
    assert case.assigned_to == users["analyst"].id
    assert len(audit_rows(db_session, case, "case.claim")) == 1


def test_claiming_your_own_case_again_changes_nothing(api, db_session, case, users):
    assert post(api, users["analyst"], f"{case.id}/claim").status_code == 200
    assert post(api, users["analyst"], f"{case.id}/claim").status_code == 200
    assert len(audit_rows(db_session, case, "case.claim")) == 1


def test_claim_unknown_case(api, users):
    assert post(api, users["analyst"], "999999/claim").status_code == 404


def test_claim_requires_login(api, case):
    assert api.post(f"{URL}/{case.id}/claim").status_code == 401


def test_closed_case_cannot_be_claimed(api, db_session, case, users):
    case.status = "closed"
    db_session.flush()
    assert post(api, users["analyst"], f"{case.id}/claim").status_code == 409


def test_escalated_case_admin_only(api, db_session, case, users):
    escalate(db_session, case)
    assert post(api, users["analyst"], f"{case.id}/claim").status_code == 403
    resp = post(api, users["admin"], f"{case.id}/claim")
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"  # stays escalated, now with an admin
    assert resp.json()["assignee"]["id"] == users["admin"].id


# ------------------------------------------------------------------ assign


def test_admin_assigns_and_reassigns(api, db_session, case, users):
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "assigned"
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst2"].id})
    assert resp.json()["assignee"]["id"] == users["analyst2"].id

    first, second = audit_rows(db_session, case, "case.assign")
    assert first.actor_id == users["admin"].id
    assert second.before["assigned_to"] == users["analyst"].id
    assert second.after["assigned_to"] == users["analyst2"].id


def test_assign_to_current_assignee_changes_nothing(api, db_session, case, users):
    body = {"user_id": users["analyst"].id}
    post(api, users["admin"], f"{case.id}/assign", json=body)
    assert post(api, users["admin"], f"{case.id}/assign", json=body).status_code == 200
    assert len(audit_rows(db_session, case, "case.assign")) == 1


def test_analyst_cannot_assign(api, case, users):
    resp = post(api, users["analyst"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    assert resp.status_code == 403


def test_assign_to_unknown_or_inactive_user(api, make_user, case, users):
    inactive = make_user("analyst", active=False)
    for user_id in (999999, inactive.id):
        resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": user_id})
        assert resp.status_code == 422


def test_assign_request_is_validated(api, case, users):
    assert post(api, users["admin"], f"{case.id}/assign", json={}).status_code == 422
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": 1, "x": 2})
    assert resp.status_code == 422


def test_escalated_case_can_only_go_to_an_admin(api, db_session, case, users):
    escalate(db_session, case)
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    assert resp.status_code == 422
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["admin2"].id})
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"


def test_closed_case_cannot_be_assigned(api, db_session, case, users):
    case.status = "closed"
    db_session.flush()
    resp = post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    assert resp.status_code == 409


def test_assigned_case_appears_in_my_cases(api, case, users):
    post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    resp = api.get(URL, headers=auth_header(api, users["analyst"]), params={"assignee": "me"})
    assert [c["id"] for c in resp.json()["items"]] == [case.id]


# ------------------------------------------------------------------ decisions


def claimed(api, case, user):
    assert post(api, user, f"{case.id}/claim").status_code == 200


@pytest.mark.parametrize("decision", ["confirmed_fraud", "false_positive"])
def test_final_decision_closes_the_case(api, db_session, case, users, decision):
    claimed(api, case, users["analyst"])
    resp = post(api, users["analyst"], f"{case.id}/decision", json={"decision": decision})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "closed"
    assert body["closed_at"] is not None
    assert body["final_decision"] == decision
    assert body["assignee"]["id"] == users["analyst"].id  # who closed it stays visible

    (row,) = audit_rows(db_session, case, "case.decide")
    assert row.before == {"status": "assigned", "assigned_to": users["analyst"].id}
    assert row.after["status"] == "closed"
    assert row.after["decision"] == decision
    assert row.after["comment_length"] == 0


def test_closed_case_accepts_no_more_decisions(api, case, users):
    claimed(api, case, users["analyst"])
    post(api, users["analyst"], f"{case.id}/decision", json={"decision": "false_positive"})
    resp = post(api, users["analyst"], f"{case.id}/decision", json={"decision": "confirmed_fraud"})
    assert resp.status_code == 409


def test_only_the_assignee_decides(api, case, users):
    body = {"decision": "confirmed_fraud"}
    # Unclaimed case: nobody can decide yet, not even an admin.
    assert post(api, users["admin"], f"{case.id}/decision", json=body).status_code == 403
    claimed(api, case, users["analyst"])
    assert post(api, users["analyst2"], f"{case.id}/decision", json=body).status_code == 403
    assert post(api, users["admin"], f"{case.id}/decision", json=body).status_code == 403


def test_escalation_chain(api, db_session, case, users):
    claimed(api, case, users["analyst"])
    resp = post(
        api,
        users["analyst"],
        f"{case.id}/decision",
        json={"decision": "escalate", "comment": "  Amount pattern unclear  "},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "escalated"
    assert resp.json()["assignee"] is None
    assert resp.json()["final_decision"] is None

    # Analysts can no longer take it; an admin claims and closes it.
    assert post(api, users["analyst2"], f"{case.id}/claim").status_code == 403
    claimed(api, case, users["admin"])
    resp = post(
        api, users["admin"], f"{case.id}/decision", json={"decision": "escalate", "comment": "x"}
    )
    assert resp.status_code == 409  # already escalated
    resp = post(api, users["admin"], f"{case.id}/decision", json={"decision": "confirmed_fraud"})
    assert resp.json()["status"] == "closed"
    assert resp.json()["final_decision"] == "confirmed_fraud"

    detail = api.get(f"{URL}/{case.id}", headers=auth_header(api, users["analyst"])).json()
    assert [(d["decision"], d["comment"]) for d in detail["decisions"]] == [
        ("escalate", "Amount pattern unclear"),
        ("confirmed_fraud", None),
    ]


def test_admin_cannot_escalate_a_case_they_hold(api, case, users):
    claimed(api, case, users["admin"])
    resp = post(
        api, users["admin"], f"{case.id}/decision", json={"decision": "escalate", "comment": "x"}
    )
    assert resp.status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        {"decision": "escalate"},
        {"decision": "escalate", "comment": "   "},
        {"decision": "fraud"},
        {"decision": "false_positive", "comment": "x" * 2001},
        {"decision": "false_positive", "label": True},
        {},
    ],
)
def test_decision_request_is_validated(api, case, users, body):
    claimed(api, case, users["analyst"])
    assert post(api, users["analyst"], f"{case.id}/decision", json=body).status_code == 422


def test_decision_on_unknown_case(api, users):
    resp = post(api, users["analyst"], "999999/decision", json={"decision": "false_positive"})
    assert resp.status_code == 404


# ------------------------------------------------------------------ timeline


def test_timeline_tells_the_story_in_order(api, case, users):
    post(api, users["admin"], f"{case.id}/assign", json={"user_id": users["analyst"].id})
    post(
        api,
        users["analyst"],
        f"{case.id}/decision",
        json={"decision": "escalate", "comment": "Need a second look"},
    )
    claimed(api, case, users["admin"])
    post(api, users["admin"], f"{case.id}/decision", json={"decision": "false_positive"})

    resp = api.get(f"{URL}/{case.id}/timeline", headers=auth_header(api, users["analyst2"]))
    assert resp.status_code == 200, resp.text
    events = resp.json()
    assert [e["action"] for e in events] == [
        "case.create",
        "case.assign",
        "case.decide",
        "case.claim",
        "case.decide",
    ]
    assert events[0]["actor"] is None  # opened by the scoring service
    assert events[1]["actor"]["id"] == users["admin"].id
    assert events[2]["comment"] == "Need a second look"
    assert events[2]["after"]["status"] == "escalated"
    assert events[4]["after"]["decision"] == "false_positive"
    assert events[4]["comment"] is None
    assert "ip_address" not in resp.text


def test_timeline_unknown_case(api, users):
    resp = api.get(f"{URL}/999999/timeline", headers=auth_header(api, users["analyst"]))
    assert resp.status_code == 404
