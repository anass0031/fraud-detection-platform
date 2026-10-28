"""Review queue: opening cases from predictions, listing / reading them, and the workflow
(claim, assign, decide) with its audit timeline.

A case is opened for every prediction that is an alert (reason `model_score`, tier 1) or an
"unusual pattern" flag (reason `anomaly`, tier 2). The anomaly flag is only set on rows the
main model did not alert on, so one prediction has at most one reason.
Priority inside a tier = expected loss (calibrated probability x amount).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, contains_eager, joinedload

from app.db.models import AuditLog, Case, Decision, Prediction, Transaction, User
from app.services import audit

TIER_MODEL_ALERT = 1
TIER_UNUSUAL = 2


def case_reason(pred: Prediction) -> tuple[str, int] | None:
    """(reason, priority tier) of a prediction, or None if it does not open a case."""
    if pred.is_alert:
        return "model_score", TIER_MODEL_ALERT
    if pred.is_unusual:
        return "anomaly", TIER_UNUSUAL
    return None


def open_case(db: Session, pred: Prediction) -> Case | None:
    """Open the case of a prediction if it needs one and the transaction has none yet.

    Idempotent (one case per transaction). Returns the new case, or None. The audit row
    (`case.create`, system action) is written in the caller's transaction.
    """
    found = case_reason(pred)
    if found is None:
        return None
    reason, tier = found
    case_id = db.scalar(
        insert(Case)
        .values(
            transaction_id=pred.transaction_id,
            prediction_id=pred.id,
            reason=reason,
            priority_tier=tier,
            priority_score=pred.expected_loss,
        )
        .on_conflict_do_nothing(index_elements=[Case.transaction_id])
        .returning(Case.id)
    )
    if case_id is None:
        return None
    audit.record(
        db,
        "case.create",
        entity_type="case",
        entity_id=case_id,
        after={
            "reason": reason,
            "priority_tier": tier,
            "priority_score": round(pred.expected_loss, 4),
            "status": "new",
            "prediction_id": pred.id,
        },
    )
    return db.get(Case, case_id)


# ---------------------------------------------------------------- listing

HISTORY_SIZE = 10


class InvalidFilter(ValueError):
    pass


@dataclass
class CaseFilters:
    """Queue filters (shared by `GET /cases` and the CSV export). Empty = no filter."""

    status: list[str] = field(default_factory=list)
    risk_level: list[str] = field(default_factory=list)
    reason: str | None = None
    assignee: str | None = None  # "me", "unassigned" or a user id
    amount_min: float | None = None
    amount_max: float | None = None
    date_from: dt.date | None = None  # transaction date, inclusive
    date_to: dt.date | None = None  # transaction date, inclusive


def _base_query() -> Select:
    return (
        select(Case)
        .join(Case.transaction)
        .join(Case.prediction)
        .options(
            contains_eager(Case.transaction),
            contains_eager(Case.prediction),
            joinedload(Case.assignee),
        )
    )


def apply_filters(query: Select, f: CaseFilters, user: User) -> Select:
    if f.status:
        query = query.where(Case.status.in_(f.status))
    if f.risk_level:
        query = query.where(Prediction.risk_level.in_(f.risk_level))
    if f.reason:
        query = query.where(Case.reason == f.reason)
    if f.assignee == "me":
        query = query.where(Case.assigned_to == user.id)
    elif f.assignee == "unassigned":
        query = query.where(Case.assigned_to.is_(None))
    elif f.assignee is not None:
        if not f.assignee.isdigit():
            raise InvalidFilter("assignee must be 'me', 'unassigned' or a user id")
        query = query.where(Case.assigned_to == int(f.assignee))
    if f.amount_min is not None:
        query = query.where(Transaction.amount >= f.amount_min)
    if f.amount_max is not None:
        query = query.where(Transaction.amount <= f.amount_max)
    if f.date_from is not None:
        query = query.where(Transaction.transaction_time >= f.date_from)
    if f.date_to is not None:
        query = query.where(Transaction.transaction_time < f.date_to + dt.timedelta(days=1))
    return query


def apply_sort(query: Select, sort: str, order: str) -> Select:
    """Sort key + case id as tie-breaker (equal expected losses exist), so pages are stable.

    `priority` desc = the queue order: model alerts first, then unusual patterns, each by
    expected loss (highest first).
    """
    desc = order == "desc"
    if sort == "priority":
        keys = (
            [Case.priority_tier.asc(), Case.priority_score.desc()]
            if desc
            else [Case.priority_tier.desc(), Case.priority_score.asc()]
        )
    else:
        col = {
            "amount": Transaction.amount,
            "transaction_time": Transaction.transaction_time,
            "created_at": Case.created_at,
        }[sort]
        keys = [col.desc() if desc else col.asc()]
    return query.order_by(*keys, Case.id.asc())


def list_cases(
    db: Session,
    f: CaseFilters,
    user: User,
    sort: str = "priority",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Case], int]:
    """One page of cases and the total number matching the filters."""
    filtered = apply_filters(_base_query(), f, user)
    total = db.scalar(select(func.count()).select_from(filtered.order_by(None).subquery()))
    page = db.scalars(apply_sort(filtered, sort, order).limit(limit).offset(offset)).unique()
    return list(page), int(total or 0)


def final_decisions(db: Session, case_ids: Sequence[int]) -> dict[int, str]:
    """Latest confirmed_fraud / false_positive decision per case (escalations ignored)."""
    if not case_ids:
        return {}
    rows = db.execute(
        select(Decision.case_id, Decision.decision)
        .where(Decision.case_id.in_(case_ids), Decision.decision != "escalate")
        .order_by(Decision.id)
    )
    return {case_id: decision for case_id, decision in rows}  # later rows win


def all_cases(db: Session, f: CaseFilters, user: User) -> list[Case]:
    """Every case matching the filters, in queue order (for the CSV export)."""
    query = apply_sort(apply_filters(_base_query(), f, user), "priority", "desc")
    return list(db.scalars(query).unique())


def final_decision_rows(db: Session, case_ids: Sequence[int]) -> dict[int, Decision]:
    """Latest confirmed_fraud / false_positive Decision per case, with its analyst loaded."""
    result: dict[int, Decision] = {}
    for start in range(0, len(case_ids), 5000):  # keep the IN list a reasonable size
        rows = db.scalars(
            select(Decision)
            .options(joinedload(Decision.analyst))
            .where(
                Decision.case_id.in_(case_ids[start : start + 5000]),
                Decision.decision != "escalate",
            )
            .order_by(Decision.id)
        )
        result.update({d.case_id: d for d in rows})  # later rows win
    return result


def get_case(db: Session, case_id: int) -> Case | None:
    return db.scalars(_base_query().where(Case.id == case_id)).unique().one_or_none()


def card_history(db: Session, txn: Transaction, size: int = HISTORY_SIZE) -> list:
    """Up to `size` earlier transactions of the same customer key, newest first, with a flag
    telling whether each one had a case. Same "strictly earlier" rule as the features."""
    if txn.customer_key is None:
        return []
    had_case = select(Case.id).where(Case.transaction_id == Transaction.id).exists()
    return list(
        db.execute(
            select(
                Transaction.external_id.label("transaction_id"),
                Transaction.transaction_time,
                Transaction.amount,
                Transaction.product_code,
                had_case.label("had_case"),
            )
            .where(
                Transaction.customer_key == txn.customer_key,
                Transaction.transaction_dt < txn.transaction_dt,
            )
            .order_by(Transaction.transaction_dt.desc(), Transaction.external_id.desc())
            .limit(size)
        )
    )


# ---------------------------------------------------------------- workflow


class CaseNotFound(LookupError):
    pass


class CaseForbidden(PermissionError):
    """The user's role or assignment does not allow this action (HTTP 403)."""


class CaseConflict(RuntimeError):
    """The case is not in a state that allows this action (HTTP 409)."""


class InvalidAssignee(ValueError):
    """The target user cannot receive this case (HTTP 422)."""


class InvalidDecision(ValueError):
    """The decision request is incomplete (HTTP 422)."""


def lock_case(db: Session, case_id: int) -> Case:
    """Read the case with a row lock (SELECT ... FOR UPDATE) until the commit, so two users
    acting on the same case at the same time are serialised; fresh values, not cached ones."""
    case = db.scalar(
        select(Case)
        .where(Case.id == case_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if case is None:
        raise CaseNotFound(f"case {case_id} not found")
    return case


def _state(case: Case) -> dict:
    return {"status": case.status, "assigned_to": case.assigned_to}


def _change(
    db: Session,
    case: Case,
    action: str,
    actor: User,
    *,
    status: str,
    assigned_to: int | None,
    request=None,
    extra: dict | None = None,
) -> Case:
    before = _state(case)
    case.status = status
    case.assigned_to = assigned_to
    case.updated_at = dt.datetime.now(dt.UTC)
    if status == "closed":
        case.closed_at = case.updated_at
    db.flush()
    audit.record(
        db,
        action,
        actor=actor,
        entity_type="case",
        entity_id=case.id,
        before=before,
        after={**_state(case), **(extra or {})},
        request=request,
    )
    return case


def claim(db: Session, case_id: int, user: User, request=None) -> Case:
    """Take a case for yourself.

    new -> assigned (analyst or admin); escalated (unassigned) -> escalated, admins only.
    Claiming a case you already hold changes nothing. Closed or held by someone else -> 409.
    """
    case = lock_case(db, case_id)
    if case.status == "closed":
        raise CaseConflict("Case is closed")
    if case.status == "escalated" and user.role != "admin":
        raise CaseForbidden("Escalated cases can only be handled by an admin")
    if case.assigned_to == user.id:
        return case
    if case.assigned_to is not None:
        raise CaseConflict("Case is already assigned to someone else")
    new_status = "escalated" if case.status == "escalated" else "assigned"
    return _change(
        db, case, "case.claim", user, status=new_status, assigned_to=user.id, request=request
    )


def assign(db: Session, case_id: int, admin: User, user_id: int, request=None) -> Case:
    """Admin: (re)assign an open case to an active user. Escalated cases stay escalated and
    can only go to an admin. Assigning to the current assignee changes nothing."""
    if admin.role != "admin":
        raise CaseForbidden("Only an admin can assign cases")
    case = lock_case(db, case_id)
    if case.status == "closed":
        raise CaseConflict("Case is closed")
    target = db.get(User, user_id)
    if target is None or not target.is_active:
        raise InvalidAssignee("Assignee must be an active user")
    if case.status == "escalated" and target.role != "admin":
        raise InvalidAssignee("Escalated cases can only be assigned to an admin")
    if case.assigned_to == target.id:
        return case
    new_status = "escalated" if case.status == "escalated" else "assigned"
    return _change(
        db, case, "case.assign", admin, status=new_status, assigned_to=target.id, request=request
    )


FINAL_DECISIONS = ("confirmed_fraud", "false_positive")


def decide(
    db: Session,
    case_id: int,
    user: User,
    decision: str,
    comment: str | None,
    request=None,
) -> Case:
    """Record the assignee's decision.

    confirmed_fraud / false_positive -> `closed` (the decision is the case's label).
    escalate (analysts, comment required) -> `escalated` and unassigned, for an admin to
    claim and close. Only the current assignee can decide; closed cases -> 409.
    """
    case = lock_case(db, case_id)
    if case.status == "closed":
        raise CaseConflict("Case is closed")
    if case.assigned_to != user.id:
        raise CaseForbidden("Only the analyst assigned to the case can decide (claim it first)")
    if decision == "escalate":
        if case.status == "escalated":
            raise CaseConflict("Case is already escalated: confirm fraud or false positive")
        if user.role == "admin":
            raise CaseForbidden("Admins decide escalated cases themselves")
        if not comment:
            raise InvalidDecision("A comment is required to escalate")
        new_status, new_assignee = "escalated", None
    elif decision in FINAL_DECISIONS:
        new_status, new_assignee = "closed", case.assigned_to
    else:
        raise InvalidDecision(f"Unknown decision {decision!r}")

    row = Decision(case_id=case.id, analyst_id=user.id, decision=decision, comment=comment)
    db.add(row)
    db.flush()
    # The comment text is kept in `decisions`; the audit row records only its length.
    return _change(
        db,
        case,
        "case.decide",
        user,
        status=new_status,
        assigned_to=new_assignee,
        request=request,
        extra={
            "decision": decision,
            "decision_id": row.id,
            "comment_length": len(comment or ""),
        },
    )


def timeline(db: Session, case_id: int) -> list[tuple[AuditLog, User | None, str | None]]:
    """Audit rows of a case, oldest first, with the actor (None = system) and, for decisions,
    the comment from the `decisions` table."""
    rows = db.execute(
        select(AuditLog, User)
        .outerjoin(User, AuditLog.actor_id == User.id)
        .where(AuditLog.entity_type == "case", AuditLog.entity_id == str(case_id))
        .order_by(AuditLog.id)
    ).all()
    comments = dict(
        db.execute(select(Decision.id, Decision.comment).where(Decision.case_id == case_id)).all()
    )
    out = []
    for log, actor in rows:
        decision_id = (log.after or {}).get("decision_id")
        out.append((log, actor, comments.get(decision_id) if decision_id else None))
    return out
