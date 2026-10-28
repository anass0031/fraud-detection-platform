"""Review queue: list and read cases (any logged-in user), claim (analyst or admin),
assign (admin), decide (the assignee) and the case timeline. Every change is audited in the
same database transaction."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, require_role
from app.db.models import Case, User
from app.db.session import get_db
from app.schemas.cases import (
    AssignRequest,
    CaseDetail,
    CaseListItem,
    CasePage,
    CaseReason,
    CaseSort,
    CaseStatus,
    DecisionOut,
    DecisionRequest,
    HistoryItem,
    PredictionOut,
    RiskLevel,
    TimelineEntry,
    TransactionFacts,
    UserRef,
)
from app.schemas.score import RiskFactor
from app.services import cases as svc

router = APIRouter(prefix="/cases", tags=["cases"])

Db = Annotated[Session, Depends(get_db)]
AdminUser = Annotated[User, Depends(require_role("admin"))]

# Workflow errors -> HTTP status.
ERRORS: dict[type[Exception], int] = {
    svc.CaseNotFound: status.HTTP_404_NOT_FOUND,
    svc.CaseForbidden: status.HTTP_403_FORBIDDEN,
    svc.CaseConflict: status.HTTP_409_CONFLICT,
    svc.InvalidAssignee: status.HTTP_422_UNPROCESSABLE_CONTENT,
    svc.InvalidDecision: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


def run_action(db: Session, action: Callable[[], Case]) -> CaseListItem:
    """Run a workflow action, commit it (with its audit row) and return the updated case."""
    try:
        case = action()
    except tuple(ERRORS) as exc:
        db.rollback()
        raise HTTPException(ERRORS[type(exc)], str(exc)) from exc
    db.commit()
    fresh = svc.get_case(db, case.id)
    if fresh is None:  # deleted in between (not possible through the API)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    return CaseListItem(**list_item(fresh, svc.final_decisions(db, [fresh.id]).get(fresh.id)))


def case_filters(
    status: Annotated[list[CaseStatus] | None, Query(description="Repeat for several")] = None,
    risk_level: Annotated[list[RiskLevel] | None, Query()] = None,
    reason: CaseReason | None = None,
    assignee: Annotated[
        str | None, Query(description="'me', 'unassigned' or a user id", max_length=16)
    ] = None,
    amount_min: Annotated[float | None, Query(ge=0)] = None,
    amount_max: Annotated[float | None, Query(ge=0)] = None,
    date_from: Annotated[dt.date | None, Query(description="Transaction date, inclusive")] = None,
    date_to: Annotated[dt.date | None, Query(description="Transaction date, inclusive")] = None,
) -> svc.CaseFilters:
    """Query parameters shared by the queue and the CSV export."""
    return svc.CaseFilters(
        status=list(status or []),
        risk_level=list(risk_level or []),
        reason=reason,
        assignee=assignee,
        amount_min=amount_min,
        amount_max=amount_max,
        date_from=date_from,
        date_to=date_to,
    )


Filters = Annotated[svc.CaseFilters, Depends(case_filters)]


def list_item(case: Case, final: str | None) -> dict:
    t, p = case.transaction, case.prediction
    return {
        "id": case.id,
        "transaction_id": t.external_id,
        "transaction_time": t.transaction_time,
        "amount": float(t.amount),
        "product_code": t.product_code,
        "card_network": t.card_network,
        "fraud_score": p.fraud_score,
        "expected_loss": p.expected_loss,
        "risk_level": p.risk_level,
        "reason": case.reason,
        "priority_tier": case.priority_tier,
        "status": case.status,
        "assignee": UserRef.model_validate(case.assignee) if case.assignee else None,
        "final_decision": final,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "closed_at": case.closed_at,
    }


@router.get("", response_model=CasePage)
def get_cases(
    user: CurrentUser,
    db: Db,
    filters: Filters,
    sort: CaseSort = "priority",
    order: Literal["asc", "desc"] = "desc",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CasePage:
    """The review queue. Default order = priority: model alerts first, then "unusual pattern"
    cases, each by expected loss (highest first); case id breaks ties."""
    try:
        rows, total = svc.list_cases(db, filters, user, sort, order, limit, offset)
    except svc.InvalidFilter as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    finals = svc.final_decisions(db, [c.id for c in rows])
    return CasePage(
        items=[CaseListItem(**list_item(c, finals.get(c.id))) for c in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{case_id}", response_model=CaseDetail)
def get_case(case_id: int, _user: CurrentUser, db: Db) -> CaseDetail:
    """One case: transaction facts, score and risk factors, earlier transactions of the same
    card/customer, and the decisions so far."""
    case = svc.get_case(db, case_id)
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    t, p = case.transaction, case.prediction
    return CaseDetail(
        **list_item(case, svc.final_decisions(db, [case.id]).get(case.id)),
        transaction=TransactionFacts(
            transaction_id=t.external_id,
            transaction_time=t.transaction_time,
            amount=float(t.amount),
            product_code=t.product_code,
            card_network=t.card_network,
            card_type=t.card_type,
            payer_email_domain=t.payer_email_domain,
            device_type=t.device_type,
        ),
        prediction=PredictionOut(
            model_version=p.model_version.version,
            fraud_score=p.fraud_score,
            expected_loss=p.expected_loss,
            risk_level=p.risk_level,
            is_alert=p.is_alert,
            is_unusual=p.is_unusual,
            anomaly_score=p.anomaly_score,
            base_value=p.base_value,
            raw_log_odds=p.shap_values.get("raw_log_odds"),
            other_contribution=p.shap_values.get("other_contribution"),
            top_factors=[RiskFactor(**f) for f in p.shap_values.get("factors", [])],
            scored_at=p.created_at,
        ),
        card_history=[
            HistoryItem(
                transaction_id=h.transaction_id,
                transaction_time=h.transaction_time,
                amount=float(h.amount),
                product_code=h.product_code,
                had_case=h.had_case,
            )
            for h in svc.card_history(db, t)
        ],
        decisions=[
            DecisionOut(
                id=d.id,
                decision=d.decision,
                comment=d.comment,
                analyst=UserRef.model_validate(d.analyst),
                created_at=d.created_at,
            )
            for d in case.decisions
        ],
    )


@router.post("/{case_id}/claim", response_model=CaseListItem)
def claim_case(case_id: int, user: CurrentUser, db: Db, request: Request) -> CaseListItem:
    """Take a case for yourself: `new` -> `assigned`. Escalated cases: admins only.
    409 if the case is closed or already held by someone else."""
    return run_action(db, lambda: svc.claim(db, case_id, user, request))


@router.post("/{case_id}/assign", response_model=CaseListItem)
def assign_case(
    case_id: int, body: AssignRequest, admin: AdminUser, db: Db, request: Request
) -> CaseListItem:
    """Admin: (re)assign an open case to an active user (escalated cases: to an admin)."""
    return run_action(db, lambda: svc.assign(db, case_id, admin, body.user_id, request))


@router.post("/{case_id}/decision", response_model=CaseListItem)
def decide_case(
    case_id: int, body: DecisionRequest, user: CurrentUser, db: Db, request: Request
) -> CaseListItem:
    """The assignee's decision: `confirmed_fraud` or `false_positive` close the case;
    `escalate` (analysts, comment required) sends it unassigned to the admins."""
    return run_action(
        db, lambda: svc.decide(db, case_id, user, body.decision, body.comment, request)
    )


@router.get("/{case_id}/timeline", response_model=list[TimelineEntry])
def get_timeline(case_id: int, _user: CurrentUser, db: Db) -> list[TimelineEntry]:
    """Everything that happened to the case, oldest first (from the audit trail)."""
    if db.get(Case, case_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    return [
        TimelineEntry(
            id=log.id,
            action=log.action,
            actor=UserRef.model_validate(actor) if actor else None,
            before=log.before,
            after=log.after,
            comment=comment,
            created_at=log.created_at,
        )
        for log, actor, comment in svc.timeline(db, case_id)
    ]
