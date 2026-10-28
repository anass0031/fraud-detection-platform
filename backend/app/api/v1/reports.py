"""Reports: CSV export of the review queue (any logged-in user; read-only, so not audited)."""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.api.v1.cases import Filters
from app.db.session import get_db
from app.services import cases as svc

router = APIRouter(prefix="/reports", tags=["reports"])

COLUMNS = [
    "case_id",
    "transaction_id",
    "transaction_time",
    "amount",
    "product_code",
    "card_network",
    "fraud_score",
    "expected_loss",
    "risk_level",
    "reason",
    "priority_tier",
    "status",
    "assignee",
    "created_at",
    "closed_at",
    "final_decision",
    "decided_by",
    "decided_at",
]

# A cell starting with one of these is run as a formula by spreadsheet programs.
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


def cases_csv(db: Session, cases: list, finals: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for c in cases:
        t, p, d = c.transaction, c.prediction, finals.get(c.id)
        row = [
            c.id,
            t.external_id,
            t.transaction_time,
            f"{float(t.amount):.2f}",
            t.product_code,
            t.card_network,
            f"{p.fraud_score:.4f}",
            f"{p.expected_loss:.2f}",
            p.risk_level,
            c.reason,
            c.priority_tier,
            c.status,
            c.assignee.email if c.assignee else None,
            c.created_at,
            c.closed_at,
            d.decision if d else None,
            d.analyst.email if d else None,
            d.created_at if d else None,
        ]
        writer.writerow([cell(v) for v in row])
    return buf.getvalue()


@router.get(
    "/cases.csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "CSV file"}},
)
def export_cases(user: CurrentUser, db: Annotated[Session, Depends(get_db)], filters: Filters):
    """All cases matching the queue filters, in queue order, with the final decision.
    Ground truth is never exported."""
    try:
        cases = svc.all_cases(db, filters, user)
    except svc.InvalidFilter as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    body = cases_csv(db, cases, svc.final_decision_rows(db, [c.id for c in cases]))
    name = f"cases-{dt.date.today():%Y%m%d}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
