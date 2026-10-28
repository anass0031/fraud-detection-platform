from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.db.session import get_db
from app.schemas.score import RiskFactor, ScoreResponse, TransactionIn
from app.services.model_store import ModelStore, get_model_store
from app.services.scoring import TransactionConflict, score_raw
from app.services.settings import alert_policy

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score(
    txn: TransactionIn,
    db: Annotated[Session, Depends(get_db)],
    models: Annotated[ModelStore, Depends(get_model_store)],
    _user: CurrentUser,
) -> ScoreResponse:
    """Score one transaction with the active model and store the prediction.

    Features are computed with the shared fraudml pipeline, using the customer's earlier
    transactions stored in the database. Sending the same TransactionID again returns the
    stored prediction (`already_scored`); different data under the same id is a 409.
    Requires a logged-in user (analyst or admin).
    """
    bundle = models.bundle
    if bundle is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"No model loaded: {models.error}")
    try:
        outcome = score_raw(db, bundle, txn.model_dump(), source="realtime")
        db.commit()
    except TransactionConflict as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except IntegrityError as exc:  # same TransactionID sent twice at the same moment
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Transaction is being scored") from exc

    p = outcome.prediction
    meta = bundle.model.metadata
    return ScoreResponse(
        transaction_id=outcome.transaction.external_id,
        model_version=bundle.version,
        anomaly_version=bundle.anomaly_version,
        fraud_score=p.fraud_score,
        expected_loss=p.expected_loss,
        is_alert=p.is_alert,
        alert_policy=meta["policy"],
        alert_threshold=alert_policy(db, meta["threshold"]).threshold,
        risk_level=p.risk_level,
        is_unusual=p.is_unusual,
        anomaly_score=p.anomaly_score,
        base_value=p.base_value,
        top_factors=[RiskFactor(**f) for f in p.shap_values["factors"]],
        latency_ms=p.latency_ms,
        scored_at=p.created_at,
        already_scored=outcome.already_scored,
    )
