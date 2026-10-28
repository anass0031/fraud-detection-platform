import datetime as dt

import pytest
from fraudml.registry import MODELS_DIR
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Prediction, Transaction
from app.scripts.score_batch import pending_ids, score_ids
from app.services.model_store import ModelStore, register
from app.services.transactions_loader import load_frame
from tests.test_loader import _frame

MODEL, ANOMALY = "lgbm-2026.10.27-1", "iforest-2026.10.27-1"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / MODEL).exists(), reason="trained artifacts not available"
)


def test_batch_scores_pending_rows_once(db_session: Session):
    register(db_session, MODEL, ANOMALY, activate=True)
    bundle = ModelStore().load_active(db_session)
    load_frame(db_session.connection(), _frame(), "batch")  # 2018-04-26 .. 2018-04-28

    day = (dt.date(2018, 4, 26), dt.date(2018, 4, 28))
    ids = pending_ids(db_session, bundle.model_version_id, *day)
    assert len(ids) == 2  # the 2018-04-28 row is outside [from, to)

    preds = score_ids(db_session, bundle, ids, chunk_size=1)
    assert len(preds) == 2
    assert pending_ids(db_session, bundle.model_version_id, *day) == []
    assert db_session.scalar(select(func.count()).select_from(Prediction)) == 2

    # The second transaction of the same customer used the first one as history.
    rows = db_session.scalars(select(Transaction).order_by(Transaction.external_id)).all()
    assert rows[0].features["uid_n_prior"] == 0.0
    assert rows[1].features["uid_n_prior"] == 1.0
