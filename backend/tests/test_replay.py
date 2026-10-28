import datetime as dt

import pytest
from fraudml.registry import MODELS_DIR
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Case, Prediction, Transaction
from app.scripts.replay import days_in, previous_day_loaded, replay_day, reveal_labels
from app.services.model_store import ModelStore, register
from app.services.transactions_loader import load_frame
from tests.test_loader import _frame

MODEL, ANOMALY = "lgbm-2026.10.27-1", "iforest-2026.10.27-1"
needs_artifacts = pytest.mark.skipif(
    not (MODELS_DIR / MODEL).exists(), reason="trained artifacts not available"
)

DAY = dt.date(2018, 4, 28)  # first test day; _frame() has one row on it (and two before)


def _count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


def test_days_in_excludes_the_end():
    assert days_in(dt.date(2018, 4, 29), dt.date(2018, 5, 2)) == [
        dt.date(2018, 4, 29),
        dt.date(2018, 4, 30),
        dt.date(2018, 5, 1),
    ]


def test_reveal_only_touches_that_day(db_session: Session):
    load_frame(db_session.connection(), _frame(), "batch")  # labels before the test: revealed
    rows = db_session.scalars(select(Transaction).order_by(Transaction.external_id)).all()
    assert [r.ground_truth_revealed_at is not None for r in rows] == [True, True, False]
    assert reveal_labels(db_session, DAY) == 1
    assert reveal_labels(db_session, DAY) == 0  # already revealed
    db_session.expire_all()
    assert all(r.ground_truth_revealed_at is not None for r in rows)


def test_previous_day_must_be_loaded(db_session: Session):
    assert not previous_day_loaded(db_session, DAY)
    load_frame(db_session.connection(), _frame(), "batch")
    assert previous_day_loaded(db_session, DAY)  # 2018-04-27 23:59:59 is in _frame()


@needs_artifacts
def test_replay_day_end_to_end_and_idempotent(db_session: Session):
    register(db_session, MODEL, ANOMALY, activate=True)
    bundle = ModelStore().load_active(db_session)
    frame = _frame()
    history, today = frame.iloc[:2], frame.iloc[2:]
    load_frame(db_session.connection(), history, "history")

    calls: list[dt.date] = []
    first = replay_day(db_session, bundle, DAY, today, hooks=(lambda db, b, d: calls.append(d),))
    assert (first.rows, first.loaded, first.scored, first.revealed) == (1, 1, 1, 1)
    assert first.cases == int(first.alerts + first.unusual)
    assert calls == [DAY]

    txn = db_session.scalar(select(Transaction).where(Transaction.external_id == 12))
    assert txn.source == "replay"
    assert txn.ground_truth_revealed_at is not None
    assert txn.transaction_time.date() == DAY
    assert db_session.scalar(
        select(AuditLog).where(AuditLog.action == "replay.day", AuditLog.entity_id == "2018-04-28")
    ).after == {"loaded": 1, "scored": 1, "cases": first.cases, "revealed": 1}

    # Re-running the day changes nothing and writes no second audit row.
    counts = (
        _count(db_session, Transaction),
        _count(db_session, Prediction),
        _count(db_session, Case),
    )
    again = replay_day(db_session, bundle, DAY, today)
    assert (again.loaded, again.scored, again.cases, again.revealed) == (0, 0, 0, 0)
    assert counts == (
        _count(db_session, Transaction),
        _count(db_session, Prediction),
        _count(db_session, Case),
    )
    assert (
        len(db_session.scalars(select(AuditLog).where(AuditLog.action == "replay.day")).all()) == 1
    )


class InlinePool:
    """Stand-in for the process pool: same chunking and merging, run in this process."""

    def map(self, fn, chunks):
        return [fn(c) for c in chunks]


@needs_artifacts
def test_pool_path_scores_every_pending_row_once(db_session: Session, monkeypatch):
    from app.scripts import replay

    register(db_session, MODEL, ANOMALY, activate=True)
    bundle = ModelStore().load_active(db_session)
    load_frame(db_session.connection(), _frame(), "replay")
    monkeypatch.setattr(replay, "CHUNK", 1)  # one row per chunk: exercises the merge
    monkeypatch.setitem(replay._worker, "db", db_session)
    monkeypatch.setitem(replay._worker, "bundle", bundle)

    flags = replay.score_pending(db_session, bundle, dt.date(2018, 4, 26), InlinePool())
    assert len(flags) == 1  # one row on 04-26
    flags += replay.score_pending(db_session, bundle, DAY - dt.timedelta(days=1), InlinePool())
    flags += replay.score_pending(db_session, bundle, DAY, InlinePool())
    assert len(flags) == 3
    assert _count(db_session, Prediction) == 3
    assert replay.score_pending(db_session, bundle, DAY, InlinePool()) == []  # nothing left
