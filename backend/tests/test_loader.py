import numpy as np
import pandas as pd
from fraudml import config
from fraudml.features.keys import make_uid
from fraudml.features.pipeline import RAW_INPUT_COLUMNS
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Transaction
from app.services.transactions_loader import TEST_START, load_frame


def _frame() -> pd.DataFrame:
    """Three transactions shaped like data/processed/train.parquet (float32 + categories)."""
    n = 3
    df = pd.DataFrame({c: np.full(n, np.nan, dtype="float32") for c in RAW_INPUT_COLUMNS})
    df[config.ID_COL] = np.array([10, 11, 12], dtype="int32")
    times = pd.to_datetime(["2018-04-26 10:00:00", "2018-04-27 23:59:59", "2018-04-28 00:00:00"])
    df[config.DATETIME_COL] = times
    ref = pd.Timestamp(config.REFERENCE_DATETIME)
    df[config.TIME_COL] = ((times - ref) // pd.Timedelta(seconds=1)).astype("int32")
    df["TransactionAmt"] = np.array([10.5, 99.999, 0.1], dtype="float64")
    df["card1"] = np.array([1000, 1000, 2000], dtype="int16")
    df["addr1"] = np.array([300, 300, np.nan], dtype="float32")
    df["D1"] = np.array([5, 6, 0], dtype="float32")
    df["dist1"] = np.array([0.1, np.nan, 3], dtype="float32")
    for c in ("ProductCD", "P_emaildomain", "DeviceType", "card4", "M4"):
        df[c] = pd.Categorical(["W", None, "H"] if c == "ProductCD" else ["x.com", None, "y"])
    df[config.TARGET] = np.array([0, 1, 1], dtype="int8")
    return df


def _stored(db: Session) -> list[Transaction]:
    return list(db.scalars(select(Transaction).order_by(Transaction.external_id)))


def test_load_is_idempotent(db_session: Session):
    conn = db_session.connection()
    df = _frame()
    assert load_frame(conn, df, "history", chunk_size=2) == 3
    assert load_frame(conn, df, "history", chunk_size=2) == 0
    assert len(_stored(db_session)) == 3


def test_raw_values_round_trip(db_session: Session):
    df = _frame()
    load_frame(db_session.connection(), df, "batch")
    rows = _stored(db_session)
    first = rows[0].raw
    assert set(first) == set(RAW_INPUT_COLUMNS)
    assert first["dist1"] == float(np.float32(0.1))  # exact float32 value, not 0.1
    assert first["id_01"] is None  # NaN -> JSON null
    assert rows[1].raw["P_emaildomain"] is None
    assert rows[0].payer_email_domain == "x.com"
    assert float(rows[1].amount) == 99.999
    assert [r.customer_key for r in rows] == [None if pd.isna(u) else u for u in make_uid(df)]
    assert rows[2].customer_key is None  # addr1 missing -> no uid


def test_ground_truth_revealed_only_before_test_period(db_session: Session):
    load_frame(db_session.connection(), _frame(), "history")
    rows = _stored(db_session)
    assert [r.ground_truth for r in rows] == [False, True, True]
    assert rows[1].transaction_time < TEST_START.to_pydatetime() <= rows[2].transaction_time
    assert rows[0].ground_truth_revealed_at is not None
    assert rows[1].ground_truth_revealed_at is not None
    assert rows[2].ground_truth_revealed_at is None
