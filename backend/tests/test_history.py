"""PostgresHistory must return exactly what fraudml's InMemoryHistory returns."""

import numpy as np
import pandas as pd
import pytest
from fraudml.features.history import InMemoryHistory
from fraudml.features.keys import make_uid
from sqlalchemy.orm import Session

from app.services.history import PostgresHistory
from app.services.transactions_loader import load_frame
from tests.test_loader import _frame


def _customer_frame() -> pd.DataFrame:
    """Two customers, same-second ties, mixed-case e-mails and missing devices."""
    base = _frame().iloc[[0]]
    rows = []
    times = [1000, 2000, 2000, 2000, 5000, 9000]
    for i, t in enumerate(times):
        r = base.copy()
        r["TransactionID"] = np.int32(100 + i)
        r["TransactionDT"] = np.int32(10_000_000 + t)
        r["TransactionDatetime"] = pd.Timestamp("2017-12-01") + pd.Timedelta(seconds=10_000_000 + t)
        r["TransactionAmt"] = 10.0 + i
        r["card1"] = np.int16(1000 if i % 3 else 2000)
        r["P_emaildomain"] = ["Gmail.com", None, " yahoo.com", "gmail.com", None, "x.com"][i]
        r["DeviceInfo"] = [None, "SM-G", "iOS Device", None, "SM-G", "Windows"][i]
        rows.append(r)
    df = pd.concat(rows, ignore_index=True)
    for c in ("P_emaildomain", "DeviceInfo"):
        df[c] = pd.Categorical(df[c].astype(object))
    return df


@pytest.mark.parametrize("before_offset", [0, 1000, 2000, 2001, 9000, 99_999])
def test_postgres_history_equals_in_memory(db_session: Session, before_offset):
    df = _customer_frame()
    load_frame(db_session.connection(), df, "history")
    expected_src, actual_src = InMemoryHistory(df), PostgresHistory(db_session)
    before = 10_000_000 + before_offset
    for uid in make_uid(df).dropna().unique():
        expected = expected_src.get(uid, before).reset_index(drop=True)
        actual = actual_src.get(uid, before)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        assert list(actual["TransactionDT"]) == sorted(actual["TransactionDT"])


def test_unknown_customer_has_empty_history(db_session: Session):
    hist = PostgresHistory(db_session).get("no_such_uid", before=10**9)
    assert hist.empty
    assert list(hist.columns) == ["TransactionDT", "TransactionAmt", "device", "p_email"]
