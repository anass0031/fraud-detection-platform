"""PostgreSQL history source for the shared online feature path (fraudml.features).

Returns exactly what fraudml's InMemoryHistory returns: the customer's transactions with
TransactionDT < before, ordered by (TransactionDT, TransactionID), with columns
TransactionDT, TransactionAmt, device, p_email. Values come from the stored raw JSON and are
normalised with the same fraudml helper as the batch path, so both give identical features.
"""

from __future__ import annotations

import pandas as pd
from fraudml.features.history import HISTORY_COLUMNS
from fraudml.features.static import as_str
from sqlalchemy import text
from sqlalchemy.orm import Session

_QUERY = text(
    """
    SELECT transaction_dt,
           (raw ->> 'TransactionAmt')::float8 AS amount,
           raw ->> 'DeviceInfo' AS device_info,
           raw ->> 'P_emaildomain' AS p_email
    FROM transactions
    WHERE customer_key = :uid AND transaction_dt < :before
    ORDER BY transaction_dt, external_id
    """
)


class PostgresHistory:
    def __init__(self, db: Session):
        self.db = db

    def get(self, uid: str, before: int) -> pd.DataFrame:
        rows = self.db.execute(_QUERY, {"uid": uid, "before": int(before)}).all()
        if not rows:
            return pd.DataFrame(
                {
                    "TransactionDT": pd.Series(dtype="int64"),
                    "TransactionAmt": pd.Series(dtype="float64"),
                    "device": pd.Series(dtype="object"),
                    "p_email": pd.Series(dtype="object"),
                }
            )[HISTORY_COLUMNS]
        dt_, amount, device, email = zip(*rows, strict=True)
        return pd.DataFrame(
            {
                "TransactionDT": pd.Series(dt_, dtype="int64"),
                "TransactionAmt": pd.Series(amount, dtype="float64"),
                "device": as_str(pd.Series(device, dtype="object")),
                "p_email": as_str(pd.Series(email, dtype="object"), lower=True),
            }
        )[HISTORY_COLUMNS]
