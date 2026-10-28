"""Load IEEE-CIS transactions from the processed Parquet file into the `transactions` table.

`raw` keeps every input column of the shared feature pipeline (fraudml RAW_INPUT_COLUMNS) as
exact JSON numbers/strings (NaN -> null), so the online feature path sees the same values as
training. Ground truth is stored for evaluation; it counts as "revealed" only for rows before
the test period (labels of later days are revealed by the replay, never by the scoring API).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from fraudml import config
from fraudml.features.keys import make_uid
from fraudml.features.pipeline import RAW_INPUT_COLUMNS
from sqlalchemy import Connection, text

TRAIN_PARQUET = config.PROCESSED_DIR / "train.parquet"
TEST_START = pd.Timestamp(config.SPLIT_DATES["test"][0])

# Readable columns copied out of `raw` for the UI and filters.
READABLE = {
    "product_code": "ProductCD",
    "card_network": "card4",
    "card_type": "card6",
    "payer_email_domain": "P_emaildomain",
    "device_type": "DeviceType",
}
COLUMNS = (
    "external_id",
    "transaction_dt",
    "transaction_time",
    "amount",
    *READABLE,
    "customer_key",
    "raw",
    "ground_truth",
    "ground_truth_revealed_at",
    "source",
)


def read_period(start: str | None, end: str | None, path: Path = TRAIN_PARQUET) -> pd.DataFrame:
    """Rows with start <= TransactionDatetime < end (None = open side), in canonical order."""
    cols = [*RAW_INPUT_COLUMNS, config.DATETIME_COL, config.TARGET]
    df = pd.read_parquet(path, columns=cols)
    keep = pd.Series(True, index=df.index)
    if start:
        keep &= df[config.DATETIME_COL] >= pd.Timestamp(start)
    if end:
        keep &= df[config.DATETIME_COL] < pd.Timestamp(end)
    df = df[keep]
    return df.sort_values([config.TIME_COL, config.ID_COL], kind="stable").reset_index(drop=True)


def _python_values(s: pd.Series) -> list:
    """Column -> list of plain Python values; missing -> None. Floats keep their exact value."""
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        return [None if pd.isna(v) else str(v) for v in s.astype(object)]
    values = s.to_numpy()
    out = values.tolist()  # numpy -> Python int/float (float32 widened exactly)
    if np.issubdtype(values.dtype, np.floating):
        mask = np.isnan(values)
        out = [None if m else v for v, m in zip(out, mask, strict=True)]
    return out


def to_rows(df: pd.DataFrame, source: str, revealed_at: dt.datetime) -> Iterator[tuple]:
    """Table rows (in COLUMNS order) for a frame read with `read_period`."""
    raw_cols = {c: _python_values(df[c]) for c in RAW_INPUT_COLUMNS}
    uid = make_uid(df)
    keys = [None if pd.isna(u) else str(u) for u in uid]
    times = df[config.DATETIME_COL]
    labels = df[config.TARGET].to_numpy()
    before_test = (times < TEST_START).to_numpy()
    times_py = [t.to_pydatetime() for t in times]
    for i in range(len(df)):
        raw = {c: raw_cols[c][i] for c in RAW_INPUT_COLUMNS}
        yield (
            raw[config.ID_COL],
            raw[config.TIME_COL],
            times_py[i],
            raw["TransactionAmt"],
            *(raw[c] for c in READABLE.values()),
            keys[i],
            json.dumps(raw, allow_nan=False),
            bool(labels[i]),
            revealed_at if before_test[i] else None,
            source,
        )


def load_frame(conn: Connection, df: pd.DataFrame, source: str, chunk_size: int = 50_000) -> int:
    """Insert the rows of `df` that are not stored yet. Returns the number of new rows.

    Each chunk is COPY'd into a temporary table, then inserted with ON CONFLICT DO NOTHING on
    external_id, so re-running the loader is safe.
    """
    revealed_at = dt.datetime.now(dt.UTC)
    cols = ", ".join(COLUMNS)
    inserted = 0
    for start in range(0, len(df), chunk_size):
        part = df.iloc[start : start + chunk_size]
        conn.execute(
            text(
                f"CREATE TEMP TABLE tmp_transactions ON COMMIT DROP AS "
                f"SELECT {cols} FROM transactions WITH NO DATA"
            )
        )
        cur = conn.connection.driver_connection.cursor()
        with cur.copy(f"COPY tmp_transactions ({cols}) FROM STDIN") as copy:
            for row in to_rows(part, source, revealed_at):
                copy.write_row(row)
        result = conn.execute(
            text(
                f"INSERT INTO transactions ({cols}) SELECT {cols} FROM tmp_transactions "
                "ORDER BY transaction_dt, external_id ON CONFLICT (external_id) DO NOTHING"
            )
        )
        inserted += result.rowcount
        conn.execute(text("DROP TABLE tmp_transactions"))
    return inserted
