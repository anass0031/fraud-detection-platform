"""Raw IEEE-CIS CSV files -> merged, downcast DataFrames -> Parquet.

Steps for each split (train / test):
1. read the transaction and identity CSVs,
2. rename test identity columns `id-01` -> `id_01` so train and test share one schema,
3. left-join identity onto transactions (only ~24% of transactions have identity data),
4. add a real datetime column from the `TransactionDT` seconds offset,
5. downcast types to save memory (TransactionAmt stays float64 to keep exact cents).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fraudml import config

Split = Literal["train", "test"]

# Columns that must not lose precision when downcasting.
KEEP_FLOAT64 = frozenset({"TransactionAmt"})


# ---------------------------------------------------------------------------
# Raw file verification
# ---------------------------------------------------------------------------
def count_data_rows(path: Path, chunk_size: int = 1 << 24) -> int:
    """Count CSV data rows (header excluded) by counting newlines; fast, no parsing."""
    newlines = 0
    last_byte = b""
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    lines = newlines + (1 if last_byte not in (b"\n", b"") else 0)
    return max(lines - 1, 0)


def verify_raw_files(
    raw_dir: Path = config.RAW_DIR,
    expected: dict[str, int] | None = None,
    check_rows: bool = True,
) -> list[str]:
    """Return a list of problems (empty list = all good)."""
    expected = config.RAW_FILES if expected is None else expected
    problems: list[str] = []
    for name, n_rows in expected.items():
        path = raw_dir / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
            continue
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip()
        if not header.startswith(config.ID_COL):
            problems.append(f"{name}: header does not start with {config.ID_COL}")
        if check_rows:
            found = count_data_rows(path)
            if found != n_rows:
                problems.append(f"{name}: expected {n_rows:,} rows, found {found:,}")
    return problems


# ---------------------------------------------------------------------------
# Reading and merging
# ---------------------------------------------------------------------------
def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Test identity uses `id-01`; train uses `id_01`. Use the underscore form everywhere."""
    return df.rename(columns=lambda c: c.replace("id-", "id_") if c.startswith("id-") else c)


def read_split(split: Split, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Read and merge the transaction + identity CSVs of one split (no downcasting)."""
    tx = pd.read_csv(raw_dir / f"{split}_transaction.csv")
    ident = normalize_identity_columns(pd.read_csv(raw_dir / f"{split}_identity.csv"))
    # one_to_one: raises if a TransactionID is duplicated on either side.
    return tx.merge(ident, on=config.ID_COL, how="left", validate="one_to_one")


def add_datetime(df: pd.DataFrame, reference: str = config.REFERENCE_DATETIME) -> pd.DataFrame:
    """Add TransactionDatetime = reference + TransactionDT seconds (right after TransactionDT)."""
    out = df.copy()
    dt = pd.Timestamp(reference) + pd.to_timedelta(out[config.TIME_COL], unit="s")
    out.insert(out.columns.get_loc(config.TIME_COL) + 1, config.DATETIME_COL, dt)
    return out


# ---------------------------------------------------------------------------
# Downcasting
# ---------------------------------------------------------------------------
def downcast(df: pd.DataFrame, keep_float64: frozenset[str] = KEEP_FLOAT64) -> pd.DataFrame:
    """Shrink dtypes: float64 -> float32, ints -> smallest int, text -> category.

    float32 keeps ~7 significant digits, which is exact for every integer-valued column in this
    dataset (all < 16.7M) and enough for the continuous ones. TransactionAmt is kept as float64
    because its cents part becomes a feature later.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if col in keep_float64 or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s):
            out[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[col] = s.astype("category")
    return out


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1024**2


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------
@dataclass
class BuildSummary:
    split: str
    rows: int
    columns: int
    fraud_rate: float | None
    memory_raw_mb: float
    memory_final_mb: float
    parquet_mb: float
    start: pd.Timestamp
    end: pd.Timestamp


def build_split(split: Split, raw_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, float]:
    """Return (final DataFrame, raw memory in MB before downcasting)."""
    raw = read_split(split, raw_dir)
    raw_mb = memory_mb(raw)
    final = downcast(add_datetime(raw))
    return final, raw_mb


def write_parquet(df: pd.DataFrame, path: Path) -> float:
    """Write Parquet (zstd) and return the file size in MB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path.stat().st_size / 1024**2


def build_and_save(
    split: Split, raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR
) -> BuildSummary:
    df, raw_mb = build_split(split, raw_dir)
    size_mb = write_parquet(df, out_dir / f"{split}.parquet")
    return BuildSummary(
        split=split,
        rows=len(df),
        columns=df.shape[1],
        fraud_rate=float(df[config.TARGET].mean()) if config.TARGET in df else None,
        memory_raw_mb=raw_mb,
        memory_final_mb=memory_mb(df),
        parquet_mb=size_mb,
        start=df[config.DATETIME_COL].min(),
        end=df[config.DATETIME_COL].max(),
    )


def load_split(split: Split, processed_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    """Load a processed split (what notebooks and later steps use)."""
    return pd.read_parquet(processed_dir / f"{split}.parquet", engine="pyarrow")
