"""Ingest tests on tiny synthetic CSVs (no real data needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudml.data import ingest


def _write_raw(raw: Path, split: str, with_target: bool, id_sep: str) -> None:
    tx = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [86_400, 90_000, 172_800],
            "TransactionAmt": [117.99, 49.0, 1234.567],
            "ProductCD": ["W", "C", "W"],
            "card1": [1000, 2000, 1000],
            "card2": [111.0, np.nan, 222.0],
            "V1": [1.0, 0.5, np.nan],
        }
    )
    if with_target:
        tx.insert(1, "isFraud", [0, 1, 0])
    ident = pd.DataFrame({"TransactionID": [2], f"id{id_sep}01": [-5.0], "DeviceType": ["mobile"]})
    raw.mkdir(parents=True, exist_ok=True)
    tx.to_csv(raw / f"{split}_transaction.csv", index=False)
    ident.to_csv(raw / f"{split}_identity.csv", index=False)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write_raw(raw, "train", with_target=True, id_sep="_")
    _write_raw(raw, "test", with_target=False, id_sep="-")
    return raw


def test_merge_keeps_all_transactions_and_renames_test_identity(raw_dir: Path):
    train = ingest.read_split("train", raw_dir)
    test = ingest.read_split("test", raw_dir)
    assert len(train) == 3 and len(test) == 3
    assert "id_01" in test.columns and "id-01" not in test.columns
    # Only transaction 2 has identity data; the others get NaN.
    assert train["DeviceType"].isna().tolist() == [True, False, True]
    assert set(train.columns) - set(test.columns) == {"isFraud"}


def test_duplicate_identity_rows_raise(raw_dir: Path):
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    pd.concat([ident, ident]).to_csv(raw_dir / "train_identity.csv", index=False)
    with pytest.raises(pd.errors.MergeError):
        ingest.read_split("train", raw_dir)


def test_datetime_is_reference_plus_seconds(raw_dir: Path):
    df = ingest.add_datetime(ingest.read_split("train", raw_dir))
    assert df["TransactionDatetime"].iloc[0] == pd.Timestamp("2017-12-02 00:00:00")
    cols = list(df.columns)
    assert cols.index("TransactionDatetime") == cols.index("TransactionDT") + 1


def test_downcast_types_and_exact_amounts(raw_dir: Path):
    raw = ingest.read_split("train", raw_dir)
    df = ingest.downcast(ingest.add_datetime(raw))
    assert df["TransactionAmt"].dtype == np.float64
    assert df["TransactionAmt"].tolist() == [117.99, 49.0, 1234.567]
    assert df["card2"].dtype == np.float32 and df["V1"].dtype == np.float32
    assert df["card1"].dtype == np.int16
    assert df["isFraud"].dtype == np.int8
    assert isinstance(df["ProductCD"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionDatetime"])
    # Values are unchanged apart from float32 rounding.
    np.testing.assert_allclose(df["card2"], raw["card2"], rtol=1e-6)


def test_build_and_save_roundtrip(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "processed"
    summary = ingest.build_and_save("train", raw_dir, out)
    assert summary.rows == 3 and summary.fraud_rate == pytest.approx(1 / 3)
    loaded = ingest.load_split("train", out)
    assert loaded.shape == (3, summary.columns)
    assert loaded["TransactionAmt"].dtype == np.float64


def test_verify_raw_files_reports_problems(raw_dir: Path):
    expected = {"train_transaction.csv": 3, "train_identity.csv": 5, "missing.csv": 1}
    problems = ingest.verify_raw_files(raw_dir, expected)
    assert "missing file: missing.csv" in problems
    assert any("train_identity.csv: expected 5 rows, found 1" in p for p in problems)
    assert not any(p.startswith("train_transaction.csv") for p in problems)


def test_count_data_rows_without_trailing_newline(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"TransactionID,a\n1,2\n3,4")
    assert ingest.count_data_rows(p) == 2
