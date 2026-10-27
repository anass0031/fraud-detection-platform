"""Repository paths and dataset constants shared by scripts, notebooks and tests."""

import os
from pathlib import Path

# ml/fraudml/config.py -> repo root is two levels above the package folder.
REPO_ROOT = Path(os.environ.get("FRAUD_REPO_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# TransactionDT is a seconds offset. Reference date = common community convention for this dataset.
REFERENCE_DATETIME = "2017-12-01 00:00:00"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
DATETIME_COL = "TransactionDatetime"

# Expected raw files and their row counts (header excluded), from the official Kaggle dataset.
RAW_FILES: dict[str, int] = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
    "test_transaction.csv": 506_691,
    "test_identity.csv": 141_907,
    "sample_submission.csv": 506_691,
}
