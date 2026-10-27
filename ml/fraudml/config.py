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

ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Chronological split (fixed from the EDA). Each period is [start, end): end date excluded.
# The 7-day gaps are never used for training or evaluation, only as past history for features.
SPLIT_DATES: dict[str, tuple[str, str]] = {
    "train": ("2017-12-02", "2018-03-12"),
    "gap_1": ("2018-03-12", "2018-03-19"),
    "validation": ("2018-03-19", "2018-04-21"),
    "gap_2": ("2018-04-21", "2018-04-28"),
    "test": ("2018-04-28", "2018-06-02"),
}
MODEL_PERIODS = ("train", "validation", "test")

UID_COL = "uid"
PERIOD_COL = "period"
EVAL_FLAG_COL = "uid_fraud_in_train"  # evaluation only: uid already had a fraud in the train period

# Never model features: identifiers, time keys, label, evaluation-only flags.
FORBIDDEN_FEATURES = frozenset(
    {TARGET, ID_COL, TIME_COL, DATETIME_COL, UID_COL, PERIOD_COL, EVAL_FLAG_COL}
)

# Expected raw files and their row counts (header excluded), from the official Kaggle dataset.
RAW_FILES: dict[str, int] = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
    "test_transaction.csv": 506_691,
    "test_identity.csv": 141_907,
    "sample_submission.csv": 506_691,
}
