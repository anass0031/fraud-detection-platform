"""Convert the raw IEEE-CIS CSVs into data/processed/{train,test}.parquet.

Usage: ml/.venv/Scripts/python ml/scripts/make_parquet.py
"""

import sys
import time

from fraudml import config
from fraudml.data.ingest import build_and_save, verify_raw_files


def main() -> int:
    problems = verify_raw_files()
    if problems:
        print("Raw data problems, stopping:", *problems, sep="\n  - ")
        return 1

    for split in ("train", "test"):
        t0 = time.perf_counter()
        s = build_and_save(split)
        fraud = f"{s.fraud_rate:.4%}" if s.fraud_rate is not None else "n/a"
        print(
            f"[{split}] rows={s.rows:,} cols={s.columns} fraud_rate={fraud}\n"
            f"        period {s.start} -> {s.end}\n"
            f"        memory {s.memory_raw_mb:,.0f} MB (raw) -> {s.memory_final_mb:,.0f} MB "
            f"(downcast) | parquet {s.parquet_mb:,.0f} MB | {time.perf_counter() - t0:.0f} s"
        )
    print(f"Written to {config.PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
