"""Check that the manually downloaded IEEE-CIS CSV files are complete.

Usage: ml/.venv/Scripts/python ml/scripts/verify_raw_data.py
Exit code 0 = all files present with the expected row counts, 1 = problems found.
"""

import sys

from fraudml import config
from fraudml.data.ingest import verify_raw_files


def main() -> int:
    print(f"Checking {config.RAW_DIR}")
    problems = verify_raw_files()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return 1
    for name, rows in config.RAW_FILES.items():
        size_mb = (config.RAW_DIR / name).stat().st_size / 1024**2
        print(f"  OK  {name:<24} {rows:>9,} rows  {size_mb:>7.1f} MB")
    print("All raw files present and complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
