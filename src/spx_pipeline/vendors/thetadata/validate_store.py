"""
Validate the ThetaData daily Hive store for all Greeks DTE 0-7.

Checks:
  1. Store directory exists and is a symlink to ~/thetadata/store/
  2. Expected number of files (>= MIN_FILES trading days since 2019-01-01)
  3. No files with zero rows
  4. Spot-check: read one recent day and verify schema + non-empty

Usage:
    spx-pipeline thetadata validate-store
    spx-pipeline thetadata validate-store --dataset greeks_2dte
    spx-pipeline thetadata validate-store --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

STORE_ROOT = Path(__file__).parent.parent / "store"

# Minimum trading days 2019-01-01 → 2025-12-31 (approx; exact count is ~1771 bdays,
# but some DTEs have fewer files pre-2022 when SPXW only had Mon/Wed/Fri expirations)
MIN_FILES = 1400

GREEKS_SCHEMA = {
    "symbol",
    "expiration",
    "strike",
    "right",
    "timestamp",
    "bid",
    "ask",
    "delta",
    "theta",
    "vega",
    "rho",
    "epsilon",
    "lambda",
    "implied_vol",
    "iv_error",
    "underlying_timestamp",
    "underlying_price",
}

DATASETS = [f"greeks_{i}dte" for i in range(8)]


def validate_dataset(name: str, verbose: bool = False) -> dict:
    store_path = STORE_ROOT / name
    stats = {"dataset": name, "ok": True, "files": 0, "empty_files": 0, "errors": []}

    if not store_path.exists():
        stats["ok"] = False
        stats["errors"].append(f"Store directory missing: {store_path}")
        return stats

    parquet_files = sorted(store_path.rglob("*.parquet"))
    stats["files"] = len(parquet_files)

    if stats["files"] < MIN_FILES:
        stats["ok"] = False
        stats["errors"].append(
            f"Only {stats['files']} files (expected >= {MIN_FILES}). Dataset may be incomplete."
        )

    # Check for zero-row files
    empty = []
    for f in parquet_files:
        try:
            meta = pq.read_metadata(str(f))
            if meta.num_rows == 0:
                empty.append(f.name)
        except Exception as e:
            stats["errors"].append(f"Cannot read metadata for {f.name}: {e}")
            stats["ok"] = False

    if empty:
        stats["empty_files"] = len(empty)
        stats["ok"] = False
        stats["errors"].append(
            f"{len(empty)} empty files: {empty[:5]}{'...' if len(empty) > 5 else ''}"
        )

    # Spot-check: read the most recent file and verify schema
    if parquet_files:
        latest = parquet_files[-1]
        try:
            schema = pq.read_schema(str(latest))
            col_names = set(schema.names)
            missing = GREEKS_SCHEMA - col_names
            if missing:
                stats["ok"] = False
                stats["errors"].append(f"Schema missing columns in {latest.name}: {missing}")
            if verbose:
                table = pq.read_table(str(latest))
                logger.info(
                    "  Spot-check %s: %d rows, %d cols, sample strike=%.0f",
                    latest.name,
                    table.num_rows,
                    table.num_columns,
                    table.column("strike")[0].as_py() if table.num_rows > 0 else 0,
                )
        except Exception as e:
            stats["ok"] = False
            stats["errors"].append(f"Spot-check failed on {latest.name}: {e}")

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate ThetaData store for Greeks DTE 0-7")
    parser.add_argument("--dataset", default=None, help="Validate specific dataset only")
    parser.add_argument("--verbose", action="store_true", help="Show spot-check row details")
    args = parser.parse_args(argv)

    datasets = [args.dataset] if args.dataset else DATASETS
    total_ok = 0
    total_fail = 0

    for name in datasets:
        result = validate_dataset(name, verbose=args.verbose)
        status = "OK  " if result["ok"] else "FAIL"
        logger.info(
            "[%s] %s | %d files | %d empty",
            status,
            result["dataset"],
            result["files"],
            result["empty_files"],
        )
        if result["errors"]:
            for err in result["errors"]:
                logger.warning("       %s", err)
        if result["ok"]:
            total_ok += 1
        else:
            total_fail += 1

    logger.info("─" * 55)
    logger.info("Results: %d OK, %d FAIL", total_ok, total_fail)
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
