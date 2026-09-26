"""
Migrate ThetaData monthly parquets → daily Hive-partitioned store.

ThetaData downloads produce monthly files (e.g. 2024_01.parquet).
The spx-pipeline expects daily files in Hive layout:
    store/{dataset}/year=YYYY/month=MM/YYYYMMDD.parquet

This script:
  1. Reads each monthly parquet from ~/thetadata/
  2. Splits rows by date (from timestamp or created column)
  3. For index data: synthesizes OHLC from single price column
  4. Casts to target schema (float32, utf8, etc.)
  5. Drops unwanted columns (exchange, condition)
  6. Writes per-day ZSTD parquet to store

Usage:
    spx-pipeline thetadata split --store-root ~/thetadata/store
    spx-pipeline thetadata split --store-root ... --dataset greeks_0dte
    spx-pipeline thetadata split --store-root ~/thetadata/store --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from spx_pipeline.constants import (
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_ROW_GROUP_SIZE,
)
from spx_pipeline.registry import DEFAULT_REGISTRY, Registry

logger = logging.getLogger(__name__)

from .config import DATA_ROOT as THETADATA_ROOT  # noqa: E402

# Mapping: registry dataset key → ThetaData source directory
# Source files are monthly: {source_dir}/{YYYY_MM}.parquet
TD_DATASETS = {
    "spx_ohlc": "index/spx",
    "vix_ohlc": "index/vix",
    "vvix_ohlc": "index/vvix",
    "greeks_0dte": "options/greeks/0dte",
    "greeks_1dte": "options/greeks/1dte",
    "greeks_2dte": "options/greeks/2dte",
    "greeks_3dte": "options/greeks/3dte",
    "greeks_4dte": "options/greeks/4dte",
    "greeks_5dte": "options/greeks/5dte",
    "greeks_6dte": "options/greeks/6dte",
    "greeks_7dte": "options/greeks/7dte",
    "quotes_0dte": "options/quotes/0dte",
    "quotes_1dte": "options/quotes/1dte",
    "iv_0dte": "options/iv/0dte",
    "iv_1dte": "options/iv/1dte",
    "oi": "options/oi",
    "eod": "options/eod",
}

# Migration order: light datasets first
MIGRATION_ORDER = [
    "spx_ohlc",
    "vix_ohlc",
    "vvix_ohlc",
    "oi",
    "eod",
    "quotes_0dte",
    "quotes_1dte",
    "greeks_0dte",
    "greeks_1dte",
    "greeks_2dte",
    "greeks_3dte",
    "greeks_4dte",
    "greeks_5dte",
    "greeks_6dte",
    "greeks_7dte",
    "iv_0dte",
    "iv_1dte",
]

# Columns to drop during migration (before writing to store)


def _detect_date_column(table: pa.Table) -> str:
    """Detect the primary timestamp/date column for splitting by day."""
    for col in ("timestamp", "created"):
        if col in table.schema.names:
            return col
    raise ValueError(f"No date column found in: {table.schema.names}")


def _split_by_date(table: pa.Table, date_col: str) -> dict[str, pa.Table]:
    """Split an Arrow table by date, returning {YYYYMMDD: subtable}."""
    ts = table.column(date_col)

    # Extract date as YYYYMMDD string
    dates_arr = pc.strftime(pc.cast(ts, pa.timestamp("us")), format="%Y%m%d")

    result = {}
    unique_dates = pc.unique(dates_arr).to_pylist()

    for date_str in unique_dates:
        if date_str is None:
            continue
        mask = pc.equal(dates_arr, date_str)
        subtable = table.filter(mask)
        result[date_str] = subtable

    return result


def _synthesize_ohlc(table: pa.Table) -> pa.Table:
    """Convert index price data (timestamp, price) → OHLC format.

    For 1-min index snapshots, each bar is a single price point,
    so open = high = low = close = price.
    """
    price = table.column("price")
    ts = table.column("timestamp")

    return pa.table(
        {
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
        }
    )


def _drop_columns(table: pa.Table, cols: list[str]) -> pa.Table:
    for col in cols:
        if col in table.schema.names:
            idx = table.schema.get_field_index(col)
            table = table.remove_column(idx)
    return table


def _cast_right_to_utf8(table: pa.Table) -> pa.Table:
    """ThetaData stores 'right' as dictionary(int8→string). Cast to plain utf8."""
    if "right" not in table.schema.names:
        return table
    right_field = table.schema.field("right")
    if pa.types.is_dictionary(right_field.type):
        idx = table.schema.get_field_index("right")
        right_col = table.column("right")
        # ChunkedArray needs combine_chunks before dictionary_decode
        if isinstance(right_col, pa.ChunkedArray):
            right_utf8 = pc.cast(right_col, pa.utf8())
        else:
            right_utf8 = right_col.dictionary_decode()
        table = table.set_column(idx, pa.field("right", pa.utf8()), right_utf8)
    return table


def _cast_timestamps_ns(table: pa.Table) -> pa.Table:
    """Cast timestamp columns from us → ns to match pipeline convention."""
    for col_name in ("timestamp", "created", "last_trade"):
        if col_name not in table.schema.names:
            continue
        col = table.column(col_name)
        if col.type == pa.timestamp("us"):
            idx = table.schema.get_field_index(col_name)
            casted = pc.cast(col, pa.timestamp("ns"))
            table = table.set_column(idx, pa.field(col_name, pa.timestamp("ns")), casted)
    return table


def _cast_underlying_timestamp_utf8(table: pa.Table) -> pa.Table:
    """Cast underlying_timestamp from timestamp → utf8 string to match registry."""
    col_name = "underlying_timestamp"
    if col_name not in table.schema.names:
        return table
    col = table.column(col_name)
    if pa.types.is_timestamp(col.type):
        idx = table.schema.get_field_index(col_name)
        str_col = pc.strftime(col, format="%Y-%m-%d %H:%M:%S")
        table = table.set_column(idx, pa.field(col_name, pa.utf8()), str_col)
    return table


def migrate_dataset(
    dataset_name: str,
    store_root: Path,
    registry: Registry,
    dry_run: bool = False,
) -> dict:
    """Migrate one ThetaData dataset: monthly parquets → daily Hive store."""
    config = registry.get(dataset_name)
    source_dir = THETADATA_ROOT / TD_DATASETS[config.name]
    is_index = config.name.endswith("_ohlc")
    cols_to_drop = config.drop_cols  # single source of truth: the registry

    stats = {"ingested": 0, "skipped": 0, "errors": 0, "bytes_written": 0, "rows": 0}

    month_files = sorted(source_dir.glob("*.parquet"))
    if not month_files:
        logger.warning("No parquet files in %s", source_dir)
        return stats

    for month_file in month_files:
        try:
            table = pq.read_table(str(month_file))

            if table.num_rows == 0:
                stats["skipped"] += 1
                continue

            # Transform index data → OHLC
            if is_index:
                table = _synthesize_ohlc(table)
            else:
                # Options data: cast dictionary→utf8, drop unwanted cols
                table = _cast_right_to_utf8(table)
                table = _drop_columns(table, cols_to_drop)
                table = _cast_underlying_timestamp_utf8(table)

            # Cast timestamps us → ns
            table = _cast_timestamps_ns(table)

            # Split by date
            date_col = _detect_date_column(table)
            daily = _split_by_date(table, date_col)

            for date_str, day_table in daily.items():
                year = int(date_str[:4])
                month = int(date_str[4:6])
                store_path = store_root / config.store_path.format(
                    year=year, month=month, date=date_str
                )

                # Skip if already migrated and same size
                if store_path.exists():
                    existing_meta = pq.read_metadata(str(store_path))
                    if existing_meta.num_rows == day_table.num_rows:
                        stats["skipped"] += 1
                        continue

                if dry_run:
                    logger.info(
                        "  [DRY] %s → %s (%d rows)",
                        month_file.name,
                        store_path.name,
                        day_table.num_rows,
                    )
                    stats["ingested"] += 1
                    stats["rows"] += day_table.num_rows
                    continue

                store_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    day_table,
                    str(store_path),
                    compression=DEFAULT_COMPRESSION,
                    compression_level=DEFAULT_COMPRESSION_LEVEL,
                    row_group_size=DEFAULT_ROW_GROUP_SIZE,
                )

                stats["ingested"] += 1
                stats["bytes_written"] += store_path.stat().st_size
                stats["rows"] += day_table.num_rows

        except Exception as e:
            logger.error("Failed %s/%s: %s", dataset_name, month_file.name, e)
            stats["errors"] += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate ThetaData → daily Hive store")
    parser.add_argument(
        "--store-root",
        default=str(THETADATA_ROOT),
        help="Root directory for the store (default: ~/thetadata)",
    )
    parser.add_argument("--dataset", default=None, help="Migrate specific dataset only")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without writing"
    )
    args = parser.parse_args(argv)

    store_root = Path(args.store_root)
    registry_path = DEFAULT_REGISTRY
    registry = Registry(registry_path)

    if args.dataset:
        datasets = [registry.resolve(args.dataset)]  # legacy td_* names accepted
    else:
        datasets = [d for d in MIGRATION_ORDER if d in registry.list_datasets()]

    total_stats = {"ingested": 0, "skipped": 0, "errors": 0, "bytes_written": 0, "rows": 0}

    for ds in datasets:
        logger.info("=" * 55)
        logger.info("  Migrating: %s ← %s", ds, TD_DATASETS[ds])
        logger.info("=" * 55)

        t0 = time.time()
        stats = migrate_dataset(ds, store_root, registry, dry_run=args.dry_run)
        elapsed = time.time() - t0

        for k in total_stats:
            total_stats[k] += stats[k]

        logger.info(
            "  %s: days=%d, skipped=%d, errors=%d, rows=%s, written=%.1f MB (%.1fs)",
            ds,
            stats["ingested"],
            stats["skipped"],
            stats["errors"],
            f"{stats['rows']:,}",
            stats["bytes_written"] / 1e6,
            elapsed,
        )

    logger.info("=" * 55)
    logger.info("MIGRATION COMPLETE")
    logger.info(
        "Total: days=%d, skipped=%d, errors=%d, rows=%s, written=%.1f GB",
        total_stats["ingested"],
        total_stats["skipped"],
        total_stats["errors"],
        f"{total_stats['rows']:,}",
        total_stats["bytes_written"] / 1e9,
    )

    if total_stats["errors"] > 0:
        logger.warning("There were %d errors!", total_stats["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()
