"""
One-shot migration script: raw data → standardized Parquet ZSTD.

Usage:
    spx-pipeline migrate \
        --raw-root /data/spx/raw \
        --store-root /data/spx/store

    # Migrate a specific dataset only:
    spx-pipeline migrate --raw-root "..." --store-root "..." --dataset vix_ohlc

    # Delete source files after successful conversion (saves disk):
    spx-pipeline migrate --raw-root "..." --store-root "..." --delete-source
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from spx_pipeline.registry import DEFAULT_REGISTRY, Registry
from spx_pipeline.store import DataStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate raw data to Parquet ZSTD")
    parser.add_argument("--raw-root", required=True, help="Path to raw data directory")
    parser.add_argument("--store-root", required=True, help="Path to store directory")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source files after successful conversion",
    )
    parser.add_argument("--dataset", default=None, help="Migrate specific dataset only")
    args = parser.parse_args(argv)

    registry_path = DEFAULT_REGISTRY
    registry = Registry(registry_path)
    store = DataStore(Path(args.raw_root), Path(args.store_root), registry)

    if args.dataset:
        datasets = [args.dataset]
    else:
        # Small datasets first, so a problem shows up before hours of options data.
        datasets = sorted(registry.list_datasets(), key=lambda d: registry.get(d).rows_per_day)

    total_stats = {"ingested": 0, "skipped": 0, "errors": 0, "bytes_written": 0, "bytes_freed": 0}

    for ds in datasets:
        logger.info("=" * 50)
        logger.info("Migrating: %s", ds)
        logger.info("=" * 50)

        t0 = time.time()
        stats = store.ingest_dataset(ds, delete_source=args.delete_source)
        elapsed = time.time() - t0

        for k in total_stats:
            total_stats[k] += stats[k]

        logger.info(
            "  %s: ingested=%d, skipped=%d, errors=%d, written=%.1f MB, freed=%.1f MB (%.1fs)",
            ds,
            stats["ingested"],
            stats["skipped"],
            stats["errors"],
            stats["bytes_written"] / 1024 / 1024,
            stats["bytes_freed"] / 1024 / 1024,
            elapsed,
        )

    logger.info("=" * 50)
    logger.info("MIGRATION COMPLETE")
    logger.info(
        "Total: ingested=%d, skipped=%d, errors=%d, written=%.1f MB, freed=%.1f MB",
        total_stats["ingested"],
        total_stats["skipped"],
        total_stats["errors"],
        total_stats["bytes_written"] / 1024 / 1024,
        total_stats["bytes_freed"] / 1024 / 1024,
    )

    if total_stats["errors"] > 0:
        logger.warning("There were %d errors during migration!", total_stats["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()
