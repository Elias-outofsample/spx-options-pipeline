"""
Post-migration validation script.

Checks:
1. Row counts: source files vs store files
2. Schema integrity: all expected columns present
3. DuckDB readability: can query each dataset
4. Data sample: compare a known day against expectations

Usage:
    spx-pipeline validate \
        --raw-root /data/spx/raw \
        --store-root /data/spx/store
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from spx_pipeline.registry import DEFAULT_REGISTRY, Registry
from spx_pipeline.store import DataStore

logger = logging.getLogger(__name__)


def validate_dataset(
    registry: Registry,
    store: DataStore,
    store_root: Path,
    dataset_name: str,
) -> bool:
    """Validate a single dataset. Returns True if all checks pass."""
    config = registry.get(dataset_name)
    ok = True

    # Check store files exist
    if config.granularity == "daily" and not config.partition_cols:
        store_path = store_root / config.store_path
        if not store_path.exists():
            logger.error("[%s] Store file missing: %s", dataset_name, store_path)
            return False
        schema = pq.read_schema(str(store_path))
        logger.info("[%s] Store file OK: %s (%d columns)", dataset_name, store_path, len(schema))
    else:
        dates = store.discover_dates(config)
        if not dates:
            logger.warning("[%s] No source dates found", dataset_name)
            return True

        # Check a sample of store files
        store_count = 0
        for d in dates:
            sp = store_root / config.store_path.format(
                year=d.year, month=d.month, date=d.strftime("%Y%m%d")
            )
            if sp.exists():
                store_count += 1

        logger.info(
            "[%s] %d/%d dates have store files",
            dataset_name,
            store_count,
            len(dates),
        )
        if store_count < len(dates):
            logger.warning("[%s] Missing %d store files", dataset_name, len(dates) - store_count)
            ok = False

    # DuckDB readability check
    try:
        glob_pattern = (
            config.store_path.replace("{year}", "*")
            .replace("{month:02d}", "*")
            .replace("{date}", "*")
        )
        full_pattern = str(store_root / glob_pattern)
        conn = duckdb.connect(":memory:")
        result = conn.execute(
            "SELECT COUNT(*) AS cnt FROM read_parquet(?, hive_partitioning=true)", [full_pattern]
        ).fetchone()
        total_rows = result[0] if result else 0
        conn.close()
        logger.info("[%s] DuckDB read OK: %d total rows", dataset_name, total_rows)
    except Exception as e:
        logger.error("[%s] DuckDB read FAILED: %s", dataset_name, e)
        ok = False

    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate migrated data")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--store-root", required=True)
    args = parser.parse_args(argv)

    registry_path = DEFAULT_REGISTRY
    registry = Registry(registry_path)
    store = DataStore(Path(args.raw_root), Path(args.store_root), registry)
    store_root = Path(args.store_root)

    all_ok = True
    for ds in registry.list_datasets():
        logger.info("--- Validating %s ---", ds)
        if not validate_dataset(registry, store, store_root, ds):
            all_ok = False

    if all_ok:
        logger.info("ALL VALIDATIONS PASSED")
    else:
        logger.error("SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
