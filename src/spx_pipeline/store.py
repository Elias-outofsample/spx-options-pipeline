from __future__ import annotations

import contextlib
import glob as globmod
import logging
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from .registry import DatasetConfig, Registry

logger = logging.getLogger(__name__)


class DataStore:
    """
    Ingestion layer: converts raw data to standardized Parquet (ZSTD).

    Two modes:
    - CSV sources → read with pyarrow.csv, write Parquet ZSTD
    - Parquet Snappy sources → read + rewrite with ZSTD

    Disk-safe: processes one file at a time (peak extra disk ≈ 4 MB).
    """

    def __init__(self, raw_root: Path | str, store_root: Path | str, registry: Registry):
        self._raw_root = Path(raw_root)
        self._store_root = Path(store_root)
        self._registry = registry

    def ingest_dataset(
        self,
        dataset_name: str,
        dates: list[date] | None = None,
        delete_source: bool = False,
    ) -> dict:
        """
        Ingest an entire dataset (or specific dates) into the store.

        Returns a stats dict: {ingested, skipped, errors, bytes_written, bytes_freed}.
        """
        config = self._registry.get(dataset_name)

        _check_naive_timestamps(config)

        stats = {
            "ingested": 0,
            "skipped": 0,
            "errors": 0,
            "bytes_written": 0,
            "bytes_freed": 0,
        }

        if config.is_single_file:
            self._ingest_single_file(config, stats, delete_source)
        elif config.is_bulk:
            # Monthly bulk files (a '*' in source_path): streamed in batches.
            self._ingest_tick_dataset(config, stats, delete_source)
        else:
            # One raw file per trading day.
            if dates is None:
                dates = self.discover_dates(config)
            for d in dates:
                try:
                    self._ingest_date(config, d, stats, delete_source)
                except Exception as e:
                    logger.error("Failed to ingest %s for %s: %s", dataset_name, d, e)
                    stats["errors"] += 1

        return stats

    def discover_dates(self, config: DatasetConfig) -> list[date]:
        """Scan the raw directory to discover available dates for a dataset."""
        pattern = (
            config.source_path.replace("{year}", "*")
            .replace("{month:02d}", "*")
            .replace("{date}", "*")
        )
        files = sorted(globmod.glob(str(self._raw_root / pattern)))
        dates = []
        for f in files:
            stem = Path(f).stem
            datestr = stem.rsplit("_", 1)[-1]
            with contextlib.suppress(ValueError, IndexError):
                dates.append(date(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8])))
        return dates

    def _ingest_date(
        self,
        config: DatasetConfig,
        d: date,
        stats: dict,
        delete_source: bool,
    ) -> None:
        source_path = self._resolve_source_path(config, d)
        store_path = self._resolve_store_path(config, d)

        if not source_path.exists():
            stats["skipped"] += 1
            return

        if store_path.exists() and source_path.stat().st_mtime <= store_path.stat().st_mtime:
            stats["skipped"] += 1
            return

        store_path.parent.mkdir(parents=True, exist_ok=True)

        if config.source_format == "csv":
            table = self._read_csv(source_path)
        elif config.source_format == "parquet":
            table = pq.read_table(str(source_path))
        else:
            raise ValueError(f"Unknown source format: {config.source_format}")

        table = self._drop_columns(table, config.drop_cols)

        pq.write_table(
            table,
            str(store_path),
            compression=config.compression,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )

        stats["ingested"] += 1
        stats["bytes_written"] += store_path.stat().st_size

        if delete_source and source_path.resolve() != store_path.resolve():
            freed = source_path.stat().st_size
            source_path.unlink()
            stats["bytes_freed"] += freed

    def _ingest_single_file(
        self,
        config: DatasetConfig,
        stats: dict,
        delete_source: bool,
    ) -> None:
        source_path = self._raw_root / config.source_path
        store_path = self._store_root / config.store_path

        if not source_path.exists():
            logger.warning("Source file missing: %s", source_path)
            stats["errors"] += 1
            return

        store_path.parent.mkdir(parents=True, exist_ok=True)

        if config.source_format == "csv":
            table = self._read_csv(source_path)
        else:
            table = pq.read_table(str(source_path))

        table = self._drop_columns(table, config.drop_cols)

        pq.write_table(
            table,
            str(store_path),
            compression=config.compression,
            compression_level=config.compression_level,
        )

        stats["ingested"] += 1
        stats["bytes_written"] += store_path.stat().st_size

    def _read_csv(self, path: Path) -> pa.Table:
        """Read CSV with pyarrow (not pandas)."""
        return pcsv.read_csv(str(path))

    def _resolve_source_path(self, config: DatasetConfig, d: date) -> Path:
        return self._raw_root / config.source_path.format(
            year=d.year, month=d.month, date=d.strftime("%Y%m%d")
        )

    def _resolve_store_path(self, config: DatasetConfig, d: date) -> Path:
        return self._store_root / config.store_path.format(
            year=d.year, month=d.month, date=d.strftime("%Y%m%d")
        )

    def _ingest_tick_dataset(
        self,
        config: DatasetConfig,
        stats: dict,
        delete_source: bool,
    ) -> None:
        """Discover tick source files and ingest each with batched reader."""
        pattern = (
            config.source_path.replace("{year}", "*")
            .replace("{month:02d}", "*")
            .replace("{month}", "*")
        )
        source_files = sorted(globmod.glob(str(self._raw_root / pattern)))

        # Derive store directory template (strip the /*.parquet wildcard)
        store_dir_template = config.store_path.rsplit("/", 1)[0]

        for sf_str in source_files:
            sf = Path(sf_str)
            try:
                year = month = None
                for part in sf.parts:
                    if part.startswith("year="):
                        year = int(part.split("=")[1])
                    elif part.startswith("month="):
                        month = int(part.split("=")[1])

                if year is None or month is None:
                    logger.warning("Cannot extract year/month from %s", sf)
                    stats["errors"] += 1
                    continue

                store_dir = self._store_root / store_dir_template.format(year=year, month=month)
                store_file = store_dir / sf.name

                if store_file.exists() and sf.stat().st_mtime <= store_file.stat().st_mtime:
                    stats["skipped"] += 1
                    continue

                store_file.parent.mkdir(parents=True, exist_ok=True)
                self._ingest_tick_file(sf, store_file, config)
                stats["ingested"] += 1
                stats["bytes_written"] += store_file.stat().st_size

                if delete_source and sf.resolve() != store_file.resolve():
                    freed = sf.stat().st_size
                    sf.unlink()
                    stats["bytes_freed"] += freed

            except Exception as e:
                logger.error("Failed to ingest tick file %s: %s", sf, e)
                stats["errors"] += 1

    # Column renames for tick ingestion (source name → target name)
    _TICK_RENAME_MAP = {
        "expiration": "expiry",
        "implied_vol": "iv",
    }

    def _ingest_tick_file(self, source_path: Path, store_path: Path, config: DatasetConfig) -> None:
        """Stream the file in 1M-row batches so memory stays flat on multi-GB files."""
        reader = pq.ParquetFile(str(source_path))
        schema_arrow = self._build_arrow_schema(config.schema) if config.schema else None
        keep_cols = list(config.schema.keys()) if config.schema else None
        writer = None

        try:
            for batch in reader.iter_batches(batch_size=1_000_000):
                table = pa.Table.from_batches([batch])

                # Rename columns (expiration→expiry, implied_vol→iv)
                rename = {k: v for k, v in self._TICK_RENAME_MAP.items() if k in table.schema.names}
                if rename:
                    table = table.rename_columns([rename.get(c, c) for c in table.schema.names])

                # Select only target columns and cast to target schema
                if keep_cols:
                    table = table.select([c for c in keep_cols if c in table.schema.names])
                if schema_arrow is not None:
                    table = table.cast(schema_arrow)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(store_path),
                        schema=table.schema,
                        compression=config.compression,
                        compression_level=config.compression_level,
                    )
                writer.write_table(table, row_group_size=config.row_group_size)
        except Exception:
            if writer:
                writer.close()
                writer = None
            store_path.unlink(missing_ok=True)
            raise
        finally:
            if writer:
                writer.close()

    @staticmethod
    def _build_arrow_schema(schema_dict: dict[str, str]) -> pa.Schema:
        """Convert a registry schema dict to a pyarrow Schema."""
        type_map = {
            "timestamp[ns]": pa.timestamp("ns"),
            "timestamp[us]": pa.timestamp("us"),
            "float32": pa.float32(),
            "float64": pa.float64(),
            "int8": pa.int8(),
            "int16": pa.int16(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "utf8": pa.utf8(),
            "string": pa.utf8(),
            "date32": pa.date32(),
            "bool": pa.bool_(),
        }
        fields = []
        for name, type_str in schema_dict.items():
            pa_type = type_map.get(type_str)
            if pa_type is None:
                raise ValueError(f"Unknown Arrow type mapping: {type_str}")
            fields.append(pa.field(name, pa_type))
        return pa.schema(fields)

    @staticmethod
    def _drop_columns(table: pa.Table, cols: list[str]) -> pa.Table:
        for col in cols:
            if col in table.column_names:
                idx = table.column_names.index(col)
                table = table.remove_column(idx)
        return table


_NAIVE_TIMESTAMP_TYPES = ("timestamp[us]", "timestamp[ns]", "timestamp[ms]")


def _check_naive_timestamps(config: DatasetConfig) -> None:
    """Options datasets must carry naive Eastern-Time timestamps (CBOE wall clock).

    Days-to-expiry is computed at query time as the distance to 16:00 on the
    expiration date; a timezone-aware or UTC timestamp would shift it by 4-5 hours.
    """
    schema = config.schema or {}
    if not {"strike"} <= set(schema) or "timestamp" not in schema:
        return
    if schema["timestamp"] not in _NAIVE_TIMESTAMP_TYPES:
        raise ValueError(
            f"Options dataset '{config.name}' declares timestamp type "
            f"'{schema['timestamp']}'; expected a naive timestamp "
            f"({', '.join(_NAIVE_TIMESTAMP_TYPES)})."
        )
