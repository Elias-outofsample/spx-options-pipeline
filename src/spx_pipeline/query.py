"""DuckDB query engine over the Hive-partitioned Parquet store.

The engine never loads a whole file: date bounds and user filters are pushed down
to the Parquet reader, so only the row groups that can match are read. Results are
returned as Arrow tables, which Polars consumes without a copy.

Everything that reaches the SQL text is either a validated ``date`` object, an
escaped identifier, or an escaped path. Filter *values* always travel as bind
parameters.
"""

from __future__ import annotations

import glob as globmod
import logging
import re
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from .constants import DEFAULT_MAX_MEMORY, DEFAULT_THREADS
from .params import QueryParams
from .registry import DatasetConfig, Registry

logger = logging.getLogger(__name__)

# Operators accepted in ``(column, operator, value)`` filter tuples.
_SCALAR_OPS = {"=", "!=", "<", "<=", ">", ">=", "LIKE"}
_LIST_OPS = {"IN", "NOT IN"}
_ALLOWED_OPS = _SCALAR_OPS | _LIST_OPS

_MEMORY_RE = re.compile(r"^\d+(\.\d+)?\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)$", re.IGNORECASE)


class QueryEngine:
    """DuckDB-based query engine for the Parquet store.

    - predicate pushdown: date bounds and filters are pushed to the Parquet reader;
    - memory budget: DuckDB ``memory_limit`` caps peak RAM;
    - Arrow output: zero-copy hand-off to Polars downstream;
    - works on Hive-partitioned (``year=/month=``) and flat layouts.
    """

    def __init__(
        self,
        store_root: Path | str,
        registry: Registry,
        max_memory: str = DEFAULT_MAX_MEMORY,
        threads: int = DEFAULT_THREADS,
    ):
        if not _MEMORY_RE.match(max_memory):
            raise ValueError(f"max_memory must look like '8GB' or '512MB', got {max_memory!r}")
        if threads < 1:
            raise ValueError("threads must be >= 1")
        self._store_root = Path(store_root)
        self._registry = registry
        self._conn = duckdb.connect(":memory:")
        self._conn.execute(f"SET memory_limit = '{max_memory}'")
        self._conn.execute(f"SET threads = {int(threads)}")

    # ------------------------------------------------------------------ queries

    def query(self, params: QueryParams) -> pa.Table:
        """Execute a query and return an Arrow table."""
        config = self._registry.get(params.dataset)
        sql, bind_values = self._build_sql(params, config)
        logger.debug("DuckDB SQL: %s", sql)
        return _to_arrow(self._conn.execute(sql, bind_values))

    def query_greeks_with_dte(self, params: QueryParams) -> pa.Table:
        """Query an options dataset and derive days-to-expiry at query time.

        ``dte`` is whole calendar days; ``dte_fractional`` is the time left until the
        16:00 ET settlement, in days. DTE is never stored: it depends on the
        observation time, so storing it would freeze one observer's point of view.

        Assumes timestamps are naive Eastern Time (CBOE wall clock), which is what the
        store guarantees for options datasets.
        """
        config = self._registry.get(params.dataset)
        expiry = _expiry_column(config)
        if expiry is None:
            raise ValueError(f"Dataset '{params.dataset}' has no expiration/expiry column")
        exp = _ident(expiry)
        extra = (
            f"DATE_DIFF('day', CAST(timestamp AS DATE), CAST({exp} AS DATE)) AS dte, "
            f"DATE_DIFF('second', timestamp, CAST({exp} AS DATE) + INTERVAL '16 hours')"
            " / 86400.0 AS dte_fractional"
        )
        sql, bind_values = self._build_sql(params, config, extra_columns=extra)
        logger.debug("DuckDB SQL (with DTE): %s", sql)
        return _to_arrow(self._conn.execute(sql, bind_values))

    def query_dates_available(self, dataset_name: str) -> list[date]:
        """List the dates that have a file in the store."""
        config = self._registry.get(dataset_name)
        store_pattern = (
            config.store_path.replace("{year}", "*")
            .replace("{month:02d}", "*")
            .replace("{month}", "*")
            .replace("{date}", "*")
        )
        dates = []
        for f in sorted(globmod.glob(str(self._store_root / store_pattern))):
            stem = Path(f).stem
            try:
                dates.append(date(int(stem[:4]), int(stem[4:6]), int(stem[6:8])))
            except (ValueError, IndexError):
                continue
        return dates

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> QueryEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- SQL building

    def _build_sql(
        self,
        params: QueryParams,
        config: DatasetConfig,
        extra_columns: str | None = None,
    ) -> tuple[str, list[Any]]:
        columns = self._build_columns(params)
        if extra_columns:
            columns = f"{columns}, {extra_columns}"
        where_clause, bind_values = self._build_where(params, config)
        source = f"read_parquet({_literal(self._build_glob(config))}, hive_partitioning=true)"
        sql = f"SELECT {columns} FROM {source} {where_clause} {self._build_order(config)}"
        return sql.strip(), bind_values

    def _build_glob(self, config: DatasetConfig) -> str:
        if config.partition_cols:
            base = config.store_path.split("year=")[0]
            return str(self._store_root / base / "year=*/month=*/*.parquet")
        return str(self._store_root / config.store_path.replace("{date}", "*"))

    @staticmethod
    def _build_columns(params: QueryParams) -> str:
        if params.columns:
            return ", ".join(_ident(c) for c in params.columns)
        return "*"

    @staticmethod
    def _build_where(params: QueryParams, config: DatasetConfig) -> tuple[str, list[Any]]:
        """Build the WHERE clause.

        Date bounds are inlined (they are ``date`` objects, formatted by Python).
        Filter values always use ``?`` binds; ``IN`` / ``NOT IN`` take a list.
        """
        conditions: list[str] = []
        bind_values: list[Any] = []

        index_kind = _index_kind(config)
        idx = _ident(config.index_col)
        if index_kind == "timestamp":
            next_day = params.end_date + timedelta(days=1)
            conditions.append(f"{idx} >= TIMESTAMP '{params.start_date.isoformat()} 00:00:00'")
            conditions.append(f"{idx} < TIMESTAMP '{next_day.isoformat()} 00:00:00'")
        elif index_kind == "date":
            conditions.append(f"{idx} >= DATE '{params.start_date.isoformat()}'")
            conditions.append(f"{idx} <= DATE '{params.end_date.isoformat()}'")
        else:
            raise ValueError(
                f"Dataset '{config.name}': index_col {config.index_col!r} is neither a "
                "timestamp nor a date column, so the date range cannot be pushed down"
            )

        for col, op, value in params.filters:
            op_norm = " ".join(str(op).split()).upper()
            if op_norm not in _ALLOWED_OPS:
                raise ValueError(f"Disallowed SQL operator: {op!r}")
            if op_norm in _LIST_OPS:
                conditions.append(f"{_ident(col)} {op_norm} (SELECT UNNEST(?))")
                bind_values.append(_as_list(value))
            else:
                conditions.append(f"{_ident(col)} {op_norm} ?")
                bind_values.append(value)

        if conditions:
            return "WHERE " + " AND ".join(conditions), bind_values
        return "", bind_values

    @staticmethod
    def _build_order(config: DatasetConfig) -> str:
        if config.index_col == "timestamp" and {"strike", "right"} <= set(config.schema or {}):
            return 'ORDER BY timestamp, strike, "right"'
        return f"ORDER BY {_ident(config.index_col)}"


# ---------------------------------------------------------------------- helpers


def _ident(name: str) -> str:
    """Quote an SQL identifier. Always quoted; embedded quotes are escaped."""
    return '"' + str(name).replace('"', '""') + '"'


def _literal(text: str) -> str:
    """Quote an SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _index_kind(config: DatasetConfig) -> str | None:
    """'timestamp' or 'date', from the declared type of the index column."""
    declared = (config.schema or {}).get(config.index_col, "")
    if declared.startswith("timestamp") or (not declared and config.index_col == "timestamp"):
        return "timestamp"
    if declared.startswith("date") or (not declared and config.index_col == "date"):
        return "date"
    return None


def _expiry_column(config: DatasetConfig) -> str | None:
    for col in ("expiration", "expiry"):
        if col in (config.schema or {}):
            return col
    return None


def _to_arrow(result: Any) -> pa.Table:
    """Materialise a DuckDB result as an Arrow table, across DuckDB versions."""
    to_table = getattr(result, "to_arrow_table", None)  # DuckDB >= 1.4
    if to_table is not None:
        return to_table()
    return result.fetch_arrow_table()
