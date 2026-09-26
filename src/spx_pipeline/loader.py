from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from .cache import FeatherCache
from .cleaning import apply_cleaning_rules, resample_ohlc
from .constants import DEFAULT_CACHE_SIZE_MB, DEFAULT_MAX_MEMORY
from .params import DateLike, DateRange, QueryParams, resolve_dates
from .query import QueryEngine
from .registry import DEFAULT_REGISTRY, Registry

logger = logging.getLogger(__name__)


class DataLoader:
    """
    High-level data loader: registry → query → cache → clean → Polars.

    Data flow:
    1. QueryParams from user request
    2. Check FeatherCache for existing result
    3. If miss: DuckDB query → Arrow Table
    4. Arrow → Polars (zero-copy)
    5. Apply cleaning rules from registry
    6. Cast float32 → float64 on numeric columns (hybrid precision)
    7. Optionally resample
    8. Write to cache
    9. Return Polars DataFrame
    """

    def __init__(
        self,
        store_root: Path | str,
        registry_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        max_memory: str = DEFAULT_MAX_MEMORY,
        max_cache_mb: int = DEFAULT_CACHE_SIZE_MB,
    ):
        self._store_root = Path(store_root)
        if registry_path is None:
            registry_path = DEFAULT_REGISTRY
        self._registry = Registry(registry_path)
        self._cache_dir = Path(cache_dir) if cache_dir else self._store_root / ".cache"
        self._engine = QueryEngine(self._store_root, self._registry, max_memory)
        self._cache = FeatherCache(self._cache_dir, max_cache_mb)

    @property
    def registry(self) -> Registry:
        return self._registry

    def load(
        self,
        dataset: str,
        dates: DateLike | DateRange,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
        resample: str | None = None,
        apply_cleaning: bool = True,
        use_cache: bool = True,
        upcast_float64: bool = True,
    ) -> pl.DataFrame:
        """
        Load a dataset as a Polars DataFrame.

        Parameters
        ----------
        dataset : str
            Dataset name from registry (e.g. "spx_ohlc", "greeks_0dte").
        dates : single date or (start, end) tuple
        columns : column subset to select
        filters : list of (column, operator, value) tuples
            e.g. [("right", "=", "PUT"), ("strike", ">=", 4700)]
        resample : Polars duration string, e.g. "5m", "15m", "1h"
        apply_cleaning : apply registry cleaning rules
        use_cache : use Feather disk cache
        upcast_float64 : cast float32 columns to float64 for computation
        """
        dataset = self._registry.resolve(dataset)  # aliases share one store and one cache
        start, end = resolve_dates(dates)
        params = QueryParams(
            dataset=dataset,
            start_date=start,
            end_date=end,
            columns=tuple(columns) if columns else None,
            filters=tuple(tuple(f) for f in filters) if filters else (),
            resample=resample,
            apply_cleaning=apply_cleaning,
        )

        # Check cache
        if use_cache:
            cached = self._cache.get(params)
            if cached is not None:
                df = _to_frame(cached)
                if upcast_float64:
                    df = _upcast_floats(df)
                return df

        # Query DuckDB → Arrow
        arrow_table = self._engine.query(params)

        if arrow_table.num_rows == 0:
            return pl.DataFrame()

        # Arrow → Polars (zero-copy)
        df = _to_frame(arrow_table)

        # Apply cleaning rules
        if apply_cleaning:
            config = self._registry.get(dataset)
            df = apply_cleaning_rules(df, config.cleaning_rules)

        # Resample
        if resample and not df.is_empty():
            df = resample_ohlc(df, resample)

        # Write to cache (before upcast — store as float32)
        if use_cache and not df.is_empty():
            self._cache.put(params, df.to_arrow())

        # Upcast for computation
        if upcast_float64:
            df = _upcast_floats(df)

        return df

    def load_aligned(
        self,
        dates: DateRange,
        datasets: list[str] | None = None,
        resample: str | None = None,
        regime_lag_days: int = 1,
    ) -> pl.DataFrame:
        """
        Load and merge multiple datasets on timestamp.

        Regime is lagged by regime_lag_days trading days to prevent
        intraday look-ahead bias.
        """
        if datasets is None:
            datasets = ["spx_ohlc", "vix_ohlc", "vvix_ohlc", "vol_regime"]

        intraday = [d for d in datasets if d != "vol_regime"]
        frames: dict[str, pl.DataFrame] = {}

        for ds in intraday:
            df = self.load(ds, dates, resample=resample, upcast_float64=False)
            if not df.is_empty():
                prefix = ds.replace("_ohlc", "").replace("_0dte", "")
                rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "timestamp"}
                frames[ds] = df.rename(rename_map)

        if not frames:
            return pl.DataFrame()

        # Outer join all intraday on timestamp
        keys = list(frames.keys())
        base = frames[keys[0]]
        for k in keys[1:]:
            base = base.join(frames[k], on="timestamp", how="full", coalesce=True)

        base = base.sort("timestamp")

        # Merge the daily regime, lagged. The regime is read from a few days *before*
        # the window so that its first days still receive the previous session's regime
        # (without the look-back they would silently get none).
        if "vol_regime" in datasets:
            start, end = resolve_dates(dates)
            lookback = timedelta(days=3 * max(regime_lag_days, 0) + 7)
            regime = self.load(
                "vol_regime", (start - lookback, end), apply_cleaning=True, upcast_float64=False
            )
            if not regime.is_empty() and regime_lag_days > 0:
                regime = self._apply_regime_lag(regime, regime_lag_days)
            if not regime.is_empty():
                regime = regime.filter(pl.col("date").is_between(start, end))
                base = base.with_columns(pl.col("timestamp").cast(pl.Date).alias("_date"))
                regime = regime.rename({"date": "_date"})
                base = base.join(regime, on="_date", how="left")
                base = base.drop("_date")

        return _upcast_floats(base)

    def _apply_regime_lag(self, regime: pl.DataFrame, lag_days: int) -> pl.DataFrame:
        """
        Shift regime dates forward by lag_days trading days.

        The regime computed at close of day T becomes available at
        open of day T+lag. Prevents using end-of-day VIX information
        during intraday trading.
        """
        dates_sorted = regime.get_column("date").sort().to_list()
        shift_map = {}
        for i, d in enumerate(dates_sorted):
            future_idx = i + lag_days
            if future_idx < len(dates_sorted):
                shift_map[d] = dates_sorted[future_idx]

        regime = regime.with_columns(
            pl.col("date").replace_strict(shift_map, default=None).alias("date")
        )
        return regime.drop_nulls("date")

    def available_dates(self, dataset: str) -> list[date]:
        """List dates available in the store for a given dataset."""
        return self._engine.query_dates_available(dataset)

    def invalidate_cache(self, dataset: str | None = None) -> int:
        """Clear cache for a dataset, or all caches."""
        return self._cache.invalidate(dataset)

    def close(self) -> None:
        self._engine.close()


def _to_frame(table: pa.Table) -> pl.DataFrame:
    """Arrow table -> Polars DataFrame, without a copy."""
    out = pl.from_arrow(table)
    return out.to_frame() if isinstance(out, pl.Series) else out


def _upcast_floats(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all Float32 columns to Float64 for computation precision."""
    float32_cols = [c for c in df.columns if df[c].dtype == pl.Float32]
    if float32_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in float32_cols])
    return df
