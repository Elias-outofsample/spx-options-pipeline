"""Point-in-time access: the guard rails that keep future data out of a backtest.

``PointInTimeLoader`` wraps a ``DataLoader`` and enforces, at every call:

1. a hard ``cutoff_date`` — nothing dated after it is ever returned, even if the
   store has it (defence in depth: the query is clamped *and* the result filtered);
2. a regime lag of at least one trading day — a daily regime computed from the
   close of day T is only known at the open of T+1;
3. per-instrument feature lags — ``with_feature_lag`` shifts each contract's own
   history, never a neighbouring strike's.

What it deliberately does *not* do: drop expired contracts. Removing instruments that
no longer exist today would be survivorship bias.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, time
from typing import Any

import polars as pl

from .cleaning import contract_key
from .loader import DataLoader
from .params import DateLike, DateRange, resolve_dates

logger = logging.getLogger(__name__)


class PointInTimeLoader:
    """Anti-look-ahead wrapper around :class:`DataLoader`.

    Usage in a backtest::

        pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))
        df = pit.load("greeks_0dte", ("2024-06-01", "2024-07-15"))  # ends 2024-06-30

    Walk-forward::

        for cutoff in walk_forward_dates:
            pit = PointInTimeLoader(loader, cutoff_date=cutoff)
            train = pit.load("greeks_0dte", (train_start, cutoff))
    """

    def __init__(
        self,
        loader: DataLoader,
        cutoff_date: date,
        regime_lag_days: int = 1,
        feature_lag_bars: int = 0,
    ):
        if regime_lag_days < 1:
            raise ValueError(
                "regime_lag_days must be >= 1 for PointInTimeLoader. "
                "A lag of 0 exposes the close-of-day regime to that same day's intraday "
                "bars (look-ahead). Use DataLoader directly for end-of-day studies."
            )
        if feature_lag_bars < 0:
            raise ValueError("feature_lag_bars must be >= 0")
        self._loader = loader
        self._cutoff = cutoff_date
        self._regime_lag = regime_lag_days
        self._feature_lag = feature_lag_bars

    @property
    def cutoff_date(self) -> date:
        return self._cutoff

    @property
    def regime_lag_days(self) -> int:
        return self._regime_lag

    def load(self, dataset: str, dates: DateLike | DateRange, **kwargs: Any) -> pl.DataFrame:
        """Load ``dataset`` over ``dates``, clamped to the cutoff."""
        start, end = resolve_dates(dates)
        if start > self._cutoff:
            logger.warning("start %s is after cutoff %s: returning empty", start, self._cutoff)
            return pl.DataFrame()
        if end > self._cutoff:
            logger.info("clamping end_date %s to cutoff %s", end, self._cutoff)
            end = self._cutoff

        df = self._loader.load(dataset, (start, end), **kwargs)
        return self._enforce_cutoff(df)

    def load_aligned(
        self,
        dates: DateRange,
        datasets: list[str] | None = None,
        resample: str | None = None,
    ) -> pl.DataFrame:
        """Load and align several datasets, with the cutoff and regime lag enforced."""
        start, end = resolve_dates(dates)
        if start > self._cutoff:
            return pl.DataFrame()
        df = self._loader.load_aligned(
            (start, min(end, self._cutoff)),
            datasets=datasets,
            resample=resample,
            regime_lag_days=self._regime_lag,
        )
        return self._enforce_cutoff(df)

    def with_feature_lag(
        self,
        df: pl.DataFrame,
        feature_cols: Sequence[str],
        lag: int | None = None,
        by: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Add ``{col}_lag{n}`` columns: each feature as it was ``n`` bars earlier.

        Greeks at bar *t* are computed from the underlying price at bar *t*, so they
        cannot drive a decision taken at bar *t*; shifted by one bar they can.

        The shift runs **per instrument**. On an option chain the rows are
        ``timestamp × strike × right``; a plain ``shift(1)`` would hand each contract
        the value of its neighbour at the *same* minute — no lag at all. ``by``
        defaults to the contract-key columns present in ``df`` (symbol, expiration,
        strike, right); pass ``by=[]`` to shift a single series.

        If an instrument is missing a bar, its lag is the previous bar it *has*: older
        information, never newer.
        """
        n = self._feature_lag if lag is None else lag
        if n < 0:
            raise ValueError("lag must be >= 0")
        if n == 0:
            return df

        cols = [c for c in feature_cols if c in df.columns]
        missing = sorted(set(feature_cols) - set(cols))
        if missing:
            logger.warning("with_feature_lag: columns not found and skipped: %s", missing)
        if not cols:
            return df

        keys = list(contract_key(df) if by is None else by)
        has_ts = "timestamp" in df.columns
        if has_ts:
            df = df.sort([*keys, "timestamp"])
        exprs = []
        for c in cols:
            shifted = pl.col(c).shift(n)
            exprs.append((shifted.over(keys) if keys else shifted).alias(f"{c}_lag{n}"))
        df = df.with_columns(exprs)
        return df.sort(["timestamp", *keys]) if has_ts else df

    def _enforce_cutoff(self, df: pl.DataFrame) -> pl.DataFrame:
        """Defence in depth: drop any row dated after the cutoff."""
        if df.is_empty():
            return df
        if "timestamp" in df.columns:
            limit = datetime.combine(self._cutoff, time.max)
            return df.filter(pl.col("timestamp") <= limit)
        if "date" in df.columns:
            return df.filter(pl.col("date") <= self._cutoff)
        return df
