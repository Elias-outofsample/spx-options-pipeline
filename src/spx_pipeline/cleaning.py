"""Cleaning rules, applied by name from ``registry.yaml``.

Every rule is a pure function ``(DataFrame, params) -> DataFrame`` registered in
``RULE_HANDLERS``. Rules never look forward in time: gaps are forward-filled only,
and nothing is computed from a later row than the one it is written to.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date as date_type
from typing import Any

import polars as pl

from .registry import CleaningRule

logger = logging.getLogger(__name__)

RuleHandler = Callable[[pl.DataFrame, dict[str, Any]], pl.DataFrame]

# Columns that identify one option contract inside a chain, in sort order.
CONTRACT_KEY_COLUMNS = ("symbol", "expiration", "expiry", "strike", "right")


def contract_key(df: pl.DataFrame) -> list[str]:
    """Columns of ``df`` that identify an instrument (empty for a single series)."""
    return [c for c in CONTRACT_KEY_COLUMNS if c in df.columns]


def apply_cleaning_rules(df: pl.DataFrame, rules: list[CleaningRule]) -> pl.DataFrame:
    """Apply a sequence of cleaning rules to a Polars DataFrame, in order."""
    for rule in rules:
        handler = RULE_HANDLERS.get(rule.name)
        if handler is None:
            logger.warning("Unknown cleaning rule: %s", rule.name)
            continue
        df = handler(df, rule.params)
    return df


def resample_ohlc(df: pl.DataFrame, rule: str) -> pl.DataFrame:
    """Resample 1-minute OHLC bars to a coarser timeframe.

    ``closed="left", label="left"``: the bar labelled 10:00 aggregates the 1-minute
    bars stamped 10:00 … 10:04 and nothing later, matching the convention of the
    source bars (stamped at their start). As with any start-stamped bar, its close is
    only known at label + ``rule``: a backtest must act on it at the *next* bar.
    """
    if df.is_empty() or "timestamp" not in df.columns:
        return df

    aggs = [
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]
    if "volume" in df.columns:
        aggs.append(pl.col("volume").sum())
    return (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=rule, closed="left", label="left")
        .agg(aggs)
        .drop_nulls("close")
    )


# ---------------------------------------------------------------------------
# Individual rule handlers
# ---------------------------------------------------------------------------


def _drop_opening_bar(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop the 09:30 bar (options: underlying_price = 0 and IV overflow at the open)."""
    hour, minute = (int(x) for x in params.get("time", "09:30:00").split(":")[:2])
    n_before = len(df)
    df = df.filter(
        ~((pl.col("timestamp").dt.hour() == hour) & (pl.col("timestamp").dt.minute() == minute))
    )
    logger.debug("drop_opening_bar: removed %d rows", n_before - len(df))
    return df


def _iv_column(df: pl.DataFrame) -> str | None:
    for col in ("implied_vol", "iv"):
        if col in df.columns:
            return col
    return None


def _filter_illiquid(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop untradeable rows: zero bid, IV overflow, missing underlying price.

    Accepts ``implied_vol`` (1-minute datasets) or ``iv`` (tick dataset) as the IV column.
    """
    conditions = []
    if "bid_gt" in params:
        conditions.append(pl.col("bid") > params["bid_gt"])
    if "implied_vol_lte" in params:
        iv = _iv_column(df)
        if iv is None:
            raise KeyError("filter_illiquid: no 'implied_vol' or 'iv' column to filter on")
        conditions.append(pl.col(iv) <= params["implied_vol_lte"])
    if "underlying_price_gt" in params:
        conditions.append(pl.col("underlying_price") > params["underlying_price_gt"])
    return _filter_all(df, conditions, "filter_illiquid")


def _ffill_zero_close(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Treat 0.0 prices as missing (stale VIX/VVIX prints) and forward-fill them.

    Forward only: a backward fill would copy a later price into an earlier bar.
    """
    if not params.get("enabled", True):
        return df

    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    key = contract_key(df)
    df = df.sort([*key, "timestamp"]) if key else df.sort("timestamp")
    exprs = []
    for col in price_cols:
        filled = pl.when(pl.col(col) == 0.0).then(None).otherwise(pl.col(col)).forward_fill()
        exprs.append((filled.over(key) if key else filled).alias(col))
    return df.with_columns(exprs).sort("timestamp") if exprs else df


def _add_mid(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Add ``mid = (bid + ask) / 2``."""
    if "bid" in df.columns and "ask" in df.columns:
        df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).cast(pl.Float32).alias("mid"))
    return df


def _valid_from(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop rows before a validity date (warm-up period of rolling statistics)."""
    valid_date = date_type.fromisoformat(str(params.get("value", "2023-04-13")))
    if "date" in df.columns:
        df = df.filter(pl.col("date") >= valid_date)
    return df


def _filter_duplicate_timestamps(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Remove duplicate prints of the *same instrument* at the same timestamp.

    The key is the timestamp plus whichever contract columns exist (symbol,
    expiration/expiry, strike, right). Deduplicating on timestamp alone would collapse
    a whole option chain to a single row per timestamp.
    """
    keep = params.get("keep", "last")
    subset = ["timestamp", *contract_key(df)]
    n_before = len(df)
    df = df.unique(subset=subset, keep=keep, maintain_order=True)
    logger.debug("filter_duplicate_timestamps: removed %d rows", n_before - len(df))
    return df


def _filter_outlier_spread(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop rows whose bid-ask spread exceeds ``max_spread_pct`` % of the mid.

    Rows with a non-positive mid have no meaningful spread and are dropped too.
    """
    if "bid" not in df.columns or "ask" not in df.columns:
        return df
    max_spread_pct = float(params.get("max_spread_pct", 5.0))
    mid = (pl.col("bid") + pl.col("ask")) / 2.0
    spread_pct = (pl.col("ask") - pl.col("bid")) / mid * 100.0
    return _filter_all(df, [mid > 0, spread_pct <= max_spread_pct], "filter_outlier_spread")


def _filter_zero_oi(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop rows whose open interest is at or below a threshold."""
    threshold = params.get("open_interest_gt", params.get("oi_gt"))
    if threshold is None:
        return df
    col = "open_interest" if "open_interest" in df.columns else "oi"
    return _filter_all(df, [pl.col(col) > threshold], "filter_zero_oi")


def _filter_zero_price(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Drop trade prints at or below a price threshold."""
    if "price_gt" not in params:
        return df
    return _filter_all(df, [pl.col("price") > params["price_gt"]], "filter_zero_price")


def _filter_bogus_expirations(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Remove OPRA test contracts (year 1882) and encoding errors (year > 2040).

    A vendor validation pass found such rows in the raw feed: 45 contracts expiring in
    2088-2089 in a single month of end-of-day data.
    """
    col = next((c for c in ("expiration", "expiry") if c in df.columns), None)
    if col is None:
        return df
    min_year = int(params.get("min_year", 2000))
    max_year = int(params.get("max_year", 2040))
    return _filter_all(
        df, [_as_date(df, col).dt.year().is_between(min_year, max_year)], "filter_bogus_expirations"
    )


def _add_spread_pct(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Add ``spread_pct = (ask - bid) / mid * 100`` (null when mid <= 0)."""
    if "bid" in df.columns and "ask" in df.columns:
        mid = (pl.col("bid") + pl.col("ask")) / 2.0
        spread = (pl.col("ask") - pl.col("bid")) / mid * 100.0
        df = df.with_columns(
            pl.when(mid > 0).then(spread).otherwise(None).cast(pl.Float32).alias("spread_pct")
        )
    return df


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _filter_all(df: pl.DataFrame, conditions: list[pl.Expr], name: str) -> pl.DataFrame:
    if not conditions:
        return df
    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined & cond
    n_before = len(df)
    df = df.filter(combined)
    logger.debug("%s: %d -> %d rows", name, n_before, len(df))
    return df


def _as_date(df: pl.DataFrame, col: str) -> pl.Expr:
    """``col`` as a Date expression, whatever its stored type (string, date, datetime)."""
    dtype = df.schema[col]
    if dtype == pl.Date:
        return pl.col(col)
    if isinstance(dtype, pl.Datetime):
        return pl.col(col).dt.date()
    return pl.col(col).cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)


RULE_HANDLERS: dict[str, RuleHandler] = {
    "drop_opening_bar": _drop_opening_bar,
    "filter_illiquid": _filter_illiquid,
    "filter_zero_bid": _filter_illiquid,
    "ffill_zero_close": _ffill_zero_close,
    "add_mid": _add_mid,
    "valid_from": _valid_from,
    "filter_duplicate_timestamps": _filter_duplicate_timestamps,
    "filter_outlier_spread": _filter_outlier_spread,
    "filter_zero_oi": _filter_zero_oi,
    "filter_zero_price": _filter_zero_price,
    "filter_bogus_expirations": _filter_bogus_expirations,
    "add_spread_pct": _add_spread_pct,
}
