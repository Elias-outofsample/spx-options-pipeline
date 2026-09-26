from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

DateLike = str | date | datetime
DateRange = tuple[DateLike, DateLike]


class Granularity(Enum):
    TICK = "tick"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    HOUR_1 = "1h"
    DAILY = "daily"


class DatasetName(Enum):
    SPX_OHLC = "spx_ohlc"
    GREEKS_0DTE = "greeks_0dte"
    VIX_OHLC = "vix_ohlc"
    VVIX_OHLC = "vvix_ohlc"
    VOL_REGIME = "vol_regime"


@dataclass(frozen=True)
class QueryParams:
    """Immutable query parameters — hashable for cache key generation."""

    dataset: str
    start_date: date
    end_date: date
    columns: tuple[str, ...] | None = None
    filters: tuple[tuple[str, str, object], ...] = ()
    resample: str | None = None
    apply_cleaning: bool = True

    def cache_key(self) -> str:
        """Deterministic SHA-256 hash (first 16 hex chars) for cache file naming."""
        payload = json.dumps(
            {
                "dataset": self.dataset,
                "start": str(self.start_date),
                "end": str(self.end_date),
                "columns": self.columns,
                "filters": [list(f) for f in self.filters],
                "resample": self.resample,
                "apply_cleaning": self.apply_cleaning,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def to_date(d: DateLike) -> date:
    """Normalize any DateLike to a date object."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return date.fromisoformat(d)
    raise TypeError(f"Cannot convert {type(d)} to date")


def resolve_dates(dates: DateLike | DateRange) -> tuple[date, date]:
    """Normalize to (start_date, end_date) pair."""
    if isinstance(dates, (list, tuple)) and len(dates) == 2:
        return to_date(dates[0]), to_date(dates[1])
    d = to_date(dates)
    return d, d
