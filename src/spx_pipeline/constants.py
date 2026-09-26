from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

IV_OVERFLOW_THRESHOLD = 100.0
REGIME_WARMUP_DAYS = 49
REGIME_VALID_FROM = date(2023, 4, 13)

DEFAULT_MAX_MEMORY = "8GB"
DEFAULT_CACHE_SIZE_MB = 2000
DEFAULT_ROW_GROUP_SIZE = 300_000
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_THREADS = 4
