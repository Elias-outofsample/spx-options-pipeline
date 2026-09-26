"""Configuration for ThetaData downloader — single source of truth.

All download parameters, schemas, and dataset definitions.

Gamma note: greeks/first_order does NOT include gamma (ThetaData treats it
as second-order: d²V/dS²). Recompute at analysis time from stored fields:
  underlying_price, strike, expiration, implied_vol, timestamp
  → gamma = N'(d1) / (S * sigma * sqrt(T))
"""

import os
from collections import OrderedDict
from pathlib import Path

# ─── Connection ──────────────────────────────────────────────────────────────
# The ThetaData Terminal runs locally and holds the credentials itself
# (launch_terminal.sh reads them from a creds file you create; never commit it).
BASE_URL = os.environ.get("THETADATA_URL", "http://127.0.0.1:25503")

# ─── Download parameters ────────────────────────────────────────────────────
SYMBOL = "SPXW"
INDEX_SYMBOLS = ["SPX", "VIX", "VVIX"]
START_DATE = "2019-01-01"
DATA_ROOT = Path(os.environ.get("THETADATA_ROOT", Path.home() / "thetadata")).expanduser()
COMPLETED_FILE = DATA_ROOT / "completed.txt"
ERRORS_LOG = DATA_ROOT / "logs" / "errors.log"
ZSTD_LEVEL = 9
ROW_GROUP_SIZE = 100_000
INTERVAL = "1m"  # options: quotes, greeks, IV
INDEX_INTERVAL = "1m"  # bulk history requires >= 1m

# ─── Rate limiting / retry ──────────────────────────────────────────────────
SLEEP_BETWEEN_REQUESTS = 0.15  # seconds
MAX_RETRIES = 3
BACKOFF = [1, 4, 16]  # seconds between retries
RATE_LIMIT_SLEEP = 60  # seconds on 429
MAX_CONCURRENT = 10  # optimal: +25% over 8, terminal not overloaded
MAX_MONTH_PARALLEL = 4  # months downloaded in parallel for daily datasets

# ─── Schemas (column → pandas dtype) ────────────────────────────────────────
# Used for empty parquet files (preserves schema for skip-detection) and casting.

SCHEMA_INDEX = OrderedDict(
    [
        ("timestamp", "datetime64[ms]"),
        ("price", "float32"),
    ]
)

SCHEMA_OI = OrderedDict(
    [
        ("symbol", "str"),
        ("expiration", "str"),
        ("strike", "float32"),
        ("right", "category"),
        ("timestamp", "datetime64[ms]"),
        ("open_interest", "Int32"),
    ]
)

SCHEMA_EOD = OrderedDict(
    [
        ("symbol", "str"),
        ("expiration", "str"),
        ("strike", "float32"),
        ("right", "category"),
        ("created", "datetime64[ms]"),
        ("last_trade", "datetime64[ms]"),
        ("open", "float32"),
        ("high", "float32"),
        ("low", "float32"),
        ("close", "float32"),
        ("volume", "Int32"),
        ("count", "Int32"),
        ("bid_size", "Int32"),
        ("bid_exchange", "Int32"),
        ("bid", "float32"),
        ("bid_condition", "Int32"),
        ("ask_size", "Int32"),
        ("ask_exchange", "Int32"),
        ("ask", "float32"),
        ("ask_condition", "Int32"),
    ]
)

SCHEMA_QUOTES = OrderedDict(
    [
        ("symbol", "str"),
        ("expiration", "str"),
        ("strike", "float32"),
        ("right", "category"),
        ("timestamp", "datetime64[ms]"),
        ("bid_size", "Int32"),
        ("bid_exchange", "Int32"),
        ("bid", "float32"),
        ("bid_condition", "Int32"),
        ("ask_size", "Int32"),
        ("ask_exchange", "Int32"),
        ("ask", "float32"),
        ("ask_condition", "Int32"),
    ]
)

SCHEMA_GREEKS = OrderedDict(
    [
        ("symbol", "str"),
        ("expiration", "str"),
        ("strike", "float32"),
        ("right", "category"),
        ("timestamp", "datetime64[ms]"),
        ("bid", "float32"),
        ("ask", "float32"),
        ("delta", "float32"),
        ("theta", "float32"),
        ("vega", "float32"),
        ("rho", "float32"),
        ("epsilon", "float32"),
        ("lambda", "float32"),
        ("implied_vol", "float32"),
        ("iv_error", "float32"),
        ("underlying_timestamp", "datetime64[ms]"),
        ("underlying_price", "float32"),
    ]
)

SCHEMA_IV = OrderedDict(
    [
        ("symbol", "str"),
        ("expiration", "str"),
        ("strike", "float32"),
        ("right", "category"),
        ("timestamp", "datetime64[ms]"),
        ("bid", "float32"),
        ("bid_implied_vol", "float32"),
        ("midpoint", "float32"),
        ("implied_vol", "float32"),
        ("ask", "float32"),
        ("ask_implied_vol", "float32"),
        ("iv_error", "float32"),
        ("underlying_timestamp", "datetime64[ms]"),
        ("underlying_price", "float32"),
    ]
)

# ─── Dataset registry ───────────────────────────────────────────────────────
# Execution order: light → heavy (index, OI, EOD, quotes, greeks, IV)
# batch modes: "monthly" | "daily_Xdte" (X = 0, 1, 2, ..., 7)

DATASETS = [
    # ── Index (flat response, no expiration) ──
    {
        "key": "index/spx",
        "endpoint": "/v3/index/history/price",
        "batch": "monthly",
        "response_type": "index",
        "schema": SCHEMA_INDEX,
        "extra_params": {"symbol": "SPX", "interval": INDEX_INTERVAL},
    },
    {
        "key": "index/vix",
        "endpoint": "/v3/index/history/price",
        "batch": "monthly",
        "response_type": "index",
        "schema": SCHEMA_INDEX,
        "extra_params": {"symbol": "VIX", "interval": INDEX_INTERVAL},
    },
    {
        "key": "index/vvix",
        "endpoint": "/v3/index/history/price",
        "batch": "monthly",
        "response_type": "index",
        "schema": SCHEMA_INDEX,
        "extra_params": {"symbol": "VVIX", "interval": INDEX_INTERVAL},
    },
    # ── Options OI (monthly, wildcard expiration, no interval) ──
    {
        "key": "options/oi",
        "endpoint": "/v3/option/history/open_interest",
        "batch": "monthly",
        "response_type": "options",
        "schema": SCHEMA_OI,
        "extra_params": {"symbol": SYMBOL, "expiration": "*"},
    },
    # ── Options EOD (monthly, wildcard expiration, no interval) ──
    {
        "key": "options/eod",
        "endpoint": "/v3/option/history/eod",
        "batch": "monthly",
        "response_type": "options",
        "schema": SCHEMA_EOD,
        "extra_params": {"symbol": SYMBOL, "expiration": "*"},
    },
    # ── Quotes 0DTE / 1DTE (day-by-day, explicit expiration) ──
    {
        "key": "options/quotes/0dte",
        "endpoint": "/v3/option/history/quote",
        "batch": "daily_0dte",
        "response_type": "options",
        "schema": SCHEMA_QUOTES,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/quotes/1dte",
        "endpoint": "/v3/option/history/quote",
        "batch": "daily_1dte",
        "response_type": "options",
        "schema": SCHEMA_QUOTES,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    # ── Greeks 0DTE / 1DTE (day-by-day, no wildcard allowed) ──
    {
        "key": "options/greeks/0dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_0dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/1dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_1dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    # ── Greeks 2DTE through 7DTE (day-by-day, first_order only) ──
    {
        "key": "options/greeks/2dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_2dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/3dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_3dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/4dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_4dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/5dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_5dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/6dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_6dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/greeks/7dte",
        "endpoint": "/v3/option/history/greeks/first_order",
        "batch": "daily_7dte",
        "response_type": "options",
        "schema": SCHEMA_GREEKS,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    # ── IV 0DTE / 1DTE (day-by-day, no wildcard allowed) ──
    {
        "key": "options/iv/0dte",
        "endpoint": "/v3/option/history/greeks/implied_volatility",
        "batch": "daily_0dte",
        "response_type": "options",
        "schema": SCHEMA_IV,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
    {
        "key": "options/iv/1dte",
        "endpoint": "/v3/option/history/greeks/implied_volatility",
        "batch": "daily_1dte",
        "response_type": "options",
        "schema": SCHEMA_IV,
        "extra_params": {"symbol": SYMBOL, "interval": INTERVAL},
    },
]


def empty_dataframe(schema):
    """Create an empty DataFrame with correct dtypes for schema."""
    import pandas as pd

    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
