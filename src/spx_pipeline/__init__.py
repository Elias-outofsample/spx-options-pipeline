"""spx_pipeline — a point-in-time data lake for SPX/SPXW options research.

Raw vendor files are ingested once into a Hive-partitioned Parquet store, queried
with DuckDB (predicate pushdown, bounded memory), cached as memory-mapped Arrow, and
served as Polars DataFrames — with look-ahead guards on every path that feeds a
backtest.

Quick start::

    from spx_pipeline import DataLoader, PointInTimeLoader

    loader = DataLoader(store_root="data")
    chain = loader.load("greeks_0dte", "2024-01-03",
                        filters=[("right", "=", "PUT"), ("strike", ">=", 4700)])
    pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))
"""

from .cleaning import apply_cleaning_rules, resample_ohlc
from .loader import DataLoader
from .params import Granularity, QueryParams
from .pit import PointInTimeLoader
from .query import QueryEngine
from .registry import DEFAULT_REGISTRY, CleaningRule, DatasetConfig, Registry
from .store import DataStore

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_REGISTRY",
    "CleaningRule",
    "DataLoader",
    "DataStore",
    "DatasetConfig",
    "Granularity",
    "PointInTimeLoader",
    "QueryEngine",
    "QueryParams",
    "Registry",
    "__version__",
    "apply_cleaning_rules",
    "resample_ohlc",
]
