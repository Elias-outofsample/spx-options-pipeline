from datetime import date

import pyarrow as pa
import pytest

from spx_pipeline.cache import FeatherCache
from spx_pipeline.params import QueryParams


@pytest.fixture
def cache(tmp_path):
    return FeatherCache(tmp_path / "cache", max_cache_size_mb=10)


@pytest.fixture
def params():
    return QueryParams(
        dataset="spx_ohlc",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
    )


@pytest.fixture
def sample_table():
    return pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_put_and_get(cache, params, sample_table):
    cache.put(params, sample_table)
    result = cache.get(params)
    assert result is not None
    assert result.num_rows == 3


def test_cache_miss(cache, params):
    assert cache.get(params) is None


def test_different_params_different_cache(cache, sample_table):
    p1 = QueryParams(dataset="spx_ohlc", start_date=date(2024, 1, 2), end_date=date(2024, 1, 2))
    p2 = QueryParams(dataset="spx_ohlc", start_date=date(2024, 1, 3), end_date=date(2024, 1, 3))

    cache.put(p1, sample_table)
    assert cache.get(p1) is not None
    assert cache.get(p2) is None


def test_invalidate_dataset(cache, params, sample_table):
    cache.put(params, sample_table)
    assert cache.get(params) is not None

    removed = cache.invalidate("spx_ohlc")
    assert removed == 1
    assert cache.get(params) is None


def test_invalidate_all(cache, sample_table):
    p1 = QueryParams(dataset="spx_ohlc", start_date=date(2024, 1, 2), end_date=date(2024, 1, 2))
    p2 = QueryParams(dataset="vix_ohlc", start_date=date(2024, 1, 2), end_date=date(2024, 1, 2))

    cache.put(p1, sample_table)
    cache.put(p2, sample_table)
    removed = cache.invalidate()
    assert removed == 2


def test_size_limit_eviction(tmp_path, sample_table):
    cache = FeatherCache(tmp_path / "cache", max_cache_size_mb=0)  # 0 MB limit

    params = QueryParams(dataset="spx_ohlc", start_date=date(2024, 1, 2), end_date=date(2024, 1, 2))
    cache.put(params, sample_table)
    # After eviction, the entry should have been removed due to 0 MB limit
    # (the put writes then evicts)
