from datetime import date

import pyarrow as pa
import pytest

from spx_pipeline.params import QueryParams
from spx_pipeline.query import QueryEngine
from spx_pipeline.registry import Registry


@pytest.fixture
def engine(tmp_store, tmp_registry):
    registry = Registry(tmp_registry)
    return QueryEngine(tmp_store.parent, registry)


def test_query_spx(engine):
    params = QueryParams(
        dataset="spx_ohlc",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
    )
    result = engine.query(params)
    assert isinstance(result, pa.Table)
    assert result.num_rows > 0
    assert "close" in result.column_names


def test_query_with_columns(engine):
    params = QueryParams(
        dataset="spx_ohlc",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        columns=("timestamp", "close"),
    )
    result = engine.query(params)
    assert set(result.column_names) == {"timestamp", "close"}


def test_query_with_filters(engine):
    params = QueryParams(
        dataset="greeks_0dte",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        filters=(("right", "=", "CALL"),),
    )
    result = engine.query(params)
    if result.num_rows > 0:
        rights = result.column("right").to_pylist()
        assert all(r == "CALL" for r in rights)


def test_query_empty_range(engine):
    params = QueryParams(
        dataset="spx_ohlc",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 1),
    )
    result = engine.query(params)
    assert result.num_rows == 0


def test_query_date_range(engine):
    params = QueryParams(
        dataset="spx_ohlc",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
    )
    result = engine.query(params)
    assert result.num_rows > 0
