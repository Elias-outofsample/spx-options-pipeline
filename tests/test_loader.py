from datetime import date

import polars as pl


def test_load_spx_ohlc(loader, sample_date):
    df = loader.load("spx_ohlc", sample_date)
    assert isinstance(df, pl.DataFrame)
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert not df.is_empty()


def test_load_returns_float64_by_default(loader, sample_date):
    df = loader.load("spx_ohlc", sample_date)
    if not df.is_empty():
        assert df["close"].dtype == pl.Float64


def test_load_date_range(loader):
    df = loader.load("spx_ohlc", ("2024-01-02", "2024-01-03"))
    if not df.is_empty():
        dates = df.get_column("timestamp").dt.date().unique().sort().to_list()
        assert len(dates) >= 1


def test_cache_hit_returns_same_data(loader, sample_date):
    df1 = loader.load("spx_ohlc", sample_date, use_cache=True)
    df2 = loader.load("spx_ohlc", sample_date, use_cache=True)
    assert df1.equals(df2)


def test_load_with_filters(loader, sample_date):
    df = loader.load(
        "greeks_0dte",
        sample_date,
        filters=[("right", "=", "CALL")],
    )
    if not df.is_empty():
        rights = df.get_column("right").unique().to_list()
        assert all(r == "CALL" for r in rights)


def test_load_empty_date_returns_empty(loader):
    df = loader.load("spx_ohlc", date(2020, 1, 1))
    assert df.is_empty()


def test_invalidate_cache(loader, sample_date):
    loader.load("spx_ohlc", sample_date, use_cache=True)
    removed = loader.invalidate_cache("spx_ohlc")
    assert removed >= 1


def test_load_no_cleaning(loader, sample_date):
    df = loader.load("greeks_0dte", sample_date, apply_cleaning=False)
    if not df.is_empty():
        # Without cleaning, 09:30 bar should still be present
        bar_0930 = df.filter(
            (pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() == 30)
        )
        assert not bar_0930.is_empty(), "09:30 bar should be present without cleaning"


def test_load_with_cleaning(loader, sample_date):
    df = loader.load("greeks_0dte", sample_date, apply_cleaning=True)
    if not df.is_empty():
        # With cleaning, 09:30 bar should be removed
        bar_0930 = df.filter(
            (pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() == 30)
        )
        assert bar_0930.is_empty(), "09:30 bar should be removed by cleaning"
