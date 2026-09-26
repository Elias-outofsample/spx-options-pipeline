from datetime import date, datetime

import polars as pl
import pytest

from spx_pipeline.cleaning import apply_cleaning_rules, resample_ohlc
from spx_pipeline.registry import CleaningRule


def _make_ohlc_df(n=10, base_price=100.0, include_zeros=False):
    """Helper to create a simple OHLC DataFrame."""
    timestamps = [datetime(2024, 1, 2, 9, 30 + i) for i in range(n)]
    closes = [base_price + i * 0.1 for i in range(n)]
    if include_zeros:
        closes[0] = 0.0
        closes[3] = 0.0
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [base_price] * n,
            "high": [base_price + 1] * n,
            "low": [base_price - 1] * n,
            "close": closes,
        }
    )


def _make_greeks_df():
    """Helper to create a Greeks DataFrame with corrupt opening bar."""
    rows = []
    for i in range(5):
        ts = datetime(2024, 1, 2, 9, 30 + i)
        rows.append(
            {
                "timestamp": ts,
                "strike": 4750.0,
                "right": "CALL",
                "bid": 0.0 if i == 0 else 5.0,
                "ask": 5.5,
                "delta": 0.5,
                "implied_vol": 21474.8 if i == 0 else 0.25,
                "iv_error": -0.99999 if i == 0 else 0.0,
                "underlying_price": 0.0 if i == 0 else 4750.0,
            }
        )
    return pl.DataFrame(rows)


def test_drop_opening_bar():
    df = _make_greeks_df()
    rules = [CleaningRule(name="drop_opening_bar", params={"time": "09:30:00"})]
    result = apply_cleaning_rules(df, rules)
    # 09:30 bar should be removed
    assert len(result) == 4
    min_minute = result.get_column("timestamp").dt.minute().min()
    assert min_minute == 31


def test_filter_illiquid():
    df = _make_greeks_df()
    rules = [
        CleaningRule(
            name="filter_illiquid",
            params={
                "bid_gt": 0.0,
                "implied_vol_lte": 100.0,
                "underlying_price_gt": 0.0,
            },
        )
    ]
    result = apply_cleaning_rules(df, rules)
    # Row 0 has bid=0, IV=21474, underlying=0 → removed
    assert len(result) == 4


def test_ffill_zero_close_forward_only():
    """Forward-fill zeros but NEVER backward-fill (anti look-ahead)."""
    df = _make_ohlc_df(n=5, include_zeros=True)
    rules = [CleaningRule(name="ffill_zero_close", params={"enabled": True})]
    result = apply_cleaning_rules(df, rules)

    # First bar was zero and has no predecessor → should remain null
    first_close = result.get_column("close")[0]
    assert first_close is None, "First zero bar should stay null (no backward fill)"

    # Bar at index 3 was zero → should be forward-filled from bar 2
    bar3_close = result.get_column("close")[3]
    bar2_close = result.get_column("close")[2]
    assert bar3_close == bar2_close


def test_ffill_disabled():
    df = _make_ohlc_df(n=5, include_zeros=True)
    rules = [CleaningRule(name="ffill_zero_close", params={"enabled": False})]
    result = apply_cleaning_rules(df, rules)
    # Zeros should remain
    assert result.get_column("close")[0] == 0.0


def test_add_mid():
    df = pl.DataFrame(
        {
            "bid": [1.0, 2.0],
            "ask": [1.5, 2.5],
        }
    )
    rules = [CleaningRule(name="add_mid", params={})]
    result = apply_cleaning_rules(df, rules)
    assert "mid" in result.columns
    assert result.get_column("mid")[0] == pytest.approx(1.25, abs=0.01)


def test_valid_from():
    df = pl.DataFrame(
        {
            "date": [date(2023, 3, 1), date(2023, 4, 13), date(2023, 5, 1)],
            "vix": [15.0, 16.0, 17.0],
        }
    )
    rules = [CleaningRule(name="valid_from", params={"value": "2023-04-13"})]
    result = apply_cleaning_rules(df, rules)
    assert len(result) == 2
    assert result.get_column("date")[0] == date(2023, 4, 13)


def test_resample_ohlc():
    timestamps = [datetime(2024, 1, 2, 9, 30 + i) for i in range(10)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
        }
    )
    result = resample_ohlc(df, "5m")
    assert len(result) == 2
    # First bar labeled at 09:30 (left label)
    assert result.get_column("timestamp")[0].hour == 9
    assert result.get_column("timestamp")[0].minute == 30
