"""
Tests for PointInTimeLoader — anti-lookahead bias enforcement.

Each test constructs a scenario where a naive implementation would leak
future data into the past.
"""

from datetime import date, datetime

import polars as pl
import pyarrow.parquet as pq
import pytest

from spx_pipeline.pit import PointInTimeLoader


class TestCutoffEnforcement:
    def test_cutoff_clamps_end_date(self, pit_loader, tmp_store):
        """Requesting data past cutoff returns only data up to cutoff."""
        # pit_loader has cutoff 2024-06-30, but we only have data for 2024-01-02/03
        df = pit_loader.load("spx_ohlc", ("2024-01-01", "2024-12-31"))
        if not df.is_empty():
            max_ts = df.get_column("timestamp").max()
            assert max_ts.date() <= pit_loader.cutoff_date

    def test_cutoff_returns_empty_for_future_request(self, pit_loader):
        """Requesting dates entirely after cutoff returns empty."""
        df = pit_loader.load("spx_ohlc", (date(2025, 1, 1), date(2025, 1, 1)))
        assert df.is_empty()

    def test_cutoff_partial_range(self, loader):
        """A range spanning the cutoff returns the pre-cutoff part, and only that."""
        pit = PointInTimeLoader(loader, cutoff_date=date(2024, 1, 2))
        df = pit.load("spx_ohlc", ("2024-01-02", "2024-01-03"))
        assert not df.is_empty()
        assert df.get_column("timestamp").max().date() == date(2024, 1, 2)


class TestRegimeLag:
    def test_regime_lag_zero_rejected(self, loader):
        """PointInTimeLoader must refuse regime_lag_days=0."""
        with pytest.raises(ValueError, match="regime_lag_days must be >= 1"):
            PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30), regime_lag_days=0)

    def test_regime_is_lagged_one_session(self, loader):
        """Bars of day T carry the regime computed at the close of T-1, never T's own."""
        aligned = loader.load_aligned(
            ("2024-01-02", "2024-01-03"), datasets=["spx_ohlc", "vol_regime"]
        )
        regime = loader.load("vol_regime", ("2024-01-01", "2024-01-03"), apply_cleaning=False)
        by_date = dict(zip(regime["date"].to_list(), regime["vix"].to_list(), strict=True))
        bar = aligned.filter(pl.col("timestamp").dt.date() == date(2024, 1, 3)).row(0, named=True)
        assert bar["vix"] == pytest.approx(by_date[date(2024, 1, 2)], rel=1e-6)
        assert bar["vix"] != pytest.approx(by_date[date(2024, 1, 3)], rel=1e-6)


class TestFeatureLag:
    def test_feature_lag_shifts_correctly(self, pit_loader):
        """Feature lag must produce null for the first N bars."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 31 + i) for i in range(5)],
                "delta": [0.5, 0.48, 0.52, 0.49, 0.51],
                "theta": [-5.0, -5.1, -4.9, -5.2, -5.0],
            }
        )
        result = pit_loader.with_feature_lag(df, ["delta", "theta"], lag=1)

        assert "delta_lag1" in result.columns
        assert "theta_lag1" in result.columns
        # First bar should have null lagged values
        assert result.get_column("delta_lag1")[0] is None
        assert result.get_column("theta_lag1")[0] is None
        # Second bar should have the first bar's value
        assert result.get_column("delta_lag1")[1] == pytest.approx(0.5)

    def test_feature_lag_zero_no_shift(self, loader):
        """Feature lag of 0 should not add any columns."""
        pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30), feature_lag_bars=0)
        df = pl.DataFrame({"delta": [0.5, 0.48]})
        result = pit.with_feature_lag(df, ["delta"])
        assert "delta_lag0" not in result.columns


class TestOpeningBarRemoval:
    def test_opening_bar_removed_from_greeks(self, loader):
        """09:30 bar with corrupt underlying_price must be removed."""
        df = loader.load("greeks_0dte", "2024-01-02", apply_cleaning=True)
        if not df.is_empty():
            min_time = df.get_column("timestamp").dt.time().min()
            assert min_time.hour >= 9 and min_time.minute >= 31, "09:30 bar was not removed"


class TestDTENotStored:
    def test_dte_not_in_stored_parquet(self, tmp_store):
        """The stored Parquet files must NOT contain a 'dte' column."""
        import glob

        files = glob.glob(str(tmp_store / "greeks_0dte" / "**" / "*.parquet"), recursive=True)
        for f in files:
            schema = pq.read_schema(f)
            assert "dte" not in schema.names, (
                f"DTE found stored in {f} — should be computed at query time"
            )


class TestNoBackwardFill:
    def test_vix_zero_print_takes_the_previous_value(self, loader):
        """The synthetic VIX has a zero print at 09:35: it must repeat the 09:34 close."""
        raw = loader.load("vix_ohlc", "2024-01-02", apply_cleaning=False, use_cache=False)
        clean = loader.load("vix_ohlc", "2024-01-02", apply_cleaning=True)
        t_zero, t_prev = datetime(2024, 1, 2, 9, 35), datetime(2024, 1, 2, 9, 34)
        assert raw.filter(pl.col("timestamp") == t_zero)["close"][0] == 0.0
        prev = raw.filter(pl.col("timestamp") == t_prev)["close"][0]
        assert clean.filter(pl.col("timestamp") == t_zero)["close"][0] == pytest.approx(prev)

    def test_leading_zero_is_never_backfilled(self):
        """A zero at the very first bar has no past: it stays missing."""
        from spx_pipeline.cleaning import apply_cleaning_rules
        from spx_pipeline.registry import CleaningRule

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 31 + i) for i in range(3)],
                "close": [0.0, 13.0, 13.5],
            }
        )
        out = apply_cleaning_rules(df, [CleaningRule("ffill_zero_close", {"enabled": True})])
        assert out["close"].to_list() == [None, 13.0, 13.5]


class TestOutputTypes:
    def test_output_is_polars(self, loader):
        """Pipeline must return Polars DataFrames, never pandas."""
        df = loader.load("spx_ohlc", "2024-01-02")
        assert isinstance(df, pl.DataFrame)

    def test_float64_upcast(self, loader):
        """Numeric columns should be upcast to Float64."""
        df = loader.load("spx_ohlc", "2024-01-02", upcast_float64=True)
        if not df.is_empty():
            assert df["close"].dtype == pl.Float64

    def test_float32_when_no_upcast(self, loader):
        """Without upcast, columns remain Float32."""
        df = loader.load("spx_ohlc", "2024-01-02", upcast_float64=False)
        if not df.is_empty():
            assert df["close"].dtype == pl.Float32
