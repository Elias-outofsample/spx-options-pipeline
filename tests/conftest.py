from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def sample_date():
    return date(2024, 1, 2)


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def registry_path(project_root):
    return project_root / "src" / "spx_pipeline" / "registry.yaml"


# ---------------------------------------------------------------------------
# Synthetic data fixtures for unit tests (no real data required)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """Create a minimal store with synthetic data for unit testing."""
    store_root = tmp_path / "store"

    # --- SPX OHLC ---
    _write_synthetic_ohlc(
        store_root / "spx_ohlc" / "year=2024" / "month=01" / "20240102.parquet",
        date(2024, 1, 2),
        base_price=4750.0,
    )
    _write_synthetic_ohlc(
        store_root / "spx_ohlc" / "year=2024" / "month=01" / "20240103.parquet",
        date(2024, 1, 3),
        base_price=4760.0,
    )

    # --- VIX OHLC ---
    _write_synthetic_ohlc(
        store_root / "vix_ohlc" / "year=2024" / "month=01" / "20240102.parquet",
        date(2024, 1, 2),
        base_price=13.5,
        include_zeros=True,
    )

    # --- Greeks ---
    _write_synthetic_greeks(
        store_root / "greeks_0dte" / "year=2024" / "month=01" / "20240102.parquet",
        date(2024, 1, 2),
        underlying_price=4750.0,
    )

    # --- Regime ---
    _write_synthetic_regime(store_root / "vol_regime" / "vol_regime.parquet")

    return store_root


@pytest.fixture
def tmp_registry(tmp_path, project_root):
    """Copy the real registry.yaml to tmp_path."""
    src = project_root / "src" / "spx_pipeline" / "registry.yaml"
    dst = tmp_path / "registry.yaml"
    dst.write_text(src.read_text())
    return dst


@pytest.fixture
def loader(tmp_store, tmp_registry):
    """DataLoader backed by synthetic data."""
    from spx_pipeline.loader import DataLoader

    return DataLoader(
        store_root=tmp_store.parent,
        registry_path=tmp_registry,
        cache_dir=tmp_store.parent / ".cache",
    )


@pytest.fixture
def pit_loader(loader):
    """PointInTimeLoader with cutoff mid-2024."""
    from spx_pipeline.pit import PointInTimeLoader

    return PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _write_synthetic_ohlc(
    path: Path, d: date, base_price: float, include_zeros: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bars = 391
    timestamps = [
        datetime(d.year, d.month, d.day, 9, 30) + __import__("datetime").timedelta(minutes=i)
        for i in range(n_bars)
    ]
    rng = np.random.default_rng(seed=int(d.strftime("%Y%m%d")))
    prices = base_price + np.cumsum(rng.normal(0, 0.5, n_bars))
    opens = prices.astype(np.float32)
    highs = (prices + rng.uniform(0, 1, n_bars)).astype(np.float32)
    lows = (prices - rng.uniform(0, 1, n_bars)).astype(np.float32)
    closes = (prices + rng.normal(0, 0.2, n_bars)).astype(np.float32)

    if include_zeros:
        closes[0] = 0.0
        closes[5] = 0.0

    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
            "open": pa.array(opens),
            "high": pa.array(highs),
            "low": pa.array(lows),
            "close": pa.array(closes),
        }
    )
    pq.write_table(table, str(path), compression="zstd", compression_level=3)


def _write_synthetic_greeks(path: Path, d: date, underlying_price: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strikes = [underlying_price + s for s in range(-50, 55, 5)]
    rights = ["CALL", "PUT"]
    n_bars = 391
    rows = []

    rng = np.random.default_rng(seed=int(d.strftime("%Y%m%d")))

    for i in range(n_bars):
        ts = datetime(d.year, d.month, d.day, 9, 30) + __import__("datetime").timedelta(minutes=i)
        for strike in strikes:
            for right in rights:
                bid = max(0.0, rng.normal(5.0, 2.0))
                ask = bid + rng.uniform(0.05, 0.5)
                delta = 0.5 + rng.normal(0, 0.2) if right == "CALL" else -0.5 + rng.normal(0, 0.2)
                iv = rng.uniform(0.1, 0.8)
                iv_error = 0.0

                # Corrupt opening bar: underlying_price=0, IV overflow
                up = underlying_price if i > 0 else 0.0
                if i == 0:
                    iv = 21474.8
                    iv_error = -0.99999
                    bid = 0.0

                rows.append(
                    {
                        "symbol": "SPXW",
                        "expiration": d.isoformat(),
                        "strike": float(strike),
                        "right": right,
                        "timestamp": ts,
                        "bid": float(bid),
                        "ask": float(ask),
                        "delta": float(delta),
                        "theta": float(rng.normal(-5, 2)),
                        "vega": float(rng.uniform(0.1, 3.0)),
                        "rho": float(rng.normal(0, 0.1)),
                        "epsilon": float(rng.normal(0, 0.01)),
                        "lambda": float(rng.normal(10, 5)),
                        "implied_vol": float(iv),
                        "iv_error": float(iv_error),
                        "underlying_timestamp": ts.isoformat(),
                        "underlying_price": float(up),
                    }
                )

    df = pl.DataFrame(rows)
    table = df.to_arrow()
    pq.write_table(table, str(path), compression="zstd", compression_level=3)


def _write_synthetic_regime(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = []
    d = date(2023, 2, 1)
    while d <= date(2024, 12, 31):
        if d.weekday() < 5:
            dates.append(d)
        d += __import__("datetime").timedelta(days=1)

    rng = np.random.default_rng(seed=42)
    n = len(dates)
    buckets = rng.choice(["CALM", "OTHER", "FRAGILE", "PANIC"], size=n, p=[0.5, 0.25, 0.15, 0.1])

    df = pl.DataFrame(
        {
            "date": dates,
            "vix": rng.uniform(12, 35, n).tolist(),
            "vvix": rng.uniform(70, 140, n).tolist(),
            "vov_ratio": rng.uniform(3, 8, n).tolist(),
            "vix_pct_252": [None] * 49 + rng.uniform(0, 1, n - 49).tolist(),
            "vvix_pct_252": [None] * 49 + rng.uniform(0, 1, n - 49).tolist(),
            "vov_ratio_pct_252": [None] * 49 + rng.uniform(0, 1, n - 49).tolist(),
            "vix_z_20": rng.normal(0, 1, n).tolist(),
            "vvix_z_20": rng.normal(0, 1, n).tolist(),
            "vov_z_20": rng.normal(0, 1, n).tolist(),
            "vix_regime": [None] * 49 + rng.choice(["LOW", "MID", "HIGH"], n - 49).tolist(),
            "vvix_regime": [None] * 49 + rng.choice(["LOW", "MID", "HIGH"], n - 49).tolist(),
            "vov_regime": [None] * 49 + rng.choice(["LOW", "MID", "HIGH"], n - 49).tolist(),
            "regime_score": rng.integers(0, 4, n).tolist(),
            "regime_2x2": rng.choice(
                ["LOW_VOL__LOW_VOV", "HIGH_VOL__HIGH_VOV", "MID_VOL__MID_VOV"], n
            ).tolist(),
            "desk_bucket": buckets.tolist(),
        }
    )
    table = df.to_arrow()
    pq.write_table(table, str(path), compression="zstd", compression_level=3)
