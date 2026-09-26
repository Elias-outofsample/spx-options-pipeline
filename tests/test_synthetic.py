from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from spx_pipeline.synthetic import SyntheticSpec, black_scholes, generate_raw_dataset


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    spec = SyntheticSpec(
        start=date(2024, 1, 2), end=date(2024, 1, 4), strikes_each_side=5, dtes=(0,)
    )
    generate_raw_dataset(root, spec)
    return root


def test_put_call_parity():
    rng = np.random.default_rng(0)
    spot, strike = rng.uniform(50, 150, 200), rng.uniform(50, 150, 200)
    tau, vol, rate = rng.uniform(0.01, 2, 200), rng.uniform(0.05, 1, 200), 0.05
    call = black_scholes(spot, strike, tau, vol, np.full(200, True), rate)["price"]
    put = black_scholes(spot, strike, tau, vol, np.full(200, False), rate)["price"]
    np.testing.assert_allclose(call - put, spot - strike * np.exp(-rate * tau), atol=1e-6)


def test_textbook_price_and_greeks():
    g = black_scholes(
        np.array([100.0]), np.array([100.0]), np.array([0.5]), np.array([0.2]), np.array([True])
    )
    assert g["price"][0] == pytest.approx(6.8887, abs=1e-3)
    assert g["delta"][0] == pytest.approx(0.5977, abs=1e-3)
    assert g["vega"][0] > 0 and g["theta"][0] < 0


def test_layout_matches_the_registry(raw_root):
    for rel in (
        "index/spx/20240102.parquet",
        "index/vix/20240103.parquet",
        "options/greeks/0dte/20240104.parquet",
        "vol_regime_daily.csv",
    ):
        assert (raw_root / rel).exists(), rel


def test_feed_defects_are_reproduced(raw_root):
    chain = pl.read_parquet(raw_root / "options/greeks/0dte/20240102.parquet")
    opening = chain.filter(pl.col("timestamp") == pl.col("timestamp").min())
    assert (opening["underlying_price"] == 0).all() and (opening["implied_vol"] > 100).all()
    assert chain.filter(pl.col("expiration") == "1882-01-01").height == 2 * (2 * 5 + 1)
    assert set(chain["right"].unique()) == {"CALL", "PUT"}
