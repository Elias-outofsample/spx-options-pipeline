"""Synthetic market generator: a self-consistent stand-in for the vendor feed.

It writes raw files in the same layout and schema as the real feed, so the whole
pipeline — ingestion, store, queries, cleaning, point-in-time loading — runs end to
end on a laptop with no data subscription.

What is generated, per business day:

* SPX 1-minute bars from a geometric Brownian motion whose volatility follows a
  mean-reverting VIX path; VVIX alongside;
* an SPXW option chain (0DTE, 1DTE, … as requested) priced with Black-Scholes on a
  skewed volatility smile, with first-order Greeks computed analytically and a
  bid/ask spread around the model price;
* a daily volatility-regime file derived from the VIX/VVIX closes.

It also reproduces the defects the cleaning rules exist for, so they have something
to remove: the corrupt 09:30 option bar (underlying price 0, IV overflow, zero bid),
OPRA test contracts expiring in 1882, and stale zero prints on VIX.

The numbers are synthetic. They exercise the plumbing; they carry no market signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

MINUTES_PER_SESSION = 390  # 09:30 … 15:59
_OPEN = (9, 30)
_SQRT2 = np.sqrt(2.0)
_YEAR_DAYS = 365.0


@dataclass(frozen=True)
class SyntheticSpec:
    start: date = date(2024, 1, 2)
    end: date = date(2024, 1, 12)
    spx0: float = 4750.0
    vix0: float = 14.0
    strikes_each_side: int = 20
    strike_step: float = 5.0
    dtes: tuple[int, ...] = (0, 1)
    seed: int = 7


def business_days(start: date, end: date) -> list[date]:
    """Weekdays between ``start`` and ``end`` inclusive (no holiday calendar)."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate_raw_dataset(raw_root: Path | str, spec: SyntheticSpec | None = None) -> dict[str, int]:
    """Write a synthetic raw feed under ``raw_root``. Returns row counts per dataset."""
    spec = spec or SyntheticSpec()
    raw_root = Path(raw_root)
    rng = np.random.default_rng(spec.seed)
    days = business_days(spec.start, spec.end)
    counts: dict[str, int] = {}

    spx_close, vix_level = spec.spx0, spec.vix0
    daily_rows = []
    for i, d in enumerate(days):
        ts = _session_timestamps(d)
        vix_path = _ou_path(rng, vix_level, mean=15.0, speed=0.02, vol=0.08, n=len(ts))
        vvix_path = 85.0 + 3.0 * (vix_path - 15.0) + rng.normal(0, 0.6, len(ts))
        spx_path = _gbm_path(rng, spx_close, vix_path / 100.0, n=len(ts))

        _write(raw_root / "index" / "spx" / f"{d:%Y%m%d}.parquet", _ohlc(ts, spx_path, rng))
        vix_bars = _ohlc(ts, vix_path, rng)
        if i % 2 == 0:  # stale zero prints the feed occasionally emits
            closes = vix_bars["close"].to_numpy(zero_copy_only=False).copy()
            closes[[37, 38, 201]] = 0.0
            vix_bars = vix_bars.set_column(4, "close", pa.array(closes, pa.float32()))
        _write(raw_root / "index" / "vix" / f"{d:%Y%m%d}.parquet", vix_bars)
        _write(raw_root / "index" / "vvix" / f"{d:%Y%m%d}.parquet", _ohlc(ts, vvix_path, rng))
        for k in ("spx", "vix", "vvix"):
            counts[f"index/{k}"] = counts.get(f"index/{k}", 0) + len(ts)

        expiries = business_days(d, d + timedelta(days=14))
        for dte in spec.dtes:
            if dte >= len(expiries):
                continue
            chain = _option_chain(rng, ts, spx_path, vix_path, expiries[dte], spec, bogus=(i == 0))
            _write(raw_root / "options" / "greeks" / f"{dte}dte" / f"{d:%Y%m%d}.parquet", chain)
            counts[f"options/greeks/{dte}dte"] = counts.get(f"options/greeks/{dte}dte", 0) + len(
                chain
            )

        spx_close, vix_level = float(spx_path[-1]), float(vix_path[-1])
        daily_rows.append((d, float(vix_path[-1]), float(vvix_path[-1])))

    regime = _regime_frame(daily_rows)
    regime.write_csv(raw_root / "vol_regime_daily.csv")
    counts["vol_regime"] = regime.height
    return counts


# --------------------------------------------------------------------- paths


def _session_timestamps(d: date) -> list[datetime]:
    t0 = datetime(d.year, d.month, d.day, *_OPEN)
    return [t0 + timedelta(minutes=m) for m in range(MINUTES_PER_SESSION)]


def _ou_path(rng: np.random.Generator, x0: float, mean: float, speed: float, vol: float, n: int):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + speed * (mean - x[t - 1]) / 10 + vol * rng.normal()
    return np.clip(x, 9.0, 60.0)


def _gbm_path(rng: np.random.Generator, s0: float, sigma: np.ndarray, n: int) -> np.ndarray:
    dt = 1.0 / (252 * MINUTES_PER_SESSION)
    shocks = rng.normal(0.0, 1.0, n)
    log_ret = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * shocks
    log_ret[0] = 0.0
    return s0 * np.exp(np.cumsum(log_ret))


def _ohlc(ts: list[datetime], path: np.ndarray, rng: np.random.Generator) -> pa.Table:
    noise = np.abs(rng.normal(0, 0.0004, len(path))) * path
    opens = np.concatenate([[path[0]], path[:-1]])
    return pa.table(
        {
            "timestamp": pa.array(ts, pa.timestamp("ns")),
            "open": pa.array(opens, pa.float32()),
            "high": pa.array(np.maximum(opens, path) + noise, pa.float32()),
            "low": pa.array(np.minimum(opens, path) - noise, pa.float32()),
            "close": pa.array(path, pa.float32()),
        }
    )


# ------------------------------------------------------------- option chain


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorised."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * np.exp(-x * x))


def black_scholes(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    is_call: np.ndarray,
    rate: float = 0.05,
) -> dict[str, np.ndarray]:
    """Black-Scholes price and first-order Greeks (theta per day, vega per vol point)."""
    tau = np.maximum(tau, 1e-6)
    sqrt_t = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = np.exp(-rate * tau)
    call = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    put = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    pdf = _norm_pdf(d1)
    theta_call = -spot * pdf * vol / (2 * sqrt_t) - rate * strike * disc * _norm_cdf(d2)
    theta_put = -spot * pdf * vol / (2 * sqrt_t) + rate * strike * disc * _norm_cdf(-d2)
    return {
        "price": np.where(is_call, call, put),
        "delta": np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0),
        "theta": np.where(is_call, theta_call, theta_put) / _YEAR_DAYS,
        "vega": spot * pdf * sqrt_t / 100.0,
        "rho": np.where(
            is_call, strike * tau * disc * _norm_cdf(d2), -strike * tau * disc * _norm_cdf(-d2)
        )
        / 100.0,
        "lambda": np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
        * spot
        / np.maximum(np.where(is_call, call, put), 1e-9),
    }


def _smile(atm_vol: np.ndarray, log_moneyness: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Equity-style skew: puts richer than calls, steeper at short maturities."""
    scale = np.sqrt(np.maximum(tau, 1.0 / (252 * 390)))
    skew = -0.9 * log_moneyness / scale * 0.02
    curvature = 1.5 * (log_moneyness / scale) ** 2 * 0.01
    return np.clip(atm_vol * (1.0 + skew + curvature), 0.05, 3.0)


def _option_chain(
    rng: np.random.Generator,
    ts: list[datetime],
    spx: np.ndarray,
    vix: np.ndarray,
    expiry: date,
    spec: SyntheticSpec,
    bogus: bool,
) -> pa.Table:
    atm = round(float(spx[0]) / spec.strike_step) * spec.strike_step
    offsets = np.arange(-spec.strikes_each_side, spec.strikes_each_side + 1) * spec.strike_step
    strikes = atm + offsets
    n_t, n_k = len(ts), len(strikes)

    t_idx = np.repeat(np.arange(n_t), n_k * 2)
    k = np.tile(np.repeat(strikes, 2), n_t)
    is_call = np.tile(np.array([True, False]), n_t * n_k)
    spot = spx[t_idx]
    settle = datetime(expiry.year, expiry.month, expiry.day, 16, 0)
    tau = np.array([(settle - ts[i]).total_seconds() for i in t_idx]) / (365.0 * 86400.0)
    vol = _smile(vix[t_idx] / 100.0, np.log(k / spot), tau)
    g = black_scholes(spot, k, tau, vol, is_call)

    price = np.maximum(g["price"], 0.05)
    half_spread = np.maximum(0.05, 0.03 * price) / 2
    bid = np.maximum(price - half_spread + rng.normal(0, 0.01, price.size), 0.0)
    ask = price + half_spread
    underlying = spot.copy()
    iv = vol.copy()
    iv_err = np.zeros_like(vol)

    first_bar = t_idx == 0  # the feed's corrupt opening bar
    underlying[first_bar] = 0.0
    iv[first_bar] = 21474.8
    iv_err[first_bar] = -0.99999
    bid[first_bar] = 0.0

    expirations = np.full(k.size, expiry.isoformat(), dtype=object)
    if bogus:  # OPRA test contracts, printed mid-session
        expirations[t_idx == n_t // 2] = "1882-01-01"

    ts_arr = [ts[i] for i in t_idx]
    return pa.table(
        {
            "symbol": pa.array(["SPXW"] * k.size),
            "expiration": pa.array(expirations.tolist(), pa.utf8()),
            "strike": pa.array(k, pa.float32()),
            "right": pa.array(np.where(is_call, "CALL", "PUT").tolist(), pa.utf8()),
            "timestamp": pa.array(ts_arr, pa.timestamp("ns")),
            "bid": pa.array(bid, pa.float32()),
            "ask": pa.array(ask, pa.float32()),
            "delta": pa.array(g["delta"], pa.float32()),
            "theta": pa.array(g["theta"], pa.float32()),
            "vega": pa.array(g["vega"], pa.float32()),
            "rho": pa.array(g["rho"], pa.float32()),
            "epsilon": pa.array(np.zeros(k.size), pa.float32()),
            "lambda": pa.array(g["lambda"], pa.float32()),
            "implied_vol": pa.array(iv, pa.float32()),
            "iv_error": pa.array(iv_err, pa.float32()),
            "underlying_timestamp": pa.array([t.isoformat() for t in ts_arr], pa.utf8()),
            "underlying_price": pa.array(underlying, pa.float32()),
        }
    )


# ------------------------------------------------------------------- regime


def _regime_frame(daily: list[tuple[date, float, float]]) -> pl.DataFrame:
    df = pl.DataFrame(daily, schema=["date", "vix", "vvix"], orient="row")
    df = df.with_columns((pl.col("vvix") / pl.col("vix")).alias("vov_ratio"))

    def pct(col: str) -> pl.Expr:
        return (pl.col(col).rank("average") / pl.len()).alias(f"{col}_pct_252")

    def z(col: str, out: str) -> pl.Expr:
        return ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).fill_nan(0.0).alias(out)

    df = df.with_columns(
        pct("vix"),
        pct("vvix"),
        pct("vov_ratio"),
        z("vix", "vix_z_20"),
        z("vvix", "vvix_z_20"),
        z("vov_ratio", "vov_z_20"),
    )

    def label(col: str) -> pl.Expr:
        return (
            pl.when(pl.col(col) < 0.25)
            .then(pl.lit("low"))
            .when(pl.col(col) < 0.75)
            .then(pl.lit("mid"))
            .when(pl.col(col) < 0.95)
            .then(pl.lit("high"))
            .otherwise(pl.lit("extreme"))
        )

    df = df.with_columns(
        label("vix_pct_252").alias("vix_regime"),
        label("vvix_pct_252").alias("vvix_regime"),
        label("vov_ratio_pct_252").alias("vov_regime"),
    )
    score = (pl.col("vix_pct_252") * 3).floor() + (pl.col("vvix_pct_252") * 3).floor()
    return df.with_columns(
        score.cast(pl.Int8).alias("regime_score"),
        pl.concat_str(
            [
                pl.when(pl.col("vix_pct_252") >= 0.5)
                .then(pl.lit("hiVIX"))
                .otherwise(pl.lit("loVIX")),
                pl.when(pl.col("vvix_pct_252") >= 0.5)
                .then(pl.lit("hiVVIX"))
                .otherwise(pl.lit("loVVIX")),
            ],
            separator="/",
        ).alias("regime_2x2"),
        pl.when(score >= 4)
        .then(pl.lit("defensive"))
        .when(score >= 2)
        .then(pl.lit("neutral"))
        .otherwise(pl.lit("carry"))
        .alias("desk_bucket"),
    )


def _write(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy")
