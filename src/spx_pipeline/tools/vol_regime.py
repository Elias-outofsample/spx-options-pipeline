"""
Generate vol_regime dataset from VIX/VVIX daily close data.

Computes daily volatility regime classifications:
  - Percentile ranks (252-day rolling)
  - Z-scores (20-day rolling)
  - Regime labels (low/mid/high/extreme)
  - Composite regime_score and regime_2x2

Writes to store/vol_regime.parquet (single file, not Hive-partitioned).

Usage:
    spx-pipeline regime
    spx-pipeline regime --store-root /path/to/store
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _daily_close(store_dir: Path) -> pl.DataFrame:
    """Load Hive-partitioned OHLC and extract daily close."""
    files = sorted(store_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {store_dir}")

    df = pl.scan_parquet(files).select(["timestamp", "close"]).collect()

    # Extract date, take last close per day
    df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))
    df = df.sort("timestamp").group_by("date").agg(pl.col("close").last())
    return df.sort("date")


def _rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Rolling percentile rank over a window."""
    values = series.to_list()
    result: list[float | None] = [None] * len(values)
    for i in range(window - 1, len(values)):
        window_vals = [v for v in values[i - window + 1 : i + 1] if v is not None]
        if len(window_vals) < window // 2:
            continue
        current = values[i]
        if current is None:
            continue
        rank = sum(1 for v in window_vals if v <= current)
        result[i] = rank / len(window_vals)
    return pl.Series("pct", result, dtype=pl.Float32)


def _classify_regime(pct: pl.Expr, name: str) -> pl.Expr:
    """Classify percentile into regime buckets."""
    return (
        pl.when(pct < 0.25)
        .then(pl.lit("low"))
        .when(pct < 0.50)
        .then(pl.lit("mid"))
        .when(pct < 0.75)
        .then(pl.lit("high"))
        .otherwise(pl.lit("extreme"))
        .alias(f"{name}_regime")
    )


def generate_vol_regime(store_root: Path) -> pl.DataFrame:
    """Generate vol_regime DataFrame from VIX/VVIX store data."""
    vix_dir = store_root / "store" / "vix_ohlc"
    vvix_dir = store_root / "store" / "vvix_ohlc"

    logger.info("Loading VIX daily close from %s", vix_dir)
    vix = _daily_close(vix_dir).rename({"close": "vix"})

    logger.info("Loading VVIX daily close from %s", vvix_dir)
    vvix = _daily_close(vvix_dir).rename({"close": "vvix"})

    # Join on date
    df = vix.join(vvix, on="date", how="inner").sort("date")
    logger.info("Joined VIX/VVIX: %d trading days", len(df))

    # Compute derived columns
    # NOTE: iv column removed — was pl.col("vix").alias("iv"), i.e. identical
    # to vix. This caused regime_score double-counting (iv_regime == vix_regime).
    df = df.with_columns(
        [
            # VoV ratio = VVIX / VIX (vol-of-vol relative to vol)
            (pl.col("vvix") / pl.col("vix")).cast(pl.Float32).alias("vov_ratio"),
        ]
    )

    # Rolling percentile ranks (252-day = ~1 year)
    # 3 independent signals: vix, vvix, vov_ratio
    vix_pct = _rolling_percentile_rank(df["vix"], 252)
    vvix_pct = _rolling_percentile_rank(df["vvix"], 252)
    vov_pct = _rolling_percentile_rank(df["vov_ratio"], 252)

    df = df.with_columns(
        [
            vix_pct.alias("vix_pct_252"),
            vvix_pct.alias("vvix_pct_252"),
            vov_pct.alias("vov_ratio_pct_252"),
        ]
    )

    # Rolling z-scores (20-day)
    df = df.with_columns(
        [
            ((pl.col("vix") - pl.col("vix").rolling_mean(20)) / pl.col("vix").rolling_std(20))
            .cast(pl.Float32)
            .alias("vix_z_20"),
            ((pl.col("vvix") - pl.col("vvix").rolling_mean(20)) / pl.col("vvix").rolling_std(20))
            .cast(pl.Float32)
            .alias("vvix_z_20"),
            (
                (pl.col("vov_ratio") - pl.col("vov_ratio").rolling_mean(20))
                / pl.col("vov_ratio").rolling_std(20)
            )
            .cast(pl.Float32)
            .alias("vov_z_20"),
        ]
    )

    # Regime classifications (3 independent signals)
    df = df.with_columns(
        [
            _classify_regime(pl.col("vix_pct_252"), "vix"),
            _classify_regime(pl.col("vvix_pct_252"), "vvix"),
            _classify_regime(pl.col("vov_ratio_pct_252"), "vov"),
        ]
    )

    # Regime score: 0-3 count of how many indicators are "high" or "extreme"
    df = df.with_columns(
        (
            (pl.col("vix_regime").is_in(["high", "extreme"])).cast(pl.Int8)
            + (pl.col("vvix_regime").is_in(["high", "extreme"])).cast(pl.Int8)
            + (pl.col("vov_regime").is_in(["high", "extreme"])).cast(pl.Int8)
        ).alias("regime_score")
    )

    # 2x2 regime: (VIX level) x (VIX trend)
    df = df.with_columns(
        pl.when((pl.col("vix_pct_252") >= 0.5) & (pl.col("vix_z_20") >= 0))
        .then(pl.lit("high_rising"))
        .when((pl.col("vix_pct_252") >= 0.5) & (pl.col("vix_z_20") < 0))
        .then(pl.lit("high_falling"))
        .when((pl.col("vix_pct_252") < 0.5) & (pl.col("vix_z_20") >= 0))
        .then(pl.lit("low_rising"))
        .otherwise(pl.lit("low_falling"))
        .alias("regime_2x2")
    )

    # Desk bucket: simplified action label
    df = df.with_columns(
        pl.when(pl.col("regime_score") >= 3)
        .then(pl.lit("crisis"))
        .when(pl.col("regime_score") >= 2)
        .then(pl.lit("elevated"))
        .when(pl.col("regime_score") >= 1)
        .then(pl.lit("normal"))
        .otherwise(pl.lit("calm"))
        .alias("desk_bucket")
    )

    # Cast float columns to Float32
    float_cols = [c for c in df.columns if df[c].dtype == pl.Float64]
    if float_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

    # Drop rows before we have enough history for percentile ranks
    df = df.drop_nulls(subset=["vix_pct_252"])

    logger.info(
        "Vol regime generated: %d rows, %s → %s", len(df), df["date"].min(), df["date"].max()
    )

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate vol_regime from VIX/VVIX data")
    parser.add_argument(
        "--store-root",
        default=str(Path(__file__).parent.parent),
        help="Project root (default: spx-pipeline/)",
    )
    args = parser.parse_args(argv)

    store_root = Path(args.store_root)
    df = generate_vol_regime(store_root)

    out_path = store_root / "store" / "vol_regime" / "vol_regime.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(
        str(out_path),
        compression="zstd",
        compression_level=3,
    )

    logger.info("Written to %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
