"""Reproducible micro-benchmark on the synthetic feed.

    python benchmarks/bench.py            # 20 sessions, ~1.3M option rows
    python benchmarks/bench.py --days 60

Measures ingestion throughput, the effect of predicate pushdown (a strike band vs the
whole chain), and the memory-mapped cache. Numbers depend on the machine; the point is
the ratios, and that anyone can re-run them.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from spx_pipeline import DataLoader, DataStore, Registry
from spx_pipeline.registry import DEFAULT_REGISTRY
from spx_pipeline.synthetic import SyntheticSpec, business_days, generate_raw_dataset


def timed(fn, repeat: int = 5) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20, help="calendar days of synthetic data")
    args = ap.parse_args()

    start = date(2024, 1, 2)
    spec = SyntheticSpec(start=start, end=start + timedelta(days=args.days), dtes=(0,))
    sessions = business_days(spec.start, spec.end)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        counts = generate_raw_dataset(root / "raw", spec)
        rows = counts["options/greeks/0dte"]

        store = DataStore(root / "raw", root, Registry(DEFAULT_REGISTRY))
        t0 = time.perf_counter()
        store.ingest_dataset("greeks_0dte")
        ingest = time.perf_counter() - t0

        loader = DataLoader(store_root=root, cache_dir=root / ".cache")
        day = sessions[len(sessions) // 2]
        band = [("right", "=", "PUT"), ("strike", ">=", 4700.0), ("strike", "<=", 4750.0)]
        full = timed(lambda: loader.load("greeks_0dte", day, use_cache=False))
        sliced = timed(lambda: loader.load("greeks_0dte", day, filters=band, use_cache=False))
        month = timed(
            lambda: loader.load(
                "greeks_0dte", (sessions[0], sessions[-1]), filters=band, use_cache=False
            ),
            repeat=3,
        )
        loader.load("greeks_0dte", day, filters=band)  # populate the cache
        cached = timed(lambda: loader.load("greeks_0dte", day, filters=band))
        n_band = loader.load("greeks_0dte", day, filters=band).height

    print(
        f"machine: {platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, "
        f"Python {platform.python_version()}"
    )
    print(
        f"data:    {len(sessions)} sessions, {rows:,} raw option rows "
        "(0DTE chain, 41 strikes x call/put, 1-minute)"
    )
    print(f"ingest:  {ingest:.2f}s  ->  {rows / ingest:,.0f} rows/s (schema cast + ZSTD write)")
    print(f"query:   one session, whole chain            {full * 1e3:7.1f} ms  (median of 5)")
    print(f"         one session, puts 4700<=K<=4750     {sliced * 1e3:7.1f} ms  ({n_band:,} rows)")
    print(f"         {len(sessions)} sessions, same band            {month * 1e3:7.1f} ms")
    print(f"         same band, cache hit (mmap Arrow)   {cached * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
