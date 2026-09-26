"""Command-line interface: ``spx-pipeline <command>``.

Commands
--------
datasets    list the datasets declared in the registry
demo        generate a synthetic feed and walk through the whole pipeline
migrate     ingest raw vendor files into the Parquet store
validate    check a store against its raw files
regime      build the daily volatility-regime file from VIX/VVIX
thetadata   ThetaData tooling: download | split | validate-raw | validate-store | ping
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import polars as pl

from . import __version__
from .cleaning import RULE_HANDLERS
from .loader import DataLoader
from .pit import PointInTimeLoader
from .registry import DEFAULT_REGISTRY, Registry
from .store import DataStore
from .synthetic import SyntheticSpec, business_days, generate_raw_dataset

_THETADATA = {
    "download": "spx_pipeline.vendors.thetadata.downloader",
    "split": "spx_pipeline.vendors.thetadata.split_monthly",
    "validate-raw": "spx_pipeline.vendors.thetadata.validate_raw",
    "validate-store": "spx_pipeline.vendors.thetadata.validate_store",
    "ping": "spx_pipeline.vendors.thetadata.test_connection",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="spx-pipeline", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="list the datasets declared in the registry")

    p_demo = sub.add_parser("demo", help="end-to-end tour on synthetic data")
    p_demo.add_argument("--workdir", type=Path, help="keep the generated data here")
    p_demo.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 2))
    p_demo.add_argument("--end", type=date.fromisoformat, default=date(2024, 1, 12))

    for name in ("migrate", "validate", "regime"):
        sub.add_parser(name, add_help=False, help=f"see: spx-pipeline {name} --help")
    p_td = sub.add_parser("thetadata", help="ThetaData tooling (needs the [thetadata] extra)")
    p_td.add_argument("action", choices=sorted(_THETADATA))

    args, rest = parser.parse_known_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "datasets":
        return _cmd_datasets()
    if args.command == "demo":
        return _cmd_demo(args.workdir, args.start, args.end)
    if args.command == "migrate":
        from .tools.migrate import main as run
    elif args.command == "validate":
        from .tools.validate import main as run
    elif args.command == "regime":
        from .tools.vol_regime import main as run
    else:
        import importlib

        run = importlib.import_module(_THETADATA[args.action]).main
    run(rest)
    return 0


# ------------------------------------------------------------------ datasets


def _cmd_datasets() -> int:
    registry = Registry(DEFAULT_REGISTRY)
    rows = []
    for name in registry.list_datasets():
        cfg = registry.get(name)
        rules = ", ".join(r.name for r in cfg.cleaning_rules) or "-"
        rows.append(
            (
                name,
                cfg.granularity,
                f"{cfg.rows_per_day:,}" if cfg.rows_per_day else "-",
                rules,
                cfg.description,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    header = f"{'dataset':<{widths[0]}}  {'bars':<{widths[1]}}  {'rows/day':>{widths[2]}}"
    print(f"{header}  cleaning rules")
    for name, gran, rpd, rules, _desc in rows:
        print(f"{name:<{widths[0]}}  {gran:<{widths[1]}}  {rpd:>{widths[2]}}  {rules}")
    print(
        f"\n{len(rows)} datasets, {len(registry.aliases)} legacy aliases, "
        f"{len(RULE_HANDLERS)} cleaning rules available"
    )
    return 0


# ---------------------------------------------------------------------- demo


def _cmd_demo(workdir: Path | None, start: date, end: date) -> int:
    logging.getLogger("spx_pipeline").setLevel(logging.WARNING)
    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="spx-pipeline-demo-")
        workdir = Path(tmp)
    raw_root, data_root = workdir / "raw", workdir
    try:
        _demo(raw_root, data_root, start, end)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _step(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title}")


def _demo(raw_root: Path, data_root: Path, start: date, end: date) -> None:
    days = business_days(start, end)
    if len(days) < 4:
        raise SystemExit("demo needs at least 4 business days between --start and --end")
    spec = SyntheticSpec(start=start, end=end)

    _step(1, f"Synthetic raw feed: {len(days)} sessions, SPX/VIX/VVIX + SPXW chain (0DTE, 1DTE)")
    t0 = time.perf_counter()
    counts = generate_raw_dataset(raw_root, spec)
    raw_mb = _size_mb(raw_root)
    n_opt = sum(v for k, v in counts.items() if k.startswith("options"))
    print(
        f"      {n_opt:,} option rows + {counts['index/spx']:,} index bars per series, "
        f"{raw_mb:.1f} MB Parquet/Snappy  ({time.perf_counter() - t0:.1f}s)"
    )

    _step(2, "Ingest into the store: schema enforced, Parquet ZSTD, year=/month= partitions")
    registry = Registry(DEFAULT_REGISTRY)
    store = DataStore(raw_root, data_root, registry)
    t0 = time.perf_counter()
    n_files = 0
    for ds in ("vix_ohlc", "vvix_ohlc", "vol_regime", "spx_ohlc", "greeks_0dte", "greeks_1dte"):
        n_files += store.ingest_dataset(ds)["ingested"]
    store_mb = _size_mb(data_root / "store")
    print(
        f"      {n_files} files -> {store_mb:.1f} MB under store/, one file per session  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    loader = DataLoader(store_root=data_root, cache_dir=data_root / ".cache")
    day = days[1]

    _step(3, "Query a slice of the chain: filters pushed down to the Parquet reader")
    spot = float(loader.load("spx_ohlc", day)["open"][0])
    lo, hi = round(spot / 5) * 5 - 50, round(spot / 5) * 5 + 50
    flt = [("right", "=", "PUT"), ("strike", ">=", lo), ("strike", "<=", hi)]
    t0 = time.perf_counter()
    puts = loader.load("greeks_0dte", day, filters=flt)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    loader.load("greeks_0dte", day, filters=flt)
    warm = time.perf_counter() - t0
    print(f"      greeks_0dte {day}  PUT  {lo:.0f} <= strike <= {hi:.0f}: {puts.height:,} rows")
    print(
        f"      first query {cold * 1e3:.0f} ms, repeat {warm * 1e3:.1f} ms "
        "(memory-mapped Arrow cache)"
    )

    _step(4, f"What the cleaning rules removed from greeks_0dte on {days[0]}")
    raw = loader.load("greeks_0dte", days[0], apply_cleaning=False, use_cache=False)
    df = raw
    print(f"      raw rows: {raw.height:,}")
    for rule in registry.get("greeks_0dte").cleaning_rules:
        before = df.height
        df = RULE_HANDLERS[rule.name](df, rule.params)
        dropped = before - df.height
        effect = "adds a column" if rule.name.startswith("add_") else f"-{dropped:,} rows"
        print(f"      - {rule.name:<26} {effect:>14}   {_WHY.get(rule.name, '')}")
    print(f"      clean rows: {df.height:,}")

    _step(5, "Point-in-time guarantees")
    cutoff = days[2]
    pit = PointInTimeLoader(loader, cutoff_date=cutoff, feature_lag_bars=1)
    spx = pit.load("spx_ohlc", (days[0], days[-1]))
    last_bar: datetime = spx["timestamp"].max()  # type: ignore[assignment]
    print(
        f"      cutoff {cutoff}: asked up to {days[-1]}, last bar returned "
        f"{last_bar:%Y-%m-%d %H:%M}"
    )

    aligned = loader.load_aligned((days[1], days[2]), datasets=["spx_ohlc", "vol_regime"])
    regime = loader.load("vol_regime", (days[0], days[2]), apply_cleaning=False)
    bar = aligned.filter(pl.col("timestamp").dt.date() == days[2]).row(0, named=True)
    prev_close_regime = regime.filter(pl.col("date") == days[1]).row(0, named=True)
    same = abs(bar["vix"] - prev_close_regime["vix"]) < 1e-3
    print(
        f"      regime lag: bars of {days[2]} carry the regime of the {days[1]} close "
        f"(vix {bar['vix']:.2f})  {'OK' if same else 'MISMATCH'}"
    )

    chain = pit.load("greeks_0dte", day, filters=[("strike", "=", lo), ("right", "=", "PUT")])
    both = loader.load("greeks_0dte", day, filters=[("strike", "IN", [lo, lo + 5])])
    lagged = pit.with_feature_lag(both, ["delta"])
    k0 = lagged.filter((pl.col("strike") == lo) & (pl.col("right") == "PUT"))
    ok = k0["delta_lag1"].slice(1).to_list() == chain["delta"].slice(0, k0.height - 1).to_list()
    t1 = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=32)
    print(
        f"      per-contract feature lag: delta_lag1 of the {lo:.0f} put at {t1:%H:%M} is its own "
        f"09:31 delta, not a neighbour's  {'OK' if ok else 'MISMATCH'}"
    )
    print("\nDone. Everything above ran on synthetic data: plumbing, not market signal.")


_WHY = {
    "filter_bogus_expirations": "OPRA test contracts (expiry 1882)",
    "drop_opening_bar": "09:30 bar: underlying = 0, IV overflow",
    "filter_illiquid": "zero bid / IV > 100 / no underlying",
    "add_mid": "mid = (bid + ask) / 2",
}


def _size_mb(root: Path) -> float:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6


if __name__ == "__main__":
    raise SystemExit(main())
