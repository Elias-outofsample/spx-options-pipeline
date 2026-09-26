#!/usr/bin/env python3
"""ThetaData historical data downloader — parallel version.

Usage:
    spx-pipeline thetadata download                  # full download (2019 → today)
    spx-pipeline thetadata download --dataset quotes # single dataset family
    spx-pipeline thetadata download --from 2022-01  # resume from specific month
    spx-pipeline thetadata download --check          # verify parquet integrity
"""

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pandas.tseries.offsets import BDay, MonthBegin, MonthEnd

from .config import (
    BACKOFF,
    BASE_URL,
    COMPLETED_FILE,
    DATA_ROOT,
    DATASETS,
    ERRORS_LOG,
    MAX_CONCURRENT,
    MAX_RETRIES,
    RATE_LIMIT_SLEEP,
    ROW_GROUP_SIZE,
    START_DATE,
    ZSTD_LEVEL,
    empty_dataframe,
)

log = logging.getLogger("thetadl")

# ─── Thread safety ───────────────────────────────────────────────────────────
_api_sem = threading.Semaphore(MAX_CONCURRENT)
_completed_lock = threading.Lock()
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """One requests.Session per thread (Session is not thread-safe)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ─── Logging setup ───────────────────────────────────────────────────────────


def setup_logging():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    errfile = logging.FileHandler(str(ERRORS_LOG), mode="a")
    errfile.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    errfile.setLevel(logging.WARNING)

    log.setLevel(logging.DEBUG)
    log.addHandler(console)
    log.addHandler(errfile)


# ─── Resume system ───────────────────────────────────────────────────────────


def load_completed() -> set[str]:
    if not COMPLETED_FILE.exists():
        return set()
    keys = set()
    with open(COMPLETED_FILE) as f:
        for line in f:
            key = line.split("|")[0].strip()
            if key:
                keys.add(key)
    return keys


def mark_completed(key: str, rows: int, size_mb: float):
    with _completed_lock:
        COMPLETED_FILE.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        with open(COMPLETED_FILE, "a") as f:
            f.write(f"{key} | {ts} | {rows} rows | {size_mb:.1f} MB\n")


# ─── HTTP fetch with retry ──────────────────────────────────────────────────


def log_error(endpoint: str, params: dict, status: int, msg: str):
    log.warning(f"{endpoint} | {params} | {status} | {msg}")


def fetch(endpoint: str, params: dict) -> dict | None:
    """Fetch JSON from ThetaData API with retry + semaphore concurrency control.

    The semaphore is held only during the HTTP call itself,
    released during backoff sleeps so other threads can proceed.
    """
    url = f"{BASE_URL}{endpoint}"
    params = {**params, "format": "json"}
    sess = _get_session()

    for attempt in range(MAX_RETRIES + 1):
        try:
            with _api_sem:
                resp = sess.get(url, params=params, timeout=120)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                log.warning(f"Rate limited, sleeping {RATE_LIMIT_SLEEP}s...")
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            if resp.status_code == 472:
                return None

            if resp.status_code == 570:
                log_error(endpoint, params, 570, "LARGE_REQUEST — skipping")
                return None

            log.warning(f"HTTP {resp.status_code} for {endpoint} (attempt {attempt + 1})")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
                continue
            log_error(endpoint, params, resp.status_code, resp.text[:200])
            return None

        except requests.exceptions.Timeout:
            if attempt == 0:
                continue
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
                continue
            log_error(endpoint, params, 0, "timeout")
            return None

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
                continue
            log_error(endpoint, params, 0, str(e))
            return None

    return None


# ─── Response parsing ────────────────────────────────────────────────────────


def parse_response(data: dict, response_type: str) -> pd.DataFrame:
    """Parse ThetaData JSON into DataFrame.

    Index: flat list.  Options: grouped by contract, flattened.
    """
    rows = data.get("response", [])
    if not rows:
        return pd.DataFrame()

    if response_type == "index":
        return pd.DataFrame(rows)

    records = []
    for item in rows:
        contract = item.get("contract", {})
        for record in item.get("data", []):
            records.append({**contract, **record})
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ─── DataFrame casting ──────────────────────────────────────────────────────


def cast_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        if dtype == "category":
            df[col] = df[col].astype("category")
        elif dtype.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype == "float32":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        elif dtype == "Int32":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
        elif dtype == "str":
            df[col] = df[col].astype(str)
    cols = [c for c in schema if c in df.columns]
    return df[cols]


# ─── Parquet save ────────────────────────────────────────────────────────────


def save_parquet(df: pd.DataFrame, path: Path, schema: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        compression="zstd",
        compression_level=ZSTD_LEVEL,
        write_statistics=True,
        row_group_size=ROW_GROUP_SIZE,
    )


# ─── Month generation ───────────────────────────────────────────────────────


def generate_months(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    current = s.replace(day=1)
    months = []
    while current <= e:
        ms = max(current, s)
        me = min(current + MonthEnd(0), e)
        key = current.strftime("%Y_%m")
        months.append((ms, me, key))
        current += MonthBegin(1)
    return months


# ─── Download logic ──────────────────────────────────────────────────────────


def download_monthly(ds: dict, month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    """Single API call for one month (index, OI, EOD)."""
    params = {
        "start_date": month_start.strftime("%Y%m%d"),
        "end_date": month_end.strftime("%Y%m%d"),
        **ds["extra_params"],
    }
    data = fetch(ds["endpoint"], params)
    if data is None:
        return empty_dataframe(ds["schema"])
    df = parse_response(data, ds["response_type"])
    if df.empty:
        return empty_dataframe(ds["schema"])
    return cast_dataframe(df, ds["schema"])


def _fetch_single_day(ds: dict, trade_date: pd.Timestamp, dte: int) -> pd.DataFrame:
    """Fetch one day — called from ThreadPoolExecutor."""
    expiration = trade_date if dte == 0 else trade_date + BDay(dte)
    params = {
        "date": trade_date.strftime("%Y%m%d"),
        "expiration": expiration.strftime("%Y%m%d"),
        **ds["extra_params"],
    }
    data = fetch(ds["endpoint"], params)
    if data is None:
        return pd.DataFrame()
    return parse_response(data, ds["response_type"])


def download_daily_in_month(
    ds: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    dte: int,
) -> pd.DataFrame:
    """Fetch day-by-day with parallel requests within the month."""
    bdays = pd.bdate_range(month_start, month_end)
    if len(bdays) == 0:
        return empty_dataframe(ds["schema"])

    frames = []
    # Parallel days — semaphore inside fetch() caps actual concurrency.
    # Inner pool of 2 is enough: outer pool provides the real parallelism.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_fetch_single_day, ds, d, dte): d for d in bdays}
        for future in as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                day = futures[future]
                log.error(f"  day {day.strftime('%Y%m%d')} failed: {e}")

    if not frames:
        return empty_dataframe(ds["schema"])
    combined = pd.concat(frames, ignore_index=True)
    return cast_dataframe(combined, ds["schema"])


# ─── Process one month ──────────────────────────────────────────────────────


def _process_month(ds, month_start, month_end, month_key):
    """Download + save + mark_completed. Returns (resume_key, rows, size_mb, elapsed)."""
    resume_key = f"{ds['key']}/{month_key}"
    t0 = time.time()

    if ds["batch"] == "monthly":
        df = download_monthly(ds, month_start, month_end)
    elif ds["batch"].startswith("daily_") and ds["batch"][6:-3].isdigit():
        dte = int(ds["batch"][6:-3])  # "daily_3dte" → 3
        df = download_daily_in_month(ds, month_start, month_end, dte=dte)
    else:
        log.error(f"Unknown batch mode: {ds['batch']}")
        return None

    path = DATA_ROOT / ds["key"] / f"{month_key}.parquet"
    save_parquet(df, path, ds["schema"])
    size_mb = path.stat().st_size / 1e6
    elapsed = time.time() - t0

    mark_completed(resume_key, len(df), size_mb)
    log.info(f"\u2713 {resume_key} | {len(df)} rows | {size_mb:.1f} MB | {elapsed:.1f}s")
    return resume_key, len(df), size_mb, elapsed


# ─── Integrity check ────────────────────────────────────────────────────────


def check_integrity():
    issues = 0
    total = 0
    for ds in DATASETS:
        ds_path = DATA_ROOT / ds["key"]
        if not ds_path.exists():
            log.warning(f"Missing directory: {ds_path}")
            continue
        for f in sorted(ds_path.glob("*.parquet")):
            total += 1
            try:
                meta = pq.read_metadata(str(f))
                size_mb = f.stat().st_size / 1e6
                log.info(
                    f"OK {f.relative_to(DATA_ROOT)} | "
                    f"{meta.num_rows} rows | {meta.num_columns} cols | {size_mb:.1f} MB"
                )
            except Exception as e:
                log.error(f"CORRUPT {f.relative_to(DATA_ROOT)} | {e}")
                issues += 1
    log.info(f"Checked {total} files, {issues} issues")
    return issues


# ─── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ThetaData historical downloader")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by key substring")
    parser.add_argument("--check", action="store_true", help="Verify parquet integrity")
    parser.add_argument(
        "--from", dest="from_month", type=str, default=None, help="Start month (YYYY-MM)"
    )
    args = parser.parse_args(argv)

    setup_logging()

    if args.check:
        sys.exit(check_integrity())

    # Filter datasets
    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if args.dataset in d["key"]]
        if not datasets:
            log.error(f"No datasets matching '{args.dataset}'")
            log.info(f"Available: {', '.join(d['key'] for d in DATASETS)}")
            sys.exit(1)
        log.info(f"Filtered to {len(datasets)} dataset(s): {', '.join(d['key'] for d in datasets)}")

    # Generate months
    today = datetime.now().strftime("%Y-%m-%d")
    months = generate_months(START_DATE, today)
    if args.from_month:
        from_key = args.from_month.replace("-", "_")
        months = [(ms, me, mk) for ms, me, mk in months if mk >= from_key]
        log.info(f"Starting from month {args.from_month} ({len(months)} months)")

    # Load resume state
    completed = load_completed()
    log.info(f"Loaded {len(completed)} completed keys | concurrency: {MAX_CONCURRENT}")

    start_time = time.time()
    files_done = 0
    total_rows = 0
    last_report = start_time

    # ── Collect all pending (dataset, month) tasks across ALL datasets ────────
    # Monthly datasets run first (fast); daily datasets are interleaved
    # round-robin across DTEs so all DTEs progress simultaneously.
    monthly_tasks = []
    daily_task_lists = []  # one list per daily dataset
    for ds in datasets:
        ds_pending = [
            (ds, ms, me, mk) for ms, me, mk in months if f"{ds['key']}/{mk}" not in completed
        ]
        if not ds_pending:
            log.info(f"  {ds['key']}: all months done, skipping")
        else:
            log.info(f"  {ds['key']}: {len(ds_pending)} months pending")
            if ds["batch"] == "monthly":
                monthly_tasks.extend(ds_pending)
            else:
                daily_task_lists.append(ds_pending)

    # Interleave tasks round-robin across DTEs:
    # [3DTE/m1, 4DTE/m1, 5DTE/m1, 6DTE/m1, 7DTE/m1, 3DTE/m2, 4DTE/m2, ...]
    # This ensures all DTEs get workers simultaneously instead of one DTE
    # monopolizing the pool until it finishes.
    daily_tasks = [
        task
        for round_tasks in zip_longest(*daily_task_lists)
        for task in round_tasks
        if task is not None
    ]

    # Monthly datasets: process first (fast, few tasks)
    if monthly_tasks:
        log.info(f"Monthly tasks: {len(monthly_tasks)} — running now")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {
                pool.submit(_process_month, ds, ms, me, mk): (ds["key"], mk)
                for ds, ms, me, mk in monthly_tasks
            }
            for future in as_completed(futures):
                ds_key, mk = futures[future]
                try:
                    result = future.result()
                    if result:
                        rk, rows, _, _ = result
                        completed.add(rk)
                        files_done += 1
                        total_rows += rows
                except Exception as e:
                    log.error(f"\u2717 {ds_key}/{mk} | EXCEPTION: {e}")

    # Daily datasets: all DTEs in one flat pool — all datasets progress at once.
    # n_workers >> MAX_CONCURRENT so the HTTP semaphore stays saturated across
    # all active DTEs simultaneously.
    if daily_tasks:
        n_workers = min(len(daily_tasks), MAX_CONCURRENT * 4)
        log.info(
            f"Daily tasks: {len(daily_tasks)} months across all DTEs — "
            f"{n_workers} workers, {MAX_CONCURRENT} HTTP slots"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_month, ds, ms, me, mk): (ds["key"], mk)
                for ds, ms, me, mk in daily_tasks
            }
            for future in as_completed(futures):
                ds_key, mk = futures[future]
                try:
                    result = future.result()
                    if result:
                        rk, rows, _, _ = result
                        completed.add(rk)
                        files_done += 1
                        total_rows += rows
                except Exception as e:
                    log.error(f"\u2717 {ds_key}/{mk} | EXCEPTION: {e}")

                # Hourly progress report
                now = time.time()
                if now - last_report > 3600:
                    elapsed_h = (now - start_time) / 3600
                    speed = files_done / elapsed_h if elapsed_h > 0 else 0
                    disk_gb = sum(f.stat().st_size for f in DATA_ROOT.rglob("*.parquet")) / 1e9
                    log.info(
                        f"REPORT | {files_done} done | {disk_gb:.1f} GB disk | {speed:.0f} files/hr"
                    )
                    last_report = now

    total_elapsed = time.time() - start_time
    log.info(f"DONE | {files_done} files | {total_rows:,} rows | {total_elapsed / 60:.0f} min")


if __name__ == "__main__":
    main()
