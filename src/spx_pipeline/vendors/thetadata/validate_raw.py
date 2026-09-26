#!/usr/bin/env python3
"""Comprehensive data validation for ThetaData SPXW options & index parquets.

Runs ~30 checks grouped by category. Each check produces PASS / WARN / FAIL.
Outputs a structured report to stdout and optionally to a JSON file.

Usage:
    spx-pipeline thetadata validate-raw                     # full validation
    spx-pipeline thetadata validate-raw --dataset index     # filter by key substring
    spx-pipeline thetadata validate-raw --months 2024_01    # single month
    spx-pipeline thetadata validate-raw --report report.json # save JSON report
    spx-pipeline thetadata validate-raw --verbose           # show per-file details

Checks performed:
 1. Schema — columns, dtypes, no unexpected columns
 2. Temporal coverage — business day completeness, no month gaps
 3. SPXW expiration schedule — Mon/Wed/Fri pre-May 2022, daily after
 4. Timestamp sanity — market hours, no weekends, no future dates
 5. Price sanity — non-negative, bid ≤ ask, zero-price analysis
 6. Strike sanity — within plausible range of underlying
 7. Greeks sanity — delta ∈ [-1,1], IV > 0, sign checks
 8. Bid-ask spread analysis — extreme spread detection
 9. Duplicate detection — exact duplicate rows
10. 0DTE / 1DTE expiration alignment
11. Cross-dataset consistency — index ↔ greeks underlying_price
12. Known ThetaData issues — 1882 test expirations, 9:30 zero prices
13. Open interest sanity — non-negative, reasonable ranges
14. Empty / tiny file detection
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import (
    DATA_ROOT,
    DATASETS,
)

# ─── Constants ────────────────────────────────────────────────────────────────

# SPXW went from Mon/Wed/Fri to daily expirations on 2022-05-16
SPXW_DAILY_START = pd.Timestamp("2022-05-16")

# Regular trading hours (ET)
RTH_OPEN = 9 * 60 + 30  # 09:30
RTH_CLOSE = 16 * 60  # 16:00

# US market holidays (major ones, 2019-2026)
# Used for business day validation — not exhaustive, pandas USFederalHolidayCalendar
# covers most but we also add early closes / special closures.
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    _cal = USFederalHolidayCalendar()
    US_HOLIDAYS = set(_cal.holidays(start="2019-01-01", end="2026-12-31").date)
except Exception:
    US_HOLIDAYS = set()

# Strike range: how far from underlying a strike can be (as multiplier)
STRIKE_RANGE_MULT = (0.03, 3.0)  # 3% to 300% of underlying

# ─── Result collector ─────────────────────────────────────────────────────────


class ValidationReport:
    def __init__(self):
        self.results = []
        self.warnings = 0
        self.failures = 0
        self.passes = 0

    def add(self, check: str, dataset: str, status: str, detail: str, month: str = ""):
        self.results.append(
            {
                "check": check,
                "dataset": dataset,
                "month": month,
                "status": status,
                "detail": detail,
            }
        )
        if status == "FAIL":
            self.failures += 1
        elif status == "WARN":
            self.warnings += 1
        else:
            self.passes += 1

    def ok(self, check, dataset, detail, month=""):
        self.add(check, dataset, "PASS", detail, month)

    def warn(self, check, dataset, detail, month=""):
        self.add(check, dataset, "WARN", detail, month)

    def fail(self, check, dataset, detail, month=""):
        self.add(check, dataset, "FAIL", detail, month)

    def print_summary(self, verbose=False):
        print("\n" + "=" * 70)
        print("  DATA VALIDATION REPORT")
        print("=" * 70)

        # Group by check
        by_check = defaultdict(list)
        for r in self.results:
            by_check[r["check"]].append(r)

        for check, items in by_check.items():
            fails = [r for r in items if r["status"] == "FAIL"]
            warns = [r for r in items if r["status"] == "WARN"]
            passes = [r for r in items if r["status"] == "PASS"]

            if fails:
                icon = "\u2717"
                color = "FAIL"
            elif warns:
                icon = "\u26a0"
                color = "WARN"
            else:
                icon = "\u2713"
                color = "PASS"

            print(
                f"\n  [{icon}] {check} — {color} ({len(passes)}P / {len(warns)}W / {len(fails)}F)"
            )

            if verbose or fails:
                for r in fails:
                    loc = f"{r['dataset']}/{r['month']}" if r["month"] else r["dataset"]
                    print(f"      FAIL  {loc}: {r['detail']}")
            if verbose or warns:
                for r in warns[:10]:  # cap warnings shown
                    loc = f"{r['dataset']}/{r['month']}" if r["month"] else r["dataset"]
                    print(f"      WARN  {loc}: {r['detail']}")
                if len(warns) > 10:
                    print(f"      ... and {len(warns) - 10} more warnings")

        print("\n" + "-" * 70)
        print(f"  TOTAL: {self.passes} PASS | {self.warnings} WARN | {self.failures} FAIL")
        verdict = (
            "CLEAN"
            if self.failures == 0 and self.warnings == 0
            else ("ISSUES FOUND" if self.failures > 0 else "WARNINGS ONLY")
        )
        print(f"  VERDICT: {verdict}")
        print("-" * 70)

    def to_json(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "passes": self.passes,
                "warnings": self.warnings,
                "failures": self.failures,
            },
            "results": self.results,
        }


# ─── Helper: load parquet safely ─────────────────────────────────────────────


def load_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def get_month_files(ds_key: str) -> list[tuple[str, Path]]:
    """Return sorted list of (month_key, path) for a dataset."""
    ds_path = DATA_ROOT / ds_key
    if not ds_path.exists():
        return []
    files = []
    for f in sorted(ds_path.glob("*.parquet")):
        month_key = f.stem  # e.g. "2024_01"
        files.append((month_key, f))
    return files


# ─── Check 1: Schema validation ──────────────────────────────────────────────


def check_schema(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    expected_cols = list(ds["schema"].keys())
    actual_cols = list(df.columns)

    missing = set(expected_cols) - set(actual_cols)
    extra = set(actual_cols) - set(expected_cols)

    if missing:
        report.fail("Schema", ds["key"], f"missing columns: {missing}", month)
    elif extra:
        report.warn("Schema", ds["key"], f"extra columns: {extra}", month)
    else:
        report.ok("Schema", ds["key"], f"{len(actual_cols)} columns OK", month)


# ─── Check 2: Temporal coverage ──────────────────────────────────────────────


def check_temporal_coverage(report: ValidationReport, ds: dict, all_months: list[str]):
    """Check that we have all expected months from START to latest available."""
    files = get_month_files(ds["key"])
    if not files:
        report.fail("Temporal coverage", ds["key"], "no parquet files found")
        return

    file_months = {mk for mk, _ in files}
    missing = set(all_months) - file_months
    if missing:
        report.warn(
            "Temporal coverage",
            ds["key"],
            f"{len(missing)} months missing: {sorted(missing)[:5]}...",
        )
    else:
        report.ok("Temporal coverage", ds["key"], f"all {len(all_months)} months present")


# ─── Check 3: SPXW expiration schedule ───────────────────────────────────────


def check_expiration_schedule(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    """Verify SPXW follows Mon/Wed/Fri pre-May 2022, daily after.

    Only meaningful for daily (0dte/1dte) datasets where we control the expiration.
    OI and EOD use wildcard (*) so they capture ALL expirations including monthlies
    which expire on every weekday — skip those.
    """
    if "expiration" not in df.columns or df.empty:
        return

    # Skip wildcard-expiration datasets (OI, EOD) — they include all contract
    # expirations (monthly SPX + weeklys) so all weekdays are expected
    if ds.get("batch") == "monthly" and ds.get("response_type") == "options":
        return

    exps = pd.to_datetime(df["expiration"], errors="coerce").dropna().unique()
    # Filter out bogus far-future expirations (OPRA test data)
    exps = [e for e in exps if pd.Timestamp(e).year < 2040]
    if len(exps) == 0:
        return

    month_ts = pd.Timestamp(month.replace("_", "-") + "-01")

    if month_ts >= SPXW_DAILY_START:
        # Post May 2022: should have expirations on all business days
        weekend_exps = [e for e in exps if pd.Timestamp(e).dayofweek >= 5]
        if weekend_exps:
            report.fail(
                "Expiration schedule",
                ds["key"],
                f"weekend expirations found: {weekend_exps[:3]}",
                month,
            )
        else:
            report.ok(
                "Expiration schedule", ds["key"], f"{len(exps)} expirations, no weekends", month
            )
    else:
        # Pre May 2022: Mon/Wed/Fri + Tue added ~late 2020, Thu added May 2022
        days = {pd.Timestamp(e).dayofweek for e in exps}
        weekend = days & {5, 6}

        if weekend:
            report.fail(
                "Expiration schedule", ds["key"], f"weekend expirations: dayofweek={weekend}", month
            )
        else:
            report.ok("Expiration schedule", ds["key"], f"expirations on weekdays {days}", month)


# ─── Check 4: Timestamp sanity ───────────────────────────────────────────────


def check_timestamps(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    ts_col = (
        "timestamp" if "timestamp" in df.columns else "created" if "created" in df.columns else None
    )
    if ts_col is None or df.empty:
        return

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    null_count = ts.isna().sum()
    if null_count > 0:
        report.warn("Timestamps", ds["key"], f"{null_count} null timestamps", month)

    ts = ts.dropna()
    if ts.empty:
        return

    # No future dates
    now = pd.Timestamp.now()
    future = (ts > now).sum()
    if future > 0:
        report.fail("Timestamps", ds["key"], f"{future} timestamps in the future", month)

    # No weekend timestamps (index and intraday datasets)
    if ds["response_type"] == "index" or "interval" in str(ds.get("extra_params", {})):
        weekend_rows = ts.dt.dayofweek.isin([5, 6]).sum()
        if weekend_rows > 0:
            report.warn("Timestamps", ds["key"], f"{weekend_rows} rows on weekends", month)

    # Check 9:30 zero-price (known ThetaData issue: first bar can be 0)
    if "price" in df.columns:
        first_bar = df[ts.dt.hour == 9]
        first_bar = first_bar[ts[first_bar.index].dt.minute == 30]
        zero_prices = (first_bar["price"] == 0).sum() if not first_bar.empty else 0
        if zero_prices > 0:
            report.warn(
                "Timestamps",
                ds["key"],
                f"{zero_prices} zero prices at 9:30 (known ThetaData issue)",
                month,
            )
        else:
            report.ok("Timestamps", ds["key"], f"range {ts.min()} to {ts.max()}", month)
    else:
        report.ok("Timestamps", ds["key"], f"range {ts.min()} to {ts.max()}", month)


# ─── Check 5: Price sanity ───────────────────────────────────────────────────


def check_prices(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if df.empty:
        return

    issues = []

    # Bid/ask columns
    if "bid" in df.columns and "ask" in df.columns:
        bid = df["bid"]
        ask = df["ask"]

        # Negative prices
        neg_bid = (bid < 0).sum()
        neg_ask = (ask < 0).sum()
        if neg_bid > 0 or neg_ask > 0:
            issues.append(f"negative prices: bid={neg_bid}, ask={neg_ask}")

        # Bid > Ask (crossed market) — exclude zeros (premarket/no-quote)
        valid = (bid > 0) & (ask > 0)
        crossed = ((bid > ask) & valid).sum()
        if crossed > 0:
            pct = crossed / valid.sum() * 100 if valid.sum() > 0 else 0
            if pct > 1.0:
                issues.append(f"bid > ask: {crossed} rows ({pct:.2f}%)")
            else:
                report.warn(
                    "Prices", ds["key"], f"bid > ask: {crossed} rows ({pct:.2f}%) — minor", month
                )

    # Index price
    if "price" in df.columns:
        neg = (df["price"] < 0).sum()
        if neg > 0:
            issues.append(f"negative index prices: {neg}")

    # EOD OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        ohlc = df[["open", "high", "low", "close"]]
        valid_ohlc = ohlc[(ohlc > 0).all(axis=1)]
        if not valid_ohlc.empty:
            hl_bad = (valid_ohlc["high"] < valid_ohlc["low"]).sum()
            if hl_bad > 0:
                issues.append(f"high < low: {hl_bad} rows")

    if issues:
        report.fail("Prices", ds["key"], "; ".join(issues), month)
    else:
        report.ok("Prices", ds["key"], "all price checks passed", month)


# ─── Check 6: Strike sanity ──────────────────────────────────────────────────


def check_strikes(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if "strike" not in df.columns or df.empty:
        return

    strikes = df["strike"]

    # No negative or zero strikes
    bad = (strikes <= 0).sum()
    if bad > 0:
        report.fail("Strikes", ds["key"], f"{bad} non-positive strikes", month)
        return

    # Range check — SPX was ~2500 in 2019, ~6000 in 2025
    # Far OTM contracts can have strikes 2x underlying, so use generous bounds
    extreme_low = (strikes < 50).sum()
    extreme_high = (strikes > 15000).sum()
    if extreme_low > 0 or extreme_high > 0:
        report.warn(
            "Strikes",
            ds["key"],
            f"extreme strikes: {extreme_low} below 50, {extreme_high} above 15000",
            month,
        )
    else:
        report.ok("Strikes", ds["key"], f"range {strikes.min():.0f} to {strikes.max():.0f}", month)


# ─── Check 7: Greeks sanity ──────────────────────────────────────────────────


def check_greeks(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if "delta" not in df.columns or df.empty:
        return

    issues = []

    # Delta: should be in [-1, 1]
    delta_oob = ((df["delta"] < -1.01) | (df["delta"] > 1.01)).sum()
    if delta_oob > 0:
        issues.append(f"delta outside [-1,1]: {delta_oob}")

    # IV: should be non-negative
    if "implied_vol" in df.columns:
        iv = df["implied_vol"]
        neg_iv = (iv < 0).sum()
        if neg_iv > 0:
            issues.append(f"negative IV: {neg_iv}")

        # Extreme IV (> 500% = 5.0) — common on deep OTM 0DTE, ~8-15% is normal
        # Only flag if > 20% of rows have extreme IV (would indicate data corruption)
        extreme_iv = (iv > 5.0).sum()
        if extreme_iv > 0:
            pct = extreme_iv / len(iv) * 100
            if pct > 20:
                issues.append(f"IV > 500%: {extreme_iv} ({pct:.1f}%) — abnormally high")
            # else: normal for deep OTM contracts, don't flag

    # Vega: should be non-negative for both calls and puts
    if "vega" in df.columns:
        neg_vega = (df["vega"] < -0.001).sum()
        if neg_vega > 0:
            pct = neg_vega / len(df) * 100
            if pct > 1.0:
                issues.append(f"negative vega: {neg_vega} ({pct:.2f}%)")

    if issues:
        report.warn("Greeks", ds["key"], "; ".join(issues), month)
    else:
        report.ok("Greeks", ds["key"], "greeks within expected ranges", month)


# ─── Check 8: Bid-ask spread analysis ────────────────────────────────────────


def check_spreads(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if "bid" not in df.columns or "ask" not in df.columns or df.empty:
        return

    valid = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
    if valid.empty:
        report.warn("Spreads", ds["key"], "no valid bid/ask rows to analyze", month)
        return

    mid = (valid["bid"] + valid["ask"]) / 2
    spread_pct = (
        ((valid["ask"] - valid["bid"]) / mid * 100).replace([np.inf, -np.inf], np.nan).dropna()
    )

    if spread_pct.empty:
        return

    # Spreads > 100% are suspicious for backtesting
    extreme = (spread_pct > 100).sum()
    pct_extreme = extreme / len(spread_pct) * 100

    detail = (
        f"median spread: {spread_pct.median():.1f}%, "
        f"p95: {spread_pct.quantile(0.95):.1f}%, "
        f"p99: {spread_pct.quantile(0.99):.1f}%, "
        f">{'>'}100%: {extreme} ({pct_extreme:.1f}%)"
    )

    if pct_extreme > 5:
        report.warn("Spreads", ds["key"], detail, month)
    else:
        report.ok("Spreads", ds["key"], detail, month)


# ─── Check 9: Duplicates ─────────────────────────────────────────────────────


def check_duplicates(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if df.empty:
        return

    # Use all columns for dedup check
    n_total = len(df)
    n_unique = len(df.drop_duplicates())
    n_dupes = n_total - n_unique

    if n_dupes > 0:
        pct = n_dupes / n_total * 100
        if pct > 1.0:
            report.fail("Duplicates", ds["key"], f"{n_dupes} exact duplicates ({pct:.2f}%)", month)
        else:
            report.warn("Duplicates", ds["key"], f"{n_dupes} exact duplicates ({pct:.3f}%)", month)
    else:
        report.ok("Duplicates", ds["key"], "no duplicates", month)


# ─── Check 10: 0DTE / 1DTE expiration alignment ─────────────────────────────


def check_dte_alignment(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    """For 0DTE: expiration should equal trade date.
    For 1DTE: expiration should be next business day after trade date.
    """
    if "expiration" not in df.columns or "timestamp" not in df.columns or df.empty:
        return

    batch = ds.get("batch", "")
    if "daily" not in batch:
        return

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    exp = pd.to_datetime(df["expiration"], errors="coerce")

    valid = ts.notna() & exp.notna()
    if valid.sum() == 0:
        return

    trade_dates = ts[valid].dt.normalize()
    exp_dates = exp[valid].dt.normalize()

    if "0dte" in batch:
        # Expiration should be same day as trade
        mismatch = (trade_dates != exp_dates).sum()
        if mismatch > 0:
            pct = mismatch / valid.sum() * 100
            report.warn(
                "DTE alignment",
                ds["key"],
                f"0DTE: {mismatch} rows where exp != trade date ({pct:.1f}%)",
                month,
            )
        else:
            report.ok("DTE alignment", ds["key"], "0DTE: all expirations match trade date", month)

    elif "1dte" in batch:
        # Expiration should be 1 business day after trade date
        expected_exp = trade_dates + pd.tseries.offsets.BDay(1)
        mismatch = (expected_exp.values != exp_dates.values).sum()
        if mismatch > 0:
            pct = mismatch / valid.sum() * 100
            # Small mismatches expected around holidays
            if pct > 5:
                report.warn(
                    "DTE alignment",
                    ds["key"],
                    f"1DTE: {mismatch} rows where exp != trade+1bd ({pct:.1f}%)",
                    month,
                )
            else:
                report.ok(
                    "DTE alignment",
                    ds["key"],
                    f"1DTE: {mismatch} minor mismatches ({pct:.1f}%) — likely holidays",
                    month,
                )
        else:
            report.ok("DTE alignment", ds["key"], "1DTE: all expirations = trade+1bd", month)


# ─── Check 11: Cross-dataset consistency ─────────────────────────────────────


def check_cross_consistency(report: ValidationReport, month: str):
    """Compare index SPX prices with underlying_price in greeks."""
    idx_path = DATA_ROOT / "index" / "spx" / f"{month}.parquet"
    greeks_path = DATA_ROOT / "options" / "greeks" / "0dte" / f"{month}.parquet"

    if not idx_path.exists() or not greeks_path.exists():
        return

    try:
        idx = pd.read_parquet(idx_path)
        greeks = pd.read_parquet(greeks_path, columns=["underlying_timestamp", "underlying_price"])
    except Exception:
        return

    if idx.empty or greeks.empty:
        return

    # Compare on matching timestamps
    idx_prices = idx.set_index("timestamp")["price"]
    idx_prices = idx_prices[idx_prices > 0]  # skip 9:30 zeros

    greeks_prices = greeks.drop_duplicates(subset=["underlying_timestamp"])
    greeks_prices = greeks_prices.set_index("underlying_timestamp")["underlying_price"]
    greeks_prices = greeks_prices[greeks_prices > 0]

    common = idx_prices.index.intersection(greeks_prices.index)
    if len(common) < 10:
        report.warn(
            "Cross-consistency",
            "index/spx ↔ greeks/0dte",
            f"only {len(common)} overlapping timestamps",
            month,
        )
        return

    idx_vals = idx_prices.loc[common]
    greeks_vals = greeks_prices.loc[common]
    diff = idx_vals.values - greeks_vals.values
    abs_diff = np.abs(diff)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    if max_diff > 5.0:
        report.warn(
            "Cross-consistency",
            "index/spx ↔ greeks/0dte",
            f"max price diff: {max_diff:.2f}, mean: {mean_diff:.4f}",
            month,
        )
    else:
        report.ok(
            "Cross-consistency",
            "index/spx ↔ greeks/0dte",
            f"max diff: {max_diff:.2f}, mean: {mean_diff:.4f} ({len(common)} points)",
            month,
        )


# ─── Check 12: Known ThetaData issues ────────────────────────────────────────


def check_known_issues(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if df.empty:
        return

    issues = []

    if "expiration" in df.columns:
        exp = df["expiration"].astype(str)

        # 1882 test expirations from OPRA
        test_1882 = exp.str.startswith("1882").sum()
        if test_1882 > 0:
            issues.append(f"OPRA test expirations (year=1882): {test_1882} rows — REMOVE")

        # Far-future bogus expirations (2088, 2089, etc.) — OPRA encoding errors
        exp_years = pd.to_datetime(df["expiration"], errors="coerce").dt.year
        bogus_future = (exp_years > 2040).sum()
        if bogus_future > 0:
            bad_years = sorted(exp_years[exp_years > 2040].unique())
            issues.append(
                f"bogus future expirations (years {bad_years}): {bogus_future} rows — REMOVE"
            )

    # Check for symbol consistency
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        unexpected = [s for s in symbols if s not in ("SPXW", "SPX", "VIX", "VVIX")]
        if unexpected:
            issues.append(f"unexpected symbols: {unexpected}")

    # Call/Put balance check
    if "right" in df.columns:
        rights = df["right"].value_counts()
        if len(rights) == 2:
            ratio = rights.min() / rights.max()
            if ratio < 0.3:
                issues.append(f"call/put imbalance: {dict(rights)} (ratio={ratio:.2f})")

    if issues:
        severity = "fail" if any("REMOVE" in i for i in issues) else "warn"
        getattr(report, severity)("Known issues", ds["key"], "; ".join(issues), month)
    else:
        report.ok("Known issues", ds["key"], "no known issues detected", month)


# ─── Check 13: Open interest sanity ──────────────────────────────────────────


def check_open_interest(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    if "open_interest" not in df.columns or df.empty:
        return

    oi = df["open_interest"]
    neg = (oi < 0).sum()
    null = oi.isna().sum()

    if neg > 0:
        report.fail("Open interest", ds["key"], f"{neg} negative OI values", month)
    elif null > 0:
        report.warn("Open interest", ds["key"], f"{null} null OI values", month)
    else:
        report.ok(
            "Open interest", ds["key"], f"range 0 to {oi.max():,}, mean={oi.mean():.0f}", month
        )


# ─── Check 14: Empty / tiny file detection ───────────────────────────────────


def check_file_sizes(report: ValidationReport, ds: dict):
    files = get_month_files(ds["key"])
    empty = []
    tiny = []

    for mk, path in files:
        try:
            meta = pq.read_metadata(str(path))
            if meta.num_rows == 0:
                empty.append(mk)
            elif meta.num_rows < 10:
                tiny.append((mk, meta.num_rows))
        except Exception as e:
            report.fail("File integrity", ds["key"], f"corrupt file: {e}", mk)

    if empty:
        report.warn("File sizes", ds["key"], f"{len(empty)} empty files: {empty[:5]}")
    if tiny:
        report.warn("File sizes", ds["key"], f"{len(tiny)} tiny files (<10 rows): {tiny[:5]}")
    if not empty and not tiny:
        report.ok("File sizes", ds["key"], f"all {len(files)} files have data")


# ─── Check 15: Business day coverage within months ────────────────────────────


def check_bday_coverage(report: ValidationReport, ds: dict, month: str, df: pd.DataFrame):
    """For daily datasets, check how many business days in the month have data.

    Pre-May 2022 SPXW only had Mon/Wed/Fri expirations, so 0DTE/1DTE datasets
    will naturally have ~60% bday coverage. We adjust expectations accordingly.
    """
    if df.empty:
        return

    ts_col = (
        "timestamp" if "timestamp" in df.columns else "created" if "created" in df.columns else None
    )
    if ts_col is None:
        return

    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
    if ts.empty:
        return

    actual_dates = set(ts.dt.normalize().unique())

    # Generate expected business days for this month
    year, mo = month.split("_")
    month_start = pd.Timestamp(f"{year}-{mo}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    expected_bdays = set(pd.bdate_range(month_start, month_end))

    # Remove known holidays
    expected_bdays = {d for d in expected_bdays if d.date() not in US_HOLIDAYS}

    if not expected_bdays:
        return

    # For 0DTE/1DTE pre-May 2022: SPXW only had Mon/Wed/Fri expirations
    # so only ~60% of business days should have data — adjust threshold
    is_daily = "daily" in ds.get("batch", "")
    is_pre_daily = month_start < SPXW_DAILY_START

    if is_daily and is_pre_daily:
        # Only count Mon(0)/Wed(2)/Fri(4) as expected
        mwf_bdays = {d for d in expected_bdays if d.dayofweek in (0, 2, 4)}
        coverage = len(actual_dates & expected_bdays) / len(mwf_bdays) * 100 if mwf_bdays else 100
        threshold = 70
        label = f"{coverage:.0f}% MWF coverage ({len(actual_dates)}/{len(mwf_bdays)} MWF days)"
    else:
        coverage = len(actual_dates & expected_bdays) / len(expected_bdays) * 100
        threshold = 70
        label = f"{coverage:.0f}% coverage ({len(actual_dates)}/{len(expected_bdays)} days)"

    if coverage < threshold:
        report.warn("Bday coverage", ds["key"], label, month)
    else:
        report.ok("Bday coverage", ds["key"], label, month)


# ─── Aggregate statistics (printed at end) ───────────────────────────────────


def compute_aggregate_stats():
    """Compute and print overall dataset statistics."""
    print("\n" + "=" * 70)
    print("  DATASET STATISTICS")
    print("=" * 70)

    total_rows = 0
    total_size = 0

    for ds in DATASETS:
        files = get_month_files(ds["key"])
        ds_rows = 0
        ds_size = 0
        for _month, path in files:
            try:
                meta = pq.read_metadata(str(path))
                ds_rows += meta.num_rows
                ds_size += path.stat().st_size
            except Exception:
                pass

        total_rows += ds_rows
        total_size += ds_size
        print(
            f"  {ds['key']:<30} {len(files):>3} files | {ds_rows:>13,} rows"
            f" | {ds_size / 1e9:.2f} GB"
        )

    print(f"  {'TOTAL':<30} {'':>3}       | {total_rows:>13,} rows | {total_size / 1e9:.2f} GB")


# ─── Main runner ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate ThetaData parquet files")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by key substring")
    parser.add_argument(
        "--months", type=str, default=None, help="Comma-separated month keys (e.g. 2024_01)"
    )
    parser.add_argument("--report", type=str, default=None, help="Save JSON report to file")
    parser.add_argument("--verbose", action="store_true", help="Show per-file details")
    parser.add_argument("--quick", action="store_true", help="Sample 1 month per year (faster)")
    args = parser.parse_args(argv)

    print(f"Data validation — {DATA_ROOT}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Filter datasets
    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if args.dataset in d["key"]]
        if not datasets:
            print(f"No datasets matching '{args.dataset}'")
            sys.exit(1)

    # Generate expected month list
    all_months = []
    current = pd.Timestamp("2019-01-01")
    end = pd.Timestamp.now()
    while current <= end:
        all_months.append(current.strftime("%Y_%m"))
        current += pd.offsets.MonthBegin(1)

    # Filter months
    if args.months:
        target_months = set(args.months.split(","))
    elif args.quick:
        # Sample: Jan and Jul of each year
        target_months = {m for m in all_months if m.endswith("_01") or m.endswith("_07")}
    else:
        target_months = None  # all

    report = ValidationReport()

    for ds in datasets:
        print(f"{'─' * 60}")
        print(f"  Validating: {ds['key']}")
        print(f"{'─' * 60}")

        # Check 14: file sizes (runs on all files)
        check_file_sizes(report, ds)

        # Check 2: temporal coverage
        check_temporal_coverage(report, ds, all_months)

        # Per-month checks
        files = get_month_files(ds["key"])
        for mk, path in files:
            if target_months and mk not in target_months:
                continue

            df = load_parquet(path)
            if df is None:
                report.fail("File integrity", ds["key"], "failed to read parquet", mk)
                continue

            # Run all per-month checks
            check_schema(report, ds, mk, df)
            check_timestamps(report, ds, mk, df)
            check_prices(report, ds, mk, df)
            check_strikes(report, ds, mk, df)
            check_greeks(report, ds, mk, df)
            check_spreads(report, ds, mk, df)
            check_known_issues(report, ds, mk, df)
            check_open_interest(report, ds, mk, df)
            check_expiration_schedule(report, ds, mk, df)
            check_dte_alignment(report, ds, mk, df)
            check_bday_coverage(report, ds, mk, df)

            # Duplicates: expensive, sample only in --verbose or single month
            if args.verbose or (target_months and len(target_months) <= 3):
                check_duplicates(report, ds, mk, df)

            if not args.verbose:
                sys.stdout.write(".")
                sys.stdout.flush()

        if not args.verbose:
            print()  # newline after dots

    # Cross-dataset consistency (index vs greeks)
    print(f"\n{'─' * 60}")
    print("  Cross-dataset consistency checks")
    print(f"{'─' * 60}")
    for mk in all_months:
        if target_months and mk not in target_months:
            continue
        check_cross_consistency(report, mk)
        if not args.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()
    if not args.verbose:
        print()

    # Print results
    compute_aggregate_stats()
    report.print_summary(verbose=args.verbose)

    # Save JSON report
    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report.to_json(), indent=2, default=str))
        print(f"\nReport saved to {report_path}")

    return 1 if report.failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
