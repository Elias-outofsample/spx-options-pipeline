#!/usr/bin/env python3
"""Pre-flight check for ThetaData terminal — Go/No-Go before downloading."""

import sys

import requests

from .config import BASE_URL

# ─── Test definitions ────────────────────────────────────────────────────────
# (label, path, params, response_type)
# response_type: "flat" = raw JSON array, "options" = grouped by contract
TESTS = [
    (
        "Terminal alive",
        "/v3/index/list/symbols",
        {},
        "flat",
    ),
    (
        "SPX index 1m",
        "/v3/index/history/price",
        {"symbol": "SPX", "date": "20240102", "interval": "1m"},
        "index",
    ),
    (
        "VIX index 1m",
        "/v3/index/history/price",
        {"symbol": "VIX", "date": "20240102", "interval": "1m"},
        "index",
    ),
    (
        "VVIX index 1m",
        "/v3/index/history/price",
        {"symbol": "VVIX", "date": "20240102", "interval": "1m"},
        "index",
    ),
    (
        "SPXW quotes 1m",
        "/v3/option/history/quote",
        {
            "symbol": "SPXW",
            "expiration": "20240103",
            "start_date": "20240102",
            "end_date": "20240102",
            "interval": "1m",
        },
        "options",
    ),
    (
        "SPXW greeks 1st",
        "/v3/option/history/greeks/first_order",
        {
            "symbol": "SPXW",
            "expiration": "20240103",
            "date": "20240102",
            "interval": "1m",
        },
        "options",
    ),
    (
        "SPXW IV 1m",
        "/v3/option/history/greeks/implied_volatility",
        {
            "symbol": "SPXW",
            "expiration": "20240103",
            "start_date": "20240102",
            "end_date": "20240102",
            "interval": "1m",
        },
        "options",
    ),
    (
        "SPXW OI",
        "/v3/option/history/open_interest",
        {"symbol": "SPXW", "expiration": "*", "date": "20240102"},
        "options",
    ),
    (
        "SPXW EOD",
        "/v3/option/history/eod",
        {
            "symbol": "SPXW",
            "expiration": "*",
            "start_date": "20240102",
            "end_date": "20240102",
        },
        "options",
    ),
]

# Informational probe — professional tier, expected to fail on standard
PROBE = (
    "greeks/all probe",
    "/v3/option/history/greeks/all",
    {
        "symbol": "SPXW",
        "expiration": "20240103",
        "date": "20240102",
        "interval": "1m",
    },
    "options",
)


def count_rows(data, response_type):
    """Count rows in API response."""
    resp = data.get("response", [])
    if not resp:
        return 0
    if response_type == "index":
        return len(resp)
    if response_type == "flat":
        return len(resp) if isinstance(resp, list) else 1
    # options: grouped by contract
    total = 0
    for item in resp:
        total += len(item.get("data", []))
    return total


def run_test(label, path, params, response_type):
    """Run a single test, return (passed: bool, detail: str)."""
    url = f"{BASE_URL}{path}"
    params = {**params, "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=120)
    except requests.exceptions.ConnectionError:
        return False, "connection refused — terminal not running?"
    except requests.exceptions.Timeout:
        return False, "timeout (120s)"
    except requests.exceptions.RequestException as e:
        return False, str(e)

    if resp.status_code == 403:
        return False, "403 Forbidden — subscription insufficient"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"

    try:
        data = resp.json()
    except ValueError:
        return False, "invalid JSON response"

    rows = count_rows(data, response_type)
    return True, f"200 OK, {rows} rows"


def main(argv: list[str] | None = None) -> None:
    print(f"ThetaData pre-flight check — {BASE_URL}\n")

    passed = 0
    failed = 0
    blocked_endpoints = []

    for label, path, params, rtype in TESTS:
        ok, detail = run_test(label, path, params, rtype)
        icon = "\u2713" if ok else "\u2717"
        print(f"  [{icon}] {label:<22} {detail}")
        if ok:
            passed += 1
        else:
            failed += 1
            blocked_endpoints.append(label)

    # Informational probe for greeks/all (professional tier)
    print()
    plabel, ppath, pparams, prtype = PROBE
    ok, detail = run_test(plabel, ppath, pparams, prtype)
    if ok:
        print(f"  [!] {plabel:<22} AVAILABLE — professional tier accessible")
    else:
        print(f"  [!] {plabel:<22} not available (expected on standard plan)")

    # Verdict
    print()
    print("\u2501" * 50)
    if failed == 0:
        print("  READY TO DOWNLOAD")
    else:
        print(f"  BLOCKED \u2014 {failed} endpoint(s) inaccessible:")
        for ep in blocked_endpoints:
            print(f"    - {ep}")
    print("\u2501" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
