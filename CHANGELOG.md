# Changelog

## 0.2.0 — public release

- Package restructured as `spx_pipeline` (src layout) with a `spx-pipeline` CLI:
  `datasets`, `demo`, `migrate`, `validate`, `regime`, `thetadata …`.
- Registry rewritten with YAML anchors (1,190 → 369 lines); legacy `td_*` names kept
  as aliases resolving to one store and one cache.
- Synthetic market generator (Black-Scholes chain on a skewed smile) powering an
  end-to-end demo, the benchmark and new tests.
- Fixes, each pinned by a regression test that fails on 0.1: per-contract feature lag;
  chain-aware duplicate filter; bulk tick ingestion; per-day tick sources; date filtering
  on non-`timestamp` index columns; `iv`/`implied_vol` handling; regime look-back at the
  start of a window; ThetaData splitter aligned with the registry.
- Hardening: quoted identifiers, validated memory budget, explicit exceptions instead of
  `assert`, no logging configuration at import time, credentials read from the
  environment only.
- Python 3.11–3.13 supported and tested; mypy-clean core; ruff lint and format.

## 0.1.0

- DuckDB/Polars pipeline over a Hive-partitioned Parquet store: registry, ingestion,
  query engine, Feather cache, cleaning rules, point-in-time loader.
