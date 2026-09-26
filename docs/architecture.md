# Architecture

```
raw vendor files ──► DataStore ──► store/ (Parquet ZSTD, year=/month=) ──► QueryEngine (DuckDB)
                                                                                │ Arrow
                     PointInTimeLoader ◄── DataLoader ◄── cleaning rules ◄── FeatherCache (mmap)
                            │
                            ▼
                     backtest / model  (Polars DataFrames, float64)
```

Every stage is driven by one declaration, [`registry.yaml`](../src/spx_pipeline/registry.yaml):
where a dataset's raw files live, where the store keeps them, the Arrow schema to enforce,
the columns to drop, and the ordered cleaning rules. The code never special-cases a
dataset by name.

## Modules

| Module | Role |
|---|---|
| `registry.py` + `registry.yaml` | Typed `DatasetConfig` objects. YAML anchors factor the eight Greeks datasets into one template; `aliases` keep legacy names working. |
| `store.py` — `DataStore` | Raw → store. One file at a time (bounded disk and memory), idempotent (skips a file whose store copy is newer), three layouts: one file per session, a single file, or monthly bulk files streamed in 1M-row batches. Rejects options datasets without naive timestamps. |
| `query.py` — `QueryEngine` | DuckDB over the store with Hive partition discovery. Date bounds and filters are pushed down to the Parquet reader; filter values are bind parameters, identifiers are always quoted; `memory_limit` caps RAM. Derives days-to-expiry at query time. |
| `cache.py` — `FeatherCache` | Query results as **uncompressed** Feather, read back memory-mapped (a compressed cache would have to be decompressed into RAM, defeating the mmap). Keyed by a SHA-256 of the query parameters; LRU eviction under a size budget. |
| `cleaning.py` | Twelve pure rules, dispatched by name from the registry. |
| `loader.py` — `DataLoader` | The entry point: registry → cache → query → Polars → cleaning → resample → cache → float64. `load_aligned` joins several intraday series and the lagged daily regime. |
| `pit.py` — `PointInTimeLoader` | Cutoff clamp, minimum regime lag, per-contract feature lags. See [point-in-time.md](point-in-time.md). |
| `synthetic.py` | A self-consistent synthetic feed (GBM underlying, OU volatility, Black-Scholes chain on a skewed smile) in the vendor layout, including the feed's known defects. Powers the demo, the benchmark and part of the tests. |
| `vendors/thetadata/` | Acquisition: a concurrent, resumable downloader for the ThetaData Terminal, a splitter from monthly vendor files to the daily store, and two validators (raw files, store). |
| `tools/` | `migrate`, `validate`, and `regime` (daily VIX/VVIX regime: rolling percentiles, z-scores, labels). |
| `cli.py` | `spx-pipeline datasets | demo | migrate | validate | regime | thetadata …` |

## Design decisions

**DuckDB for I/O, Polars for transforms, no pandas on the hot path.** DuckDB reads only
the row groups a query can match and stays inside a memory budget; Polars consumes its
Arrow output without a copy. pandas appears only at the vendor boundary (the downloader)
and nowhere in the query path.

**float32 on disk, float64 in memory.** Prices and Greeks are stored as float32 (half
the disk, twice the rows per page) and up-cast to float64 when loaded, so arithmetic
never accumulates single-precision error. `upcast_float64=False` keeps float32 when memory
matters more.

**Partitioning by year and month, one file per session.** A session is the natural unit
of re-ingestion and of most queries; `year=/month=` directories let DuckDB prune whole
months from a date filter before opening any file. Row groups of 300k rows keep
statistics fine-grained enough for strike filters to skip data inside a file.

**Days-to-expiry is derived, never stored.** It is a function of *when* you look: a row
stored with `dte = 1` is wrong for anyone reading it at a different time of day.
`dte_fractional` counts seconds to the 16:00 ET settlement, which requires the store to
hold naive Eastern-Time timestamps — enforced at ingestion.

**Cleaning at load time, not at ingestion.** The store keeps the vendor's rows (minus
dropped columns); rules run when data is loaded. Changing a threshold never requires
re-ingesting six years of data, and `apply_cleaning=False` always shows what the vendor
actually sent.

**Registry over code.** A new standard dataset is a YAML block — schema, paths, rules —
and inherits ingestion, pushdown, caching, cleaning and point-in-time handling. See
[extending.md](extending.md).

## Scale

On the private store this was built for (ThetaData feed, not redistributable): eight
Greeks datasets (0 to 7 days to expiry) from 2019, 1,425 to 1,491 sessions each at roughly
130,000 one-minute rows per session, plus quotes, IV surfaces, open interest, end-of-day
chains and trade prints. The layout, the budgets and the streaming ingestion exist
because of that volume; the synthetic feed reproduces its shape at laptop size.
