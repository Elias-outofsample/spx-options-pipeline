# Adding a dataset

Three levels, from no code at all to a new ingestion module.

## Level 1 — a standard dataset: YAML only

If the new data has ordinary columns and needs only existing cleaning rules, declare it
in `registry.yaml`:

```yaml
datasets:
  trades_1dte:
    description: "SPXW 1DTE trade prints per strike and right"
    source_format: parquet
    source_path: "spx trade/year={year}/month={month:02d}/SPXW_trade_1dte_1m_{date}.parquet"
    store_path: "store/trades_1dte/year={year}/month={month:02d}/{date}.parquet"
    partition_cols: [year, month]
    granularity: tick
    symbol: SPXW
    schema:
      <<: *contract_cols          # symbol, expiration, strike, right, timestamp
      size: int16
      price: float32
    drop_cols: [ext_condition1, ext_condition2]
    cleaning_rules:
      - filter_zero_price: {price_gt: 0.0}
    index_col: timestamp
```

It is then ingested by `spx-pipeline migrate --dataset trades_1dte`, loaded with
`DataLoader.load("trades_1dte", …)`, and inherits partition pruning, filter pushdown, the
cache and the point-in-time loader.

## Level 2 — a new cleaning rule

Write a pure function and register it:

```python
# cleaning.py
def _add_trade_direction(df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Tick rule: +1 on an uptick, -1 otherwise, per contract."""
    key = contract_key(df)
    up = pl.col("price") > pl.col("price").shift(1).over(key)
    return df.with_columns(pl.when(up).then(1).otherwise(-1).alias("direction"))

RULE_HANDLERS["add_trade_direction"] = _add_trade_direction
```

then reference it from the YAML: `- add_trade_direction: {}`. Rules must only look at the
current row and earlier ones: anything computed from a later row is look-ahead.

## Level 3 — a structurally different source

An order book (L2), full-chain snapshots every N seconds, or nested multi-symbol records
need their own ingestion module next to `store.py`. Everything downstream — `QueryEngine`,
`FeatherCache`, `DataLoader`, `PointInTimeLoader` — stays unchanged as long as the module
writes flat Parquet files under the store layout.

## Storage planning

From the 1-minute store (three years ≈ 2.9 GB with ZSTD):

| Scenario | Estimated size |
|---|---|
| 7 years, 1-minute | ≈ 6 GB |
| 7 years, tick (10× the 1-minute volume) | ≈ 60 GB |
| 7 years, tick (50×) | ≈ 300 GB — needs a retention policy (N months of ticks + 1-minute archive) |

Bulk tick files are streamed in 1M-row batches, so ingestion memory stays flat whatever
the file size; DuckDB's `memory_limit` bounds the query side.
