# Point-in-time correctness

A backtest is only as honest as the data it is allowed to see. Look-ahead bias — using
information that did not exist yet at the moment of the decision — is the most common
way a backtest lies, and in options data it hides in unusual places. This page lists
every guard in the pipeline, what it prevents, and the test that pins it.

## The rules

| Guard | Where | What it prevents | Test |
|---|---|---|---|
| **Cutoff clamp** — a query never returns rows after `cutoff_date`; the range is clamped *and* the result filtered again | `pit.PointInTimeLoader` | Training on data from the future in a walk-forward loop | `test_pit.py::TestCutoffEnforcement` |
| **Regime lag ≥ 1 session** — a daily regime computed from day *T*'s close is attached to *T+1*'s bars | `loader.load_aligned`, enforced (`ValueError` on 0) by `PointInTimeLoader` | Using the closing VIX regime to trade the same day | `test_pit.py::TestRegimeLag` |
| **Per-contract feature lag** — `with_feature_lag` shifts each contract's own history | `pit.PointInTimeLoader.with_feature_lag` | Using a Greek at bar *t* (computed from the underlying at *t*) to decide at *t* | `test_regressions.py::test_feature_lag_is_per_contract` |
| **Forward-fill only** — stale zero prints on VIX/VVIX are replaced by the previous value, never the next | `cleaning._ffill_zero_close` | Copying a later price into an earlier bar | `test_pit.py::TestNoBackwardFill` |
| **Left-closed resampling** — the 5-minute bar labelled 10:00 aggregates 10:00…10:04 only | `cleaning.resample_ohlc` | A bar containing prices from after its window | `test_cleaning.py` |
| **Days-to-expiry computed at query time** — never stored | `query.QueryEngine.query_greeks_with_dte` | Freezing one observer's clock into the data | `test_pit.py::TestDTENotStored` |
| **Warm-up excluded** — regime rows before its rolling windows are defined are dropped | `cleaning._valid_from` | Percentiles and z-scores computed on too little history | `test_registry.py::test_regime_has_valid_from` |
| **No survivorship filter** — expired contracts are kept | by design | Selecting instruments by knowing they survived | — |

## Two subtle cases

### A Greek is not a feature at its own timestamp

A 1-minute Greek at 10:00 is computed from the underlying price at 10:00. A strategy that
reads it to decide at 10:00 is using a number that only exists once that minute is over.
`with_feature_lag(df, ["delta"], lag=1)` adds `delta_lag1`, the value one bar earlier.

On an option chain the rows are `timestamp × strike × right`, sorted by timestamp first.
A plain `shift(1)` therefore hands each contract the value of **its neighbour at the same
minute**:

| timestamp | strike | right | delta | naive `shift(1)` | per-contract lag |
|---|---|---|---|---|---|
| 10:00 | 4700 | C | 0.52 | — | — |
| 10:00 | 4700 | P | −0.48 | 0.52 ← *4700 C at 10:00* | — |
| 10:00 | 4705 | C | 0.49 | −0.48 ← *4700 P at 10:00* | — |
| 10:01 | 4700 | C | 0.55 | … | 0.52 ← *4700 C at 10:00* |

The naive column is not lagged at all: it is contemporaneous information from another
contract. The pipeline shifts **within** each contract (`shift(n).over(contract key)`),
where the key is whichever of `symbol, expiration, strike, right` are present. If a
contract is missing a bar, its lag is the last bar it *has* — older information, never
newer.

### The regime of the first session of a window

The daily regime is shifted by one session before it is joined to intraday bars. If the
regime is read only over the requested window, its first session has no predecessor and
silently gets no regime at all. `load_aligned` reads a few sessions *before* the window,
shifts, then trims back to the window.

## Using it in a walk-forward loop

```python
from datetime import date
from spx_pipeline import DataLoader, PointInTimeLoader

loader = DataLoader(store_root="data")
for cutoff in [date(2023, 6, 30), date(2023, 12, 29), date(2024, 6, 28)]:
    pit = PointInTimeLoader(loader, cutoff_date=cutoff, feature_lag_bars=1)
    train = pit.load("greeks_0dte", ("2023-01-03", cutoff))
    train = pit.with_feature_lag(train, ["delta", "vega", "implied_vol"])
    ...  # fit on `train`, evaluate on the next window
```
