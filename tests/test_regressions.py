"""Regression tests for defects found while preparing the public release.

Each test fails on the previous implementation and documents why the fix matters.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spx_pipeline.cleaning import apply_cleaning_rules
from spx_pipeline.params import QueryParams
from spx_pipeline.pit import PointInTimeLoader
from spx_pipeline.query import QueryEngine
from spx_pipeline.registry import CleaningRule, Registry
from spx_pipeline.store import DataStore


def _chain(n_bars: int = 4, strikes=(4700.0, 4705.0), rights=("CALL", "PUT")) -> pl.DataFrame:
    """A tiny option chain where each contract's delta encodes (strike, right, bar)."""
    rows = []
    t0 = datetime(2024, 1, 2, 9, 31)
    for i in range(n_bars):
        for k in strikes:
            for r in rights:
                code = (k - 4700.0) * 100 + (0 if r == "CALL" else 10) + i
                rows.append(
                    {
                        "timestamp": t0 + timedelta(minutes=i),
                        "symbol": "SPXW",
                        "expiration": "2024-01-02",
                        "strike": k,
                        "right": r,
                        "delta": float(code),
                    }
                )
    return pl.DataFrame(rows).sort(["timestamp", "strike", "right"])


# --------------------------------------------------------------- feature lag


def test_feature_lag_is_per_contract(loader):
    """A plain shift(1) on a chain hands each contract its neighbour's *same-minute* value."""
    pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))
    out = pit.with_feature_lag(_chain(), ["delta"], lag=1)
    for _key, grp in out.group_by(["strike", "right"]):
        grp = grp.sort("timestamp")
        assert grp["delta_lag1"][0] is None, "first bar of each contract has no history"
        assert (
            grp["delta_lag1"].slice(1).to_list() == grp["delta"].slice(0, grp.height - 1).to_list()
        )


def test_feature_lag_single_series_unchanged(loader):
    pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))
    df = pl.DataFrame(
        {"timestamp": [datetime(2024, 1, 2, 9, 31 + i) for i in range(3)], "delta": [0.1, 0.2, 0.3]}
    )
    assert pit.with_feature_lag(df, ["delta"], lag=1)["delta_lag1"].to_list() == [None, 0.1, 0.2]


def test_feature_lag_rejects_negative(loader):
    pit = PointInTimeLoader(loader, cutoff_date=date(2024, 6, 30))
    with pytest.raises(ValueError):
        pit.with_feature_lag(_chain(), ["delta"], lag=-1)


# ------------------------------------------------------------------ cleaning


def test_dedupe_keeps_the_whole_chain():
    """Deduplicating on (timestamp, symbol) alone collapsed a chain to one row per minute."""
    chain = _chain()
    doubled = pl.concat([chain, chain.head(3)])
    out = apply_cleaning_rules(
        doubled, [CleaningRule("filter_duplicate_timestamps", {"keep": "last"})]
    )
    assert out.height == chain.height


def test_filter_illiquid_reads_iv_column_on_tick_data():
    """The tick dataset renames implied_vol to iv at ingestion; the rule must follow."""
    df = pl.DataFrame({"bid": [1.0, 1.0], "iv": [0.2, 250.0]})
    out = apply_cleaning_rules(df, [CleaningRule("filter_illiquid", {"implied_vol_lte": 100.0})])
    assert out["iv"].to_list() == [0.2]


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-02", "1882-01-01", "2089-06-30"],
        [date(2024, 1, 2), date(1882, 1, 1), date(2089, 6, 30)],
    ],
)
def test_bogus_expirations_any_dtype(values):
    df = pl.DataFrame({"expiration": values})
    out = apply_cleaning_rules(df, [CleaningRule("filter_bogus_expirations", {})])
    assert out.height == 1


def test_outlier_spread_drops_zero_mid():
    df = pl.DataFrame({"bid": [0.0, 1.00, 1.00], "ask": [0.0, 1.02, 1.50]})
    out = apply_cleaning_rules(df, [CleaningRule("filter_outlier_spread", {"max_spread_pct": 5.0})])
    assert out["ask"].to_list() == [1.02]


# --------------------------------------------------------------------- query


@pytest.fixture
def chain_engine(tmp_path, registry_path):
    registry = Registry(registry_path)
    root = tmp_path / "store" / "greeks_0dte" / "year=2024" / "month=01"
    root.mkdir(parents=True)
    df = _chain().with_columns(pl.col("strike").cast(pl.Float32))
    pq.write_table(df.to_arrow(), root / "20240102.parquet")
    with QueryEngine(tmp_path, registry) as engine:
        yield engine


def test_in_operator_binds_a_list(chain_engine):
    params = QueryParams(
        "greeks_0dte",
        date(2024, 1, 2),
        date(2024, 1, 2),
        filters=(("right", "IN", ("PUT",)), ("strike", "IN", (4705.0,))),
    )
    out = chain_engine.query(params)
    assert set(out.column("right").to_pylist()) == {"PUT"}
    assert set(out.column("strike").to_pylist()) == {4705.0}


def test_not_in_operator(chain_engine):
    params = QueryParams(
        "greeks_0dte", date(2024, 1, 2), date(2024, 1, 2), filters=(("right", "not  in", ["PUT"]),)
    )
    assert set(chain_engine.query(params).column("right").to_pylist()) == {"CALL"}


def test_column_names_are_escaped(chain_engine):
    """Column names are always quoted: a hostile name is an unknown column, not SQL."""
    params = QueryParams(
        "greeks_0dte", date(2024, 1, 2), date(2024, 1, 2), columns=('strike" FROM x; --',)
    )
    with pytest.raises(Exception, match="(?i)not found|referenced column"):
        chain_engine.query(params)


def test_disallowed_operator(chain_engine):
    params = QueryParams(
        "greeks_0dte", date(2024, 1, 2), date(2024, 1, 2), filters=(("strike", "; DROP", 1),)
    )
    with pytest.raises(ValueError, match="Disallowed"):
        chain_engine.query(params)


def test_memory_budget_is_validated(tmp_path, registry_path):
    with pytest.raises(ValueError):
        QueryEngine(tmp_path, Registry(registry_path), max_memory="8GB'; DROP TABLE x; --")


def test_date_filter_applies_to_non_timestamp_index(tmp_path, registry_path):
    """`eod` is indexed on `created`; the date range used to be silently ignored."""
    registry = Registry(registry_path)
    cfg = registry.get("eod")
    for d in (date(2024, 1, 2), date(2024, 1, 3)):
        path = tmp_path / cfg.store_path.format(year=d.year, month=d.month, date=f"{d:%Y%m%d}")
        path.parent.mkdir(parents=True, exist_ok=True)
        created = datetime(d.year, d.month, d.day, 17, 0)
        pq.write_table(
            pa.table(
                {
                    "created": pa.array([created], pa.timestamp("ns")),
                    "strike": pa.array([4700.0], pa.float32()),
                }
            ),
            path,
        )
    with QueryEngine(tmp_path, registry) as engine:
        out = engine.query(QueryParams("eod", date(2024, 1, 3), date(2024, 1, 3)))
    assert out.num_rows == 1
    assert out.column("created")[0].as_py().date() == date(2024, 1, 3)


# --------------------------------------------------------------------- store


def test_bulk_monthly_files_are_streamed_into_the_store(tmp_path, registry_path):
    """`{month:02d}` was formatted with a string: every bulk file failed, silently."""
    registry = Registry(registry_path)
    src = tmp_path / "raw" / "greeks_tick" / "year=2024" / "month=01"
    src.mkdir(parents=True)
    ts = [datetime(2024, 1, 2, 10, 0, 0, i) for i in range(4)]
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array(ts, pa.timestamp("us")),
                "symbol": ["SPXW"] * 4,
                "strike": pa.array([4700.0] * 4, pa.float32()),
                "right": ["CALL", "PUT"] * 2,
                "expiration": pa.array([date(2024, 1, 2)] * 4, pa.date32()),
                "bid": pa.array([1.0] * 4, pa.float32()),
                "ask": pa.array([1.1] * 4, pa.float32()),
                "delta": pa.array([0.5, -0.5] * 2, pa.float32()),
                "gamma": pa.array([0.01] * 4, pa.float32()),
                "vega": pa.array([1.0] * 4, pa.float32()),
                "theta": pa.array([-2.0] * 4, pa.float32()),
                "implied_vol": pa.array([0.2] * 4, pa.float32()),
                "underlying_price": pa.array([4700.0] * 4, pa.float32()),
            }
        ),
        src / "chunk_000.parquet",
    )
    stats = DataStore(tmp_path / "raw", tmp_path, registry).ingest_dataset("greeks_tick")
    assert stats == {**stats, "ingested": 1, "errors": 0}
    out = pq.read_table(
        tmp_path / "store" / "greeks_tick" / "year=2024" / "month=01" / "chunk_000.parquet"
    )
    assert {"expiry", "iv", "right"} <= set(out.column_names)  # renamed + call/put kept


def test_per_date_tick_source_is_ingested(tmp_path, registry_path):
    """`trades_0dte` is tick data stored one file per day: it used to ingest nothing."""
    registry = Registry(registry_path)
    cfg = registry.get("trades_0dte")
    d = date(2024, 1, 2)
    src = tmp_path / "raw" / cfg.source_path.format(year=d.year, month=d.month, date=f"{d:%Y%m%d}")
    src.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([datetime(2024, 1, 2, 10)], pa.timestamp("ns")),
                "strike": pa.array([4700.0], pa.float32()),
                "price": pa.array([1.5], pa.float32()),
            }
        ),
        src,
    )
    stats = DataStore(tmp_path / "raw", tmp_path, registry).ingest_dataset("trades_0dte")
    assert stats["ingested"] == 1 and stats["errors"] == 0


def test_options_dataset_must_use_naive_timestamps(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text(
        "datasets:\n  bad:\n    source_format: parquet\n    source_path: 'x/{date}.parquet'\n"
        "    store_path: 'store/x/{date}.parquet'\n    schema: {timestamp: 'timestamp[ns, UTC]',"
        " strike: float32}\n"
    )
    with pytest.raises(ValueError, match="naive"):
        DataStore(tmp_path, tmp_path, Registry(reg)).ingest_dataset("bad")


# ------------------------------------------------------------------- registry


def test_aliases_resolve_to_one_dataset(loader):
    assert loader.registry.resolve("td_greeks_0dte") == "greeks_0dte"
    a = loader.load("td_greeks_0dte", "2024-01-02")
    b = loader.load("greeks_0dte", "2024-01-02")
    assert a.equals(b)


def test_unknown_alias_target_is_rejected(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text("aliases: {old: missing}\ndatasets: {}\n")
    with pytest.raises(ValueError, match="unknown dataset"):
        Registry(reg)


# -------------------------------------------------------------------- regime


def test_first_session_of_a_window_gets_the_previous_regime(loader):
    """The regime was read from the window only, so its first session got none."""
    df = loader.load_aligned(("2024-01-03", "2024-01-03"), datasets=["spx_ohlc", "vol_regime"])
    regime = loader.load("vol_regime", ("2024-01-02", "2024-01-02"), apply_cleaning=False)
    assert df["vix"].null_count() == 0
    assert df["vix"][0] == pytest.approx(regime["vix"][0], rel=1e-6)
