from datetime import datetime

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spx_pipeline.cleaning import apply_cleaning_rules
from spx_pipeline.registry import CleaningRule, Registry
from spx_pipeline.store import DataStore


@pytest.fixture
def raw_quote_dir(tmp_path):
    """Create synthetic SPX quote parquet files at the registry source_path location."""
    raw = tmp_path / "raw"

    # quotes_0dte: source_path = "options/quotes/0dte/{date}.parquet"
    quote_dir = raw / "options" / "quotes" / "0dte"
    quote_dir.mkdir(parents=True)

    timestamps = [
        datetime(2024, 1, 2, 9, 30, 0, 100000),
        datetime(2024, 1, 2, 9, 31, 0, 0),
        datetime(2024, 1, 2, 9, 32, 0, 0),
    ]

    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.timestamp("us")),
            "symbol": ["SPXW"] * 3,
            "expiration": ["20240102"] * 3,
            "strike": pa.array([4700.0, 4750.0, 4800.0], type=pa.float32()),
            "right": ["CALL", "CALL", "PUT"],
            "bid_size": pa.array([10, 20, 30], type=pa.int32()),
            "bid": pa.array([1.5, 2.0, 3.5], type=pa.float32()),
            "ask_size": pa.array([15, 25, 35], type=pa.int32()),
            "ask": pa.array([1.6, 2.1, 3.6], type=pa.float32()),
        }
    )
    pq.write_table(table, str(quote_dir / "20240102.parquet"))

    return raw


@pytest.fixture
def store_dir(tmp_path):
    return tmp_path / "store_output"


@pytest.fixture
def registry(tmp_path, project_root):
    src = project_root / "src" / "spx_pipeline" / "registry.yaml"
    dst = tmp_path / "registry.yaml"
    dst.write_text(src.read_text())
    return Registry(dst)


def test_ingest_quotes_0dte(raw_quote_dir, store_dir, registry):
    """Test that SPX quote data is ingested correctly."""
    store = DataStore(raw_quote_dir, store_dir, registry)
    stats = store.ingest_dataset("quotes_0dte")

    assert stats["ingested"] == 1
    assert stats["errors"] == 0
    assert stats["bytes_written"] > 0

    out_path = store_dir / "store" / "quotes_0dte" / "year=2024" / "month=01" / "20240102.parquet"
    assert out_path.exists()

    schema = pq.read_schema(str(out_path))
    assert "timestamp" in schema.names
    assert "bid" in schema.names
    assert "ask" in schema.names


def test_quote_cleaning_filters_zero_bid():
    """Test that filter_zero_bid removes rows where bid <= 0."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 31, 0, 0),
                datetime(2024, 1, 2, 9, 32, 0, 0),
            ],
            "symbol": ["SPXW"] * 3,
            "bid": [0.0, 2.5, 3.0],
            "ask": [0.1, 2.6, 3.1],
            "bid_size": [10, 20, 30],
            "ask_size": [15, 25, 35],
        }
    )

    rules = [CleaningRule(name="filter_zero_bid", params={"bid_gt": 0.0})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 2
    assert all(b > 0.0 for b in result.get_column("bid").to_list())


def test_quote_cleaning_dedup_timestamps():
    """Test that filter_duplicate_timestamps keeps last occurrence."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 30, 0, 100000),  # duplicate
                datetime(2024, 1, 2, 9, 31, 0, 0),
            ],
            "symbol": ["SPXW", "SPXW", "SPXW"],
            "bid": [1.0, 1.5, 2.0],
            "ask": [1.1, 1.6, 2.1],
        }
    )

    rules = [CleaningRule(name="filter_duplicate_timestamps", params={"keep": "last"})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 2
    dup_ts_rows = result.filter(pl.col("timestamp") == datetime(2024, 1, 2, 9, 30, 0, 100000))
    assert len(dup_ts_rows) == 1
    assert dup_ts_rows.get_column("bid")[0] == 1.5


def test_quote_combined_cleaning():
    """Test that filter_zero_bid + dedup chain correctly."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 31, 0, 0),
                datetime(2024, 1, 2, 9, 32, 0, 0),
            ],
            "symbol": ["SPXW"] * 4,
            "bid": [0.0, 1.5, 2.0, 3.5],
            "ask": [0.1, 1.6, 2.1, 3.6],
        }
    )

    rules = [
        CleaningRule(name="filter_zero_bid", params={"bid_gt": 0.0}),
        CleaningRule(name="filter_duplicate_timestamps", params={"keep": "last"}),
    ]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 3


def test_quote_registry_config(registry):
    """Test that quotes_0dte config is correctly parsed from registry."""
    cfg = registry.get("quotes_0dte")
    assert cfg.source_format == "parquet"
    assert cfg.granularity == "1min"
    assert cfg.symbol == "SPXW"
    assert cfg.index_col == "timestamp"
    assert cfg.partition_cols == ["year", "month"]
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "filter_zero_bid" in rule_names
    assert "filter_bogus_expirations" in rule_names
