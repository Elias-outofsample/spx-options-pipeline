from datetime import date

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spx_pipeline.cleaning import apply_cleaning_rules
from spx_pipeline.registry import CleaningRule, Registry


@pytest.fixture
def registry(tmp_path, project_root):
    src = project_root / "src" / "spx_pipeline" / "registry.yaml"
    dst = tmp_path / "registry.yaml"
    dst.write_text(src.read_text())
    return Registry(dst)


@pytest.fixture
def synthetic_oi_parquet(tmp_path):
    """Write a synthetic OI parquet file and return the path."""
    out = tmp_path / "oi" / "20240102.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "date": pa.array([date(2024, 1, 2)] * 6, type=pa.date32()),
            "symbol": ["SPX"] * 6,
            "expiry": pa.array([date(2024, 1, 19)] * 6, type=pa.date32()),
            "strike": pa.array([4700.0, 4750.0, 4800.0, 4700.0, 4750.0, 4800.0], type=pa.float32()),
            "right": ["CALL", "CALL", "CALL", "PUT", "PUT", "PUT"],
            "oi": pa.array([1500, 0, 3200, 0, 2100, 0], type=pa.int32()),
        }
    )
    pq.write_table(table, str(out), compression="zstd", compression_level=3)
    return out


def test_oi_parquet_roundtrip(synthetic_oi_parquet):
    """Test that synthetic OI data round-trips through Parquet correctly."""
    table = pq.read_table(str(synthetic_oi_parquet))

    assert "oi" in table.column_names
    assert "strike" in table.column_names
    assert "right" in table.column_names
    assert "symbol" in table.column_names
    assert table.num_rows == 6


def test_oi_cleaning_filters_zero_oi():
    """Test that filter_zero_oi removes rows with oi <= 0."""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 4,
            "symbol": ["SPX"] * 4,
            "expiry": [date(2024, 1, 19)] * 4,
            "strike": [4700.0, 4750.0, 4800.0, 4850.0],
            "right": ["CALL", "CALL", "PUT", "PUT"],
            "oi": [1500, 0, 3200, 0],
        }
    )

    rules = [CleaningRule(name="filter_zero_oi", params={"oi_gt": 0})]
    result = apply_cleaning_rules(df, rules)

    # Two rows had oi=0 and should be removed
    assert len(result) == 2
    assert all(oi > 0 for oi in result.get_column("oi").to_list())


def test_oi_cleaning_preserves_nonzero():
    """Test that filter_zero_oi keeps all rows when all OI > 0."""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 3,
            "symbol": ["SPX"] * 3,
            "expiry": [date(2024, 1, 19)] * 3,
            "strike": [4700.0, 4750.0, 4800.0],
            "right": ["CALL", "PUT", "CALL"],
            "oi": [100, 200, 300],
        }
    )

    rules = [CleaningRule(name="filter_zero_oi", params={"oi_gt": 0})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 3


def test_oi_registry_config(registry):
    """Test that oi config is correctly parsed from registry."""
    cfg = registry.get("oi")
    assert cfg.source_format == "parquet"
    assert cfg.symbol == "SPXW"
    assert cfg.index_col == "timestamp"
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "filter_bogus_expirations" in rule_names
    assert "filter_zero_oi" in rule_names
    filter_rule = next(r for r in cfg.cleaning_rules if r.name == "filter_zero_oi")
    assert filter_rule.params.get("open_interest_gt", filter_rule.params.get("oi_gt", 0)) == 0
