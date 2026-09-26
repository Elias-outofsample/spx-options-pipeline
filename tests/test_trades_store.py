from datetime import datetime

import polars as pl
import pytest

from spx_pipeline.cleaning import apply_cleaning_rules
from spx_pipeline.registry import CleaningRule, Registry


@pytest.fixture
def registry(tmp_path, project_root):
    src = project_root / "src" / "spx_pipeline" / "registry.yaml"
    dst = tmp_path / "registry.yaml"
    dst.write_text(src.read_text())
    return Registry(dst)


def test_trade_cleaning_filters_zero_price():
    """Test that filter_zero_price removes rows where price <= 0."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 31, 0, 0),
                datetime(2024, 1, 2, 9, 32, 0, 0),
                datetime(2024, 1, 2, 9, 33, 0, 0),
            ],
            "symbol": ["SPXW"] * 4,
            "price": [0.0, 5.25, 0.0, 10.50],
            "size": [1, 10, 5, 20],
            "conditions": ["", "REGULAR", "", "REGULAR"],
        }
    )

    rules = [CleaningRule(name="filter_zero_price", params={"price_gt": 0.0})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 2
    assert all(p > 0.0 for p in result.get_column("price").to_list())


def test_trade_cleaning_preserves_valid_prices():
    """Test that filter_zero_price keeps all rows when all prices > 0."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 31, 0, 0),
                datetime(2024, 1, 2, 9, 32, 0, 0),
            ],
            "symbol": ["SPXW"] * 3,
            "price": [5.25, 10.50, 7.75],
            "size": [10, 20, 15],
            "conditions": ["REGULAR", "REGULAR", "SPREAD"],
        }
    )

    rules = [CleaningRule(name="filter_zero_price", params={"price_gt": 0.0})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 3


def test_trade_cleaning_with_threshold():
    """Test filter_zero_price with a non-zero threshold (price_gt > 0)."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30, 0, 100000),
                datetime(2024, 1, 2, 9, 31, 0, 0),
                datetime(2024, 1, 2, 9, 32, 0, 0),
            ],
            "symbol": ["SPXW"] * 3,
            "price": [0.05, 5.25, 10.50],
            "size": [1, 10, 20],
            "conditions": ["", "REGULAR", "REGULAR"],
        }
    )

    rules = [CleaningRule(name="filter_zero_price", params={"price_gt": 1.0})]
    result = apply_cleaning_rules(df, rules)

    assert len(result) == 2
    assert all(p > 1.0 for p in result.get_column("price").to_list())


def test_trade_registry_config(registry):
    """Test that trades_0dte config is correctly parsed from registry."""
    cfg = registry.get("trades_0dte")
    assert cfg.source_format == "parquet"
    assert cfg.symbol == "SPXW"
    assert cfg.index_col == "timestamp"
    assert cfg.partition_cols == ["year", "month"]
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "filter_zero_price" in rule_names
