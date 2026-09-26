import pytest

from spx_pipeline.registry import Registry


@pytest.fixture
def registry(registry_path):
    return Registry(registry_path)


def test_list_datasets(registry):
    ds = registry.list_datasets()
    assert "spx_ohlc" in ds
    assert "vix_ohlc" in ds
    assert "vvix_ohlc" in ds
    assert "vol_regime" in ds
    for dte in range(8):
        assert f"greeks_{dte}dte" in ds
        # legacy td_* names are aliases of the canonical datasets
        assert f"td_greeks_{dte}dte" not in ds
        assert f"td_greeks_{dte}dte" in registry


def test_get_spx_config(registry):
    cfg = registry.get("spx_ohlc")
    assert cfg.source_format == "parquet"
    assert cfg.granularity == "1min"
    assert cfg.symbol == "SPX"
    assert cfg.compression == "zstd"
    assert cfg.row_group_size == 300_000


def test_get_greeks_config(registry):
    cfg = registry.get("greeks_0dte")
    assert cfg.source_format == "parquet"
    assert len(cfg.cleaning_rules) == 4
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "filter_bogus_expirations" in rule_names
    assert "drop_opening_bar" in rule_names
    assert "filter_illiquid" in rule_names
    assert "add_mid" in rule_names


def test_get_vvix_config(registry):
    cfg = registry.get("vvix_ohlc")
    assert cfg.source_format == "parquet"
    assert "index/vvix" in cfg.source_path


@pytest.mark.parametrize("dte", range(8))
def test_greeks_dte_config(registry, dte):
    """All greeks_Xdte datasets share identical schema and cleaning rules."""
    cfg = registry.get(f"greeks_{dte}dte")
    assert cfg.source_format == "parquet"
    assert cfg.granularity == "1min"
    assert cfg.symbol == "SPXW"
    assert cfg.index_col == "timestamp"
    assert cfg.rows_per_day == 130000
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "filter_bogus_expirations" in rule_names
    assert "drop_opening_bar" in rule_names
    assert "filter_illiquid" in rule_names
    assert "add_mid" in rule_names
    assert f"{dte}dte" in cfg.store_path


@pytest.mark.parametrize("dte", range(8))
def test_td_greeks_dte_config(registry, dte):
    """Legacy td_greeks_Xdte names resolve to the canonical greeks_Xdte datasets."""
    cfg = registry.get(f"td_greeks_{dte}dte")
    assert cfg.name == f"greeks_{dte}dte"
    assert cfg.source_format == "parquet"
    assert cfg.granularity == "1min"
    assert cfg.symbol == "SPXW"
    assert cfg.index_col == "timestamp"
    assert cfg.rows_per_day == 130000
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert len(rule_names) == 4
    assert f"{dte}dte" in cfg.store_path


def test_get_unknown_raises(registry):
    with pytest.raises(KeyError, match="not_a_dataset"):
        registry.get("not_a_dataset")


def test_regime_has_valid_from(registry):
    cfg = registry.get("vol_regime")
    rule_names = [r.name for r in cfg.cleaning_rules]
    assert "valid_from" in rule_names
    vf_rule = [r for r in cfg.cleaning_rules if r.name == "valid_from"][0]
    assert vf_rule.params["value"] == "2023-04-13"
