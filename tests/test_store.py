from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spx_pipeline.registry import Registry
from spx_pipeline.store import DataStore


@pytest.fixture
def raw_dir(tmp_path):
    """Create minimal raw files matching current registry source_paths."""
    raw = tmp_path / "raw"

    # vix_ohlc: source_path = "index/vix/{date}.parquet"
    vix_dir = raw / "index" / "vix"
    vix_dir.mkdir(parents=True)
    table = pa.table(
        {
            "timestamp": pa.array(
                [1704196200000000000, 1704196260000000000], type=pa.timestamp("ns")
            ),
            "open": pa.array([13.5, 13.6], type=pa.float32()),
            "high": pa.array([13.7, 13.8], type=pa.float32()),
            "low": pa.array([13.4, 13.5], type=pa.float32()),
            "close": pa.array([13.6, 13.7], type=pa.float32()),
        }
    )
    pq.write_table(table, str(vix_dir / "20240102.parquet"))

    # spx_ohlc: source_path = "index/spx/{date}.parquet"
    spx_dir = raw / "index" / "spx"
    spx_dir.mkdir(parents=True)
    table = pa.table(
        {
            "timestamp": pa.array(
                [1704196200000000000, 1704196260000000000], type=pa.timestamp("ns")
            ),
            "open": pa.array([4750.0, 4751.0], type=pa.float32()),
            "high": pa.array([4752.0, 4753.0], type=pa.float32()),
            "low": pa.array([4749.0, 4750.0], type=pa.float32()),
            "close": pa.array([4751.0, 4752.0], type=pa.float32()),
        }
    )
    pq.write_table(table, str(spx_dir / "20240102.parquet"))

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


def test_ingest_vix_ohlc(raw_dir, store_dir, registry):
    store = DataStore(raw_dir, store_dir, registry)
    stats = store.ingest_dataset("vix_ohlc", dates=[date(2024, 1, 2)])

    assert stats["ingested"] == 1
    assert stats["errors"] == 0
    assert stats["bytes_written"] > 0

    out_path = store_dir / "store" / "vix_ohlc" / "year=2024" / "month=01" / "20240102.parquet"
    assert out_path.exists()

    schema = pq.read_schema(str(out_path))
    assert "close" in schema.names


def test_ingest_parquet(raw_dir, store_dir, registry):
    store = DataStore(raw_dir, store_dir, registry)
    stats = store.ingest_dataset("spx_ohlc", dates=[date(2024, 1, 2)])

    assert stats["ingested"] == 1
    out_path = store_dir / "store" / "spx_ohlc" / "year=2024" / "month=01" / "20240102.parquet"
    assert out_path.exists()

    schema = pq.read_schema(str(out_path))
    assert "close" in schema.names


def test_skip_already_ingested(raw_dir, store_dir, registry):
    store = DataStore(raw_dir, store_dir, registry)
    store.ingest_dataset("vix_ohlc", dates=[date(2024, 1, 2)])
    stats2 = store.ingest_dataset("vix_ohlc", dates=[date(2024, 1, 2)])
    assert stats2["skipped"] == 1
    assert stats2["ingested"] == 0


def test_discover_dates(raw_dir, registry):
    store = DataStore(raw_dir, raw_dir, registry)
    config = registry.get("vix_ohlc")
    dates = store.discover_dates(config)
    assert date(2024, 1, 2) in dates
