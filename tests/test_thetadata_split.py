"""The ThetaData splitter writes monthly vendor files into the canonical daily store."""

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from spx_pipeline.registry import DEFAULT_REGISTRY, Registry
from spx_pipeline.vendors.thetadata import split_monthly


def test_monthly_file_is_split_into_canonical_sessions(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    src = vendor / "options" / "quotes" / "0dte"
    src.mkdir(parents=True)
    ts = [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 3, 10, 0)]
    pq.write_table(
        pa.table(
            {
                "symbol": ["SPXW"] * 2,
                "expiration": ["2024-01-02", "2024-01-03"],
                "strike": pa.array([4700.0, 4705.0], pa.float32()),
                "right": ["CALL", "PUT"],
                "timestamp": pa.array(ts, pa.timestamp("us")),
                "bid": pa.array([1.0, 2.0], pa.float32()),
                "ask": pa.array([1.1, 2.1], pa.float32()),
                "bid_exchange": pa.array([1, 1], pa.int32()),
            }
        ),
        src / "2024_01.parquet",
    )
    monkeypatch.setattr(split_monthly, "THETADATA_ROOT", vendor)

    registry = Registry(DEFAULT_REGISTRY)
    stats = split_monthly.migrate_dataset("td_quotes_0dte", tmp_path, registry)  # legacy name

    assert stats["ingested"] == 2 and stats["errors"] == 0
    out = tmp_path / "store" / "quotes_0dte" / "year=2024" / "month=01" / "20240103.parquet"
    table = pq.read_table(out)
    assert table.num_rows == 1
    assert "bid_exchange" not in table.column_names  # dropped per the registry
    assert table.schema.field("timestamp").type == pa.timestamp("ns")
