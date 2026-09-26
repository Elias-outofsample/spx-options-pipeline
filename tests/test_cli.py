from __future__ import annotations

from spx_pipeline.cli import main


def test_datasets_lists_the_registry(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "greeks_0dte" in out and "20 datasets" in out


def test_demo_runs_end_to_end(capsys, tmp_path):
    assert main(["demo", "--workdir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MISMATCH" not in out
    assert out.count("OK") == 2
    assert (tmp_path / "store" / "greeks_0dte").is_dir()
