"""Static safeguards against leaking target test-year data in 2023 replication."""
from pathlib import Path

def test_2023_temporal_replication_contract():
    workflow=(Path(__file__).resolve().parents[1] / ".github/workflows/era5-matched-crossyear-2023.yml").read_text()
    assert "--end 2023-12-31" in workflow
    assert "--test-start 2023-01-01" in workflow
    assert "--train-until 2021-01-01" in workflow
    assert "--validate-until 2022-01-01" in workflow
    assert "historical source_mean" in workflow
    assert "42 7 123" in workflow
    assert "3 7 14 28" in workflow
    assert "matched_direct_bootstrap.py" in workflow
