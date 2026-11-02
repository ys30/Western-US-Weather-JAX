"""Check spatial replication region separation and evaluation contract."""
from pathlib import Path
import re


def test_plains_replication_contract():
    text=(Path(__file__).resolve().parents[1]/".github/workflows/era5-matched-southern-plains.yml").read_text()
    assert "--lat-min 32 --lat-max 38 --lon-min -101 --lon-max -95" in text
    assert "--lat-min 31 --lat-max 43 --lon-min -121 --lon-max -109" in text
    assert -109 < -101  # disjoint source and target longitude bands
    assert -102 < -101  # disjoint Southwest 2022 and new target longitude bands
    assert "--train-until 2021-01-01" in text
    assert "--validate-until 2022-01-01" in text
    assert "--test-start 2022-01-01" in text
    assert "historical source_mean" in text
    assert "42 7 123" in text
    assert "3 7 14 28" in text
    assert "matched_direct_bootstrap.py" in text
