import csv
import pytest
from scripts.audit_published_bootstrap import audit

def test_published_bootstrap_results():
    result = audit("results/bootstrap_block_length_sensitivity_v2.4.csv")
    assert result["comparisons"] == 36
    assert result["negative_interval_upper_bounds"] == 36

def test_detects_corrupted_interval(tmp_path):
    with open("results/bootstrap_block_length_sensitivity_v2.4.csv", newline="") as f:
        rows=list(csv.DictReader(f))
    rows[0]["ci_high_k"]="0.5"
    p=tmp_path/"corrupt.csv"
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError):
        audit(p)
