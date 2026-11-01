import csv,math
import pytest
from scripts.summarize_monthly_diagnostics import summarize

def test_monthly_summary_preserves_empty_hot_subset(tmp_path):
    p=tmp_path/"monthly.csv"
    with p.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["month","time_steps","hot_gridcell_times","cnn_rmse_k","reference_rmse_k","cnn_hot_rmse_k","reference_hot_rmse_k"])
        w.writerow(["2022-01",124,0,2.0,4.0,"nan","nan"])
        w.writerow(["2022-02",112,5,3.0,4.0,2.0,4.0])
    d=summarize(p)
    assert d[0]["hot_rmse_relative_change_pct"] is None
    assert d[0]["all_rmse_relative_change_pct"]==pytest.approx(50.)
    assert d[1]["hot_rmse_relative_change_pct"]==pytest.approx(50.)
