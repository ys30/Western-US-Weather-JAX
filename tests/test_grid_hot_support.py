import csv
import pytest
from scripts.grid_hot_support import summarize

def test_threshold_changes_denominator(tmp_path):
    files=[]
    for seed in (42,7,123):
        p=tmp_path/f"seed-{seed}-fields.grid.csv"
        with p.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["lat_index","lon_index","hot_sample_count","cnn_hot_rmse_k","reference_hot_rmse_k"])
            w.writerow([0,0,2,1,2])
            w.writerow([0,1,12,1,2])
        files.append(p)
    assert summarize(files,(1,10))[0]["all_three_improve"]==2
    assert summarize(files,(1,10))[1]["eligible_cells"]==1

def test_reject_inconsistent_hot_support(tmp_path):
    files=[]
    for seed,n in ((42,2),(7,3),(123,2)):
        p=tmp_path/f"seed-{seed}-fields.grid.csv"
        p.write_text("lat_index,lon_index,hot_sample_count,cnn_hot_rmse_k,reference_hot_rmse_k\n0,0,"+str(n)+",1,2\n")
        files.append(p)
    with pytest.raises(ValueError):
        summarize(files)
