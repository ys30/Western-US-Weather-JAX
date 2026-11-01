import csv,pytest
from scripts.compare_seed_diagnostics import compare

def test_seed_agreement_with_empty_hot_month(tmp_path):
    paths=[]
    for seed in (42,7,123):
        p=tmp_path/f"seed-{seed}-fields.monthly.csv"
        with p.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["month","hot_gridcell_times","cnn_rmse_k","reference_rmse_k","cnn_hot_rmse_k","reference_hot_rmse_k"])
            w.writerow(["2022-01",0,2,3,"nan","nan"])
            w.writerow(["2022-07",4,2,3,2,3])
        paths.append(p)
    out=compare(paths)
    assert out[0]["hot_improvement_all_seeds"] is None
    assert out[1]["hot_improvement_all_seeds"] is True
    assert out[1]["annual_improvement_all_seeds"] is True

def test_mismatched_counts_are_rejected(tmp_path):
    paths=[]
    for seed,n in zip((42,7,123),(3,4,3)):
        p=tmp_path/f"seed-{seed}-fields.monthly.csv"
        p.write_text("month,hot_gridcell_times,cnn_rmse_k,reference_rmse_k,cnn_hot_rmse_k,reference_hot_rmse_k\n2022-07,"+str(n)+",2,3,2,3\n")
        paths.append(p)
    with pytest.raises(ValueError):compare(paths)
