import csv,json,pytest
from scripts.validate_monthly_reconstruction import validate

def test_monthly_reconstruction(tmp_path):
    p=tmp_path/"m.csv";j=tmp_path/"e.json"
    with p.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["month","time_steps","hot_gridcell_times","cnn_rmse_k","reference_rmse_k","cnn_hot_rmse_k","reference_hot_rmse_k"])
        for m in range(1,13):
            w.writerow([f"2022-{m:02d}",4,2,3.,4.,2.,3.])
    j.write_text(json.dumps(dict(test_rmse_kelvin=3,source_weight_blend_rmse_kelvin=4,fixed_305k_cnn_rmse_kelvin=2,fixed_305k_blend_rmse_kelvin=3,fixed_hot_samples=24)))
    assert validate(p,j)["test_rmse_kelvin"]==pytest.approx(3)
    j.write_text(json.dumps(dict(test_rmse_kelvin=5,source_weight_blend_rmse_kelvin=4,fixed_305k_cnn_rmse_kelvin=2,fixed_305k_blend_rmse_kelvin=3,fixed_hot_samples=24)))
    with pytest.raises(ValueError):validate(p,j)
