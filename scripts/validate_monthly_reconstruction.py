"""Check that monthly diagnostics reconstruct the archived held-out RMSE."""
import argparse,csv,json,math
from pathlib import Path

def validate(monthly_csv,evaluation_json,atol=1e-5):
    with Path(monthly_csv).open(newline="") as f: rows=list(csv.DictReader(f))
    evaluation=json.loads(Path(evaluation_json).read_text())
    if len(rows)!=12: raise ValueError("Expected 12 calendar months")
    months=[r["month"] for r in rows]
    if months!=sorted(months) or len(months)!=len(set(months)):
        raise ValueError("Duplicate or unordered months")
    def metric(score,count):
        weights=[int(r[count]) for r in rows]
        if sum(weights)==0: raise ValueError("No samples")
        return math.sqrt(sum(float(r[score])**2*w for r,w in zip(rows,weights) if w)/sum(weights))
    observed={
        "test_rmse_kelvin":metric("cnn_rmse_k","time_steps"),
        "source_weight_blend_rmse_kelvin":metric("reference_rmse_k","time_steps"),
        "fixed_305k_cnn_rmse_kelvin":metric("cnn_hot_rmse_k","hot_gridcell_times"),
        "fixed_305k_blend_rmse_kelvin":metric("reference_hot_rmse_k","hot_gridcell_times")}
    if sum(int(r["hot_gridcell_times"]) for r in rows)!=evaluation["fixed_hot_samples"]:
        raise ValueError("Hot sample counts differ")
    for key,val in observed.items():
        if not math.isclose(val,float(evaluation[key]),rel_tol=0,abs_tol=atol):
            raise ValueError(f"RMSE mismatch: {key}: {val} vs {evaluation[key]}")
    return observed

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("monthly_csv");ap.add_argument("evaluation_json")
    a=ap.parse_args()
    print(json.dumps(validate(a.monthly_csv,a.evaluation_json),indent=2))
