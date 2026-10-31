"""Generate compact side-by-side evaluation reports from two model metrics files."""
import argparse
import json
from pathlib import Path

def build_report(single, multi):
    a=json.loads(Path(single).read_text())
    b=json.loads(Path(multi).read_text())
    result={"models":[],"monthly":[],"hot_tail":[],"fixed_heat":[]}
    for label,m in [("single_frame",a),("four_frame",b)]:
        result["models"].append({"model":label,"rmse_k":m["test_rmse_kelvin"],"persistence_rmse_k":m["persistence_test_rmse_kelvin"],
            "rmse_skill_pct":100*(1-m["test_rmse_kelvin"]/m["persistence_test_rmse_kelvin"]),"best_epoch":m["best_epoch"]})
        group=m.get("group_evaluation",{})
        for row in group.get("monthly",[]):
            result["monthly"].append({"model":label,**row,"skill_pct":100*(1-row["cnn_rmse_k"]/row["persistence_rmse_k"])})
        fixed=m.get("fixed_heat_evaluation")
        if fixed and fixed.get("gridcell_count",0)>0:
            result["fixed_heat"].append({"model":label,**fixed,
                "skill_pct":100*(1-fixed["cnn_rmse_k"]/fixed["persistence_rmse_k"])})
        if group:
            result["hot_tail"].append({"model":label,"threshold_k":group["hot_test_p95_kelvin"],
                "cnn_rmse_k":group["hot_cnn_rmse_k"],"persistence_rmse_k":group["hot_persistence_rmse_k"],
                "skill_pct":100*(1-group["hot_cnn_rmse_k"]/group["hot_persistence_rmse_k"])})
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("single")
    ap.add_argument("multi")
    ap.add_argument("--output",default="outputs/real-era5/comparison.json")
    args=ap.parse_args()
    result=build_report(args.single,args.multi)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
