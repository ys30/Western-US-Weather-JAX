"""Rank prespecified heat weights using validation metrics only; test scores are excluded."""
import argparse,json
from pathlib import Path

def select(paths):
    entries=[]
    for path in paths:
        m=json.loads(Path(path).read_text())
        hot=m.get("best_validation_fixed_heat_rmse_kelvin")
        if hot is None: raise ValueError("Missing nonempty validation heat subset")
        entries.append({"model_path":str(path),"hot_weight":m["hot_weight"],
                        "validation_hot_rmse_k":hot,
                        "validation_overall_rmse_k":m["best_validation_rmse_kelvin"]})
    entries.sort(key=lambda e:(e["validation_hot_rmse_k"],e["validation_overall_rmse_k"]))
    return {"criterion":"lowest 2021 validation RMSE on target T>=305 K; independent 2022 test not used",
            "selected_hot_weight":entries[0]["hot_weight"],"ranked_validation":entries}
def main():
    p=argparse.ArgumentParser();p.add_argument("metrics",nargs="+");p.add_argument("--output",required=True)
    a=p.parse_args();res=select(a.metrics)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=="__main__":main()
