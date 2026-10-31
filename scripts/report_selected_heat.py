"""Report 2022 holdout results after model selection on 2021 validation only."""
import argparse
import json
from pathlib import Path

def report(selection_path, metric_paths):
    selection=json.loads(Path(selection_path).read_text())
    weights={}
    for path in metric_paths:
        m=json.loads(Path(path).read_text())
        w=float(m["hot_weight"])
        if w in weights: raise ValueError("Duplicate hot_weight")
        weights[w]=m
    chosen=float(selection["selected_hot_weight"])
    if chosen not in weights: raise ValueError("Selected weight has no metrics")
    m=weights[chosen]
    fixed=m["fixed_heat_evaluation"]
    if fixed["gridcell_count"]==0: raise ValueError("Empty fixed heat subset")
    return {"selection_source":"2021 validation only","holdout_year":2022,
      "selected_hot_weight":chosen,"fixed_threshold_kelvin":fixed["threshold_kelvin"],
      "selected_test_overall_rmse_kelvin":m["test_rmse_kelvin"],
      "test_persistence_overall_rmse_kelvin":m["persistence_test_rmse_kelvin"],
      "selected_test_heat_rmse_kelvin":fixed["cnn_rmse_k"],
      "test_persistence_heat_rmse_kelvin":fixed["persistence_rmse_k"],
      "test_heat_sample_count":fixed["gridcell_count"],
      "overall_skill_pct":100*(1-m["test_rmse_kelvin"]/m["persistence_test_rmse_kelvin"]),
      "heat_skill_pct":100*(1-fixed["cnn_rmse_k"]/fixed["persistence_rmse_k"]),
      "note":"No test metrics enter weight selection. Report fixed-threshold and overall skill regardless of sign."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("selection");p.add_argument("metrics",nargs="+");p.add_argument("--output",required=True)
    a=p.parse_args();out=report(a.selection,a.metrics)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=="__main__":main()
