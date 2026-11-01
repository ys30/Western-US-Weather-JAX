"""Summarize verified monthly diagnostics across model seeds without averaging RMSEs as pooled skill."""
import argparse,csv,json,math
from pathlib import Path

def compare(paths):
    files={}
    for path in paths:
        p=Path(path);seed=int(p.name.split("-")[1])
        if seed in files:raise ValueError("Duplicate seed")
        with p.open(newline="") as f: rows=list(csv.DictReader(f))
        files[seed]={r["month"]:r for r in rows}
    if sorted(files)!=[7,42,123]:raise ValueError("Require exactly seeds 7,42,123")
    months=set(files[42])
    if any(set(data)!=months for data in files.values()):raise ValueError("Mismatched month coverage")
    result=[]
    for month in sorted(months):
        base_counts=[int(files[s][month]["hot_gridcell_times"]) for s in files]
        if len(set(base_counts))!=1:raise ValueError("Observed-hot counts must agree across seeds")
        annual=[
            100*(1-float(files[s][month]["cnn_rmse_k"])/float(files[s][month]["reference_rmse_k"])) for s in sorted(files)]
        hot=[100*(1-float(files[s][month]["cnn_hot_rmse_k"])/float(files[s][month]["reference_hot_rmse_k"])) for s in sorted(files)] if base_counts[0] else []
        if not all(math.isfinite(x) for x in annual+hot):raise ValueError("Non-finite derived change")
        result.append({"month":month,"hot_gridcell_times":base_counts[0],
          "annual_improvement_all_seeds":all(x>0 for x in annual),
          "hot_improvement_all_seeds":all(x>0 for x in hot) if hot else None,
          "annual_relative_changes_pct":annual,"hot_relative_changes_pct":hot if hot else None})
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("files",nargs=3);ap.add_argument("--output",required=True)
    a=ap.parse_args()
    Path(a.output).write_text(json.dumps(compare(a.files),indent=2)+"\n")
