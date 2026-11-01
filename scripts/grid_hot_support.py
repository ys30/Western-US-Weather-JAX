"""Descriptive cross-seed grid agreement by minimum hot sample support."""
import argparse,csv,json
from pathlib import Path

def summarize(paths,thresholds=(1,5,10,20,50,100)):
    records={}
    for path in paths:
        p=Path(path)
        seed=int(p.name.split("-")[1])
        if seed in records:raise ValueError("Duplicate seed")
        with p.open(newline="") as f:rows=list(csv.DictReader(f))
        cells={(int(r["lat_index"]),int(r["lon_index"])):r for r in rows}
        if len(cells)!=len(rows):raise ValueError("Duplicate cells")
        records[seed]=cells
    if set(records)!={7,42,123}:raise ValueError("Require seeds 7, 42, 123")
    keys=set(records[42])
    if any(set(c)!=keys for c in records.values()):raise ValueError("Mismatched grids")
    result=[]
    for threshold in thresholds:
        if threshold<1:raise ValueError("Threshold must be positive")
        counts={"minimum_hot_samples":threshold,"eligible_cells":0,
                "all_three_improve":0,"all_three_worsen":0,"mixed_or_tied":0}
        for key in keys:
            rows=[records[s][key] for s in (7,42,123)]
            ns=[int(r["hot_sample_count"]) for r in rows]
            if len(set(ns))!=1:raise ValueError("Inconsistent observed hot counts")
            if ns[0]<threshold:continue
            counts["eligible_cells"]+=1
            diffs=[float(r["cnn_hot_rmse_k"])-float(r["reference_hot_rmse_k"]) for r in rows]
            if all(d<0 for d in diffs):counts["all_three_improve"]+=1
            elif all(d>0 for d in diffs):counts["all_three_worsen"]+=1
            else:counts["mixed_or_tied"]+=1
        result.append(counts)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("files",nargs=3)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    Path(a.output).write_text(json.dumps(summarize(a.files),indent=2)+"\n")
