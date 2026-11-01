"""Validate and summarize real monthly diagnostics without inventing missing hot scores."""
import argparse,csv,json,math
from pathlib import Path

REQUIRED={"month","time_steps","hot_gridcell_times","cnn_rmse_k","reference_rmse_k","cnn_hot_rmse_k","reference_hot_rmse_k"}
def summarize(path):
    with Path(path).open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError("Missing monthly diagnostic columns or rows")
    out=[]
    prev=""
    for row in rows:
        month=row["month"]
        if not month or month<=prev:raise ValueError("Months must be strictly increasing")
        prev=month
        n=int(row["time_steps"]);hot=int(row["hot_gridcell_times"])
        if n<1 or hot<0:raise ValueError("Invalid sample counts")
        def value(key):
            return float(row[key])
        a,b=value("cnn_rmse_k"),value("reference_rmse_k")
        if not (math.isfinite(a) and math.isfinite(b) and a>=0 and b>0):
            raise ValueError("Invalid overall RMSE")
        h,ref=value("cnn_hot_rmse_k"),value("reference_hot_rmse_k")
        if hot==0:
            if not (math.isnan(h) and math.isnan(ref)):raise ValueError("Empty hot subset must be NaN")
            hot_change=None
        else:
            if not (math.isfinite(h) and math.isfinite(ref) and h>=0 and ref>0):
                raise ValueError("Invalid conditional RMSE")
            hot_change=100*(1-h/ref)
        out.append({"month":month,"time_steps":n,"hot_gridcell_times":hot,
            "all_rmse_relative_change_pct":100*(1-a/b),
            "hot_rmse_relative_change_pct":hot_change})
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("monthly_csv");ap.add_argument("--output",required=True)
    args=ap.parse_args()
    result=summarize(args.monthly_csv)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
