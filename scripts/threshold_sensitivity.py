"""Observed-temperature threshold sensitivity from paired archived model error arrays."""
import argparse,csv
from pathlib import Path
import numpy as np

def evaluate_thresholds(truth,cnn_error,reference_error,thresholds=(303.,305.,307.)):
    y=np.asarray(truth,dtype=float);a=np.asarray(cnn_error,dtype=float);b=np.asarray(reference_error,dtype=float)
    if y.ndim!=3 or y.shape!=a.shape or y.shape!=b.shape:
        raise ValueError("Expected aligned [time,lat,lon] arrays")
    if not all(np.isfinite(z).all() for z in (y,a,b)):
        raise ValueError("Nonfinite input")
    results=[]
    for threshold in thresholds:
        mask=y>=threshold
        n=int(mask.sum())
        if n==0:
            results.append(dict(threshold_kelvin=float(threshold),hot_gridcell_times=0,
                cnn_rmse_kelvin=None,reference_rmse_kelvin=None,relative_improvement_percent=None))
            continue
        cnn=float(np.sqrt(np.mean(a[mask]**2)))
        ref=float(np.sqrt(np.mean(b[mask]**2)))
        results.append(dict(threshold_kelvin=float(threshold),hot_gridcell_times=n,
            cnn_rmse_kelvin=cnn,reference_rmse_kelvin=ref,
            relative_improvement_percent=(100*(1-cnn/ref) if ref>0 else None)))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("npz");p.add_argument("--output",required=True)
    a=p.parse_args()
    with np.load(a.npz) as z:
        rows=evaluate_thresholds(z["truth_kelvin"],z["cnn_error_kelvin"],z["reference_error_kelvin"])
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True)
    with o.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
