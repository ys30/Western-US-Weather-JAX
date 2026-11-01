"""Reproducible circular time-block bootstrap for truth-conditional temperature RMSE."""
import argparse,csv
from pathlib import Path
import numpy as np

def bootstrap_thresholds(truth,cnn_error,reference_error,times,seed,thresholds=(303.,305.,307.),block_days=14,draws=2000):
    y=np.asarray(truth);a=np.asarray(cnn_error);b=np.asarray(reference_error)
    t=np.asarray(times).astype("datetime64[D]")
    if y.ndim!=3 or y.shape!=a.shape or y.shape!=b.shape or len(t)!=len(y):
        raise ValueError("Aligned [time,lat,lon] inputs required")
    if not all(np.isfinite(x).all() for x in (y,a,b)):raise ValueError("Nonfinite input")
    if block_days<1 or draws<1:raise ValueError("Invalid resampling parameters")
    day=(t-t[0]).astype("timedelta64[D]").astype(int)
    n_days=int(day.max())+1
    if set(np.unique(day))!=set(range(n_days)):raise ValueError("Missing days")
    rows=[]
    for threshold in thresholds:
        hot=y>=threshold
        count=np.bincount(day,weights=hot.sum(axis=(1,2)),minlength=n_days)
        aa=np.bincount(day,weights=(a*a*hot).sum(axis=(1,2)),minlength=n_days)
        bb=np.bincount(day,weights=(b*b*hot).sum(axis=(1,2)),minlength=n_days)
        if not count.sum():raise ValueError("Empty hot subset")
        point=np.sqrt(aa.sum()/count.sum())-np.sqrt(bb.sum()/count.sum())
        rng=np.random.default_rng(20261101+seed*100+int(threshold))
        sample=[]
        for _ in range(draws):
            start=rng.integers(0,n_days,size=(n_days+block_days-1)//block_days)
            ix=((start[:,None]+np.arange(block_days))%n_days).ravel()[:n_days]
            den=count[ix].sum()
            if den:
                sample.append(np.sqrt(aa[ix].sum()/den)-np.sqrt(bb[ix].sum()/den))
        if not sample:raise ValueError("Bootstrap draws all empty")
        lo,hi=np.quantile(sample,[.025,.975])
        rows.append(dict(seed=seed,threshold_kelvin=threshold,hot_gridcell_times=int(count.sum()),
            cnn_minus_baseline_rmse_k=float(point),ci_lower_k=float(lo),ci_upper_k=float(hi),
            bootstrap_draws=len(sample),block_days=block_days))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("npz");p.add_argument("--seed",required=True,type=int);p.add_argument("--output",required=True)
    args=p.parse_args()
    with np.load(args.npz) as z:
        rows=bootstrap_thresholds(z["truth_kelvin"],z["cnn_error_kelvin"],z["reference_error_kelvin"],z["time"],args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
