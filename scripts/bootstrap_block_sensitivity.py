"""Circular calendar-day block sensitivity from paired forecast error arrays."""
import argparse,csv
from pathlib import Path
import numpy as np

def assess(truth,cnn_error,baseline_error,times,seed,thresholds=(303,305,307),blocks=(3,7,14,28),draws=2000):
    y=np.asarray(truth);a=np.asarray(cnn_error);b=np.asarray(baseline_error)
    t=np.asarray(times).astype("datetime64[D]")
    if y.ndim!=3 or y.shape!=a.shape or y.shape!=b.shape or len(t)!=len(y):
        raise ValueError("Mismatched arrays")
    if not all(np.isfinite(v).all() for v in (y,a,b)):raise ValueError("Nonfinite data")
    day=(t-t[0]).astype(int);n=int(day.max())+1
    if set(np.unique(day))!=set(range(n)) or draws<=0:raise ValueError("Missing days or invalid draws")
    result=[]
    for threshold in thresholds:
        hot=y>=threshold
        count=np.bincount(day,weights=hot.sum((1,2)),minlength=n)
        aa=np.bincount(day,weights=(a*a*hot).sum((1,2)),minlength=n)
        bb=np.bincount(day,weights=(b*b*hot).sum((1,2)),minlength=n)
        if not count.sum():raise ValueError("Empty threshold")
        point=float(np.sqrt(aa.sum()/count.sum())-np.sqrt(bb.sum()/count.sum()))
        for block in blocks:
            if block<1:raise ValueError("Invalid block")
            rng=np.random.default_rng(20261101+seed*100+int(threshold)+block*10000)
            starts=rng.integers(0,n,size=(draws,(n+block-1)//block))
            ix=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
            den=count[ix].sum(1);valid=den>0
            if not np.any(valid):raise ValueError("No nonempty bootstrap draw")
            d=np.sqrt(aa[ix].sum(1)[valid]/den[valid])-np.sqrt(bb[ix].sum(1)[valid]/den[valid])
            lo,hi=np.quantile(d,[.025,.975])
            result.append(dict(seed=seed,threshold_kelvin=threshold,block_days=block,
               draws=int(valid.sum()),hot_gridcell_times=int(count.sum()),point_delta_k=point,
               ci_low_k=float(lo),ci_high_k=float(hi)))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("npz");p.add_argument("--seed",type=int,required=True);p.add_argument("--output",required=True)
    args=p.parse_args()
    with np.load(args.npz) as z:
        rows=assess(z["truth_kelvin"],z["cnn_error_kelvin"],z["reference_error_kelvin"],z["time"],args.seed)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
