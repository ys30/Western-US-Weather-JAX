"""Sensitivity of region-wide observed-hot occupancy runs to duration and coverage.

This is an observational occupancy diagnostic, NOT a meteorological heatwave identifier.
"""
import argparse,csv
from pathlib import Path
import numpy as np

def assess(truth,cnn_error,reference_error,times,threshold=305.,min_steps=1,min_fraction=0.):
    y=np.asarray(truth);a=np.asarray(cnn_error);b=np.asarray(reference_error)
    t=np.asarray(times).astype("datetime64[h]")
    if y.ndim!=3 or y.shape!=a.shape or y.shape!=b.shape or len(t)!=len(y):
        raise ValueError("Mismatch in time and spatial shapes")
    if len(t)<2 or not np.all(np.diff(t)==np.timedelta64(6,"h")):
        raise ValueError("Six-hourly continuous timestamps required")
    if not all(np.isfinite(z).all() for z in (y,a,b)):
        raise ValueError("Nonfinite values")
    if min_steps<1 or not 0<=min_fraction<=1:
        raise ValueError("Invalid run filter")
    hot=y>=threshold
    cover=hot.mean(axis=(1,2))
    active=cover>0
    start=np.flatnonzero(active & ~np.r_[False,active[:-1]])
    end=np.flatnonzero(active & ~np.r_[active[1:],False])
    rows=[]
    for i,j in zip(start,end):
        # Occupancy runs are always delimited by ANY target-grid hot cell.
        # Spatial coverage is a filter applied to the ENTIRE run, not a
        # different event definition based on pre-filtered time points.
        if j-i+1<min_steps or cover[i:j+1].max()<min_fraction:continue
        mask=hot[i:j+1]
        d=np.sqrt(np.mean(a[i:j+1][mask]**2))-np.sqrt(np.mean(b[i:j+1][mask]**2))
        rows.append({"start":str(t[i]),"end":str(t[j]),"steps":int(j-i+1),
                     "peak_coverage_pct":float(100*cover[i:j+1].max()),
                     "hot_gridcell_times":int(mask.sum()),"cnn_minus_reference_rmse_k":float(d)})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("npz");p.add_argument("--output",required=True)
    args=p.parse_args()
    with np.load(args.npz) as z:
        data=[z[k] for k in ("truth_kelvin","cnn_error_kelvin","reference_error_kelvin","time")]
        result=[dict(min_steps=steps,min_peak_coverage_pct=100*frac,runs=len(rows),
                     cnn_better_runs=sum(x["cnn_minus_reference_rmse_k"]<0 for x in rows),
                     cnn_worse_runs=sum(x["cnn_minus_reference_rmse_k"]>0 for x in rows))
                for steps in (1,2,4,8) for frac in (0.,.05,.1,.25)
                for rows in [assess(*data,min_steps=steps,min_fraction=frac)]]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
if __name__=="__main__":main()
