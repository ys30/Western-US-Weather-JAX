"""Direct paired circular-day bootstrap between matched-trained CNN arms.

Uses actual aligned gridded forecast errors, NOT independent confidence intervals
against a common reference. Positive = source_mean has higher RMSE than historical.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a,b,days=14,draws=2000,seed=42):
    for field in ("time","truth_kelvin","reference_error_kelvin","threshold_kelvin"):
        if not np.array_equal(a[field],b[field]):
            raise ValueError("Matched archives disagree on "+field)
    t=np.asarray(a["time"]).astype("datetime64[h]")
    e1=np.asarray(a["cnn_error_kelvin"],dtype=np.float64)
    e2=np.asarray(b["cnn_error_kelvin"],dtype=np.float64)
    truth=np.asarray(a["truth_kelvin"])
    if e1.shape!=e2.shape or e1.shape!=truth.shape or e1.ndim!=3:
        raise ValueError("Mismatched error grids")
    if len(t)%4 or not np.all(np.diff(t)==np.timedelta64(6,"h")):
        raise ValueError("Expected complete contiguous six-hourly days")
    if not np.isfinite(e1).all() or not np.isfinite(e2).all():
        raise ValueError("Non-finite forecast errors")
    n=len(t)//4
    mask=truth>=float(a["threshold_kelvin"])
    rng=np.random.default_rng(seed)
    output={}
    for key,keep in (("annual",np.ones_like(mask,bool)),("observed_hot",mask)):
        count=keep.reshape(n,4,*keep.shape[1:]).sum(axis=(1,2,3)).astype(float)
        sum1=np.where(keep,e1**2,0).reshape(n,4,*keep.shape[1:]).sum(axis=(1,2,3))
        sum2=np.where(keep,e2**2,0).reshape(n,4,*keep.shape[1:]).sum(axis=(1,2,3))
        if count.sum()==0: raise ValueError("Empty evaluated subset "+key)
        delta=lambda ix: float(np.sqrt(sum2[ix].sum()/count[ix].sum())-np.sqrt(sum1[ix].sum()/count[ix].sum()))
        point=delta(np.arange(n))
        samples=[]
        for _ in range(draws):
            ix=((rng.integers(0,n,int(np.ceil(n/days)))[:,None]+np.arange(days))%n).ravel()[:n]
            if count[ix].sum():samples.append(delta(ix))
        if len(samples)<int(.95*draws):raise ValueError("Too many empty bootstrap resamples")
        lo,hi=np.quantile(samples,[.025,.975])
        output[key]={"source_mean_minus_historical_rmse_K":point,
                     "ci95_lower_K":float(lo),"ci95_upper_K":float(hi),
                     "sample_count":int(count.sum()),"valid_draws":len(samples)}
    return {"comparison":"source_mean CNN minus historical CNN","bootstrap_unit":"calendar day",
            "circular_block_days":days,"requested_draws":draws,"seed":seed,
            "observed_hot_is_truth_conditioned":True,"results":output}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--historical",required=True)
    p.add_argument("--source-mean",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--days",type=int,default=14)
    p.add_argument("--draws",type=int,default=2000)
    a=p.parse_args()
    with np.load(a.historical,allow_pickle=False) as h,np.load(a.source_mean,allow_pickle=False) as s:
        result=compare(h,s,a.days,a.draws)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
