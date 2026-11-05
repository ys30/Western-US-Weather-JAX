"""Recover the exact validation-selected persistence-climatology Blend grid.

Reads the archived *source* ERA5 NetCDF without network or CNN training.
Uses h5py to read the original HDF5-backed NetCDF and mirrors the float32
operations and timestamp binning in scripts/climatology_baselines.py.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

WEIGHTS=np.asarray([0.,.25,.5,.75,1.],dtype=float)
EXPECTED_SOURCE_SHA256="ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6"

def evaluate(path):
    raw=Path(path).read_bytes()
    source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha!=EXPECTED_SOURCE_SHA256:
        raise ValueError("Archived source data SHA mismatch; refuse to report a different experiment")
    with h5py.File(path,"r") as f:
        if not {"time","t2m","lat","lon"}.issubset(f):
            raise ValueError("Missing required archived source data")
        arr=np.asarray(f["t2m"],dtype=np.float32)
        time_hours=np.asarray(f["time"],dtype=np.int64)
        units=f["time"].attrs["units"]
        if isinstance(units,bytes):units=units.decode()
        if units!="hours since 1900-01-01":
            raise ValueError(f"Unexpected time units: {units}")
    ts=np.datetime64("1900-01-01T00","h")+time_hours.astype("timedelta64[h]")
    if arr.shape!=(len(ts),49,49) or not np.isfinite(arr).all():
        raise ValueError("Wrong archived source geometry or non-finite values")
    if not np.all(np.diff(ts)==np.timedelta64(6,"h")):
        raise ValueError("Expected complete six-hour sample times")
    train_end=int(np.searchsorted(ts,np.datetime64("2021-01-01")))
    validation_end=int(np.searchsorted(ts,np.datetime64("2022-01-01")))
    if (train_end,validation_end,len(arr))!=(2924,4384,5844):
        raise ValueError("Wrong chronological split")
    months=(ts.astype("datetime64[M]").astype("int64")%12).astype(int)
    hours=(ts.astype("datetime64[h]").astype("int64")%24).astype(int)
    bins=months*24+hours
    train=arr[:train_end]
    climatology={int(k):train[bins[:train_end]==k].mean(axis=0) for k in np.unique(bins[:train_end])}
    global_mean=train.mean(axis=0)
    idx=np.arange(train_end,validation_end)
    truth=arr[idx]
    persistence=arr[idx-4]
    climate=np.stack([climatology.get(int(bins[i]),global_mean) for i in idx])
    rmses=[float(np.sqrt(np.mean(((1-w)*persistence+w*climate-truth)**2))) for w in WEIGHTS]
    winner=int(np.argmin(rmses))
    if len(climatology)!=48 or not set(bins[idx]).issubset(climatology):
        raise ValueError("Occupied bins or validation coverage mismatch")
    if WEIGHTS[winner]!=.25:
        raise ValueError("Recovered weight differs from original source reports")
    return [{"source_sha256":source_sha,
             "source_training_samples":train_end,
             "source_validation_samples":validation_end-train_end,
             "occupied_training_month_utc_bins":len(climatology),
             "validation_uses_missing_bin_fallback":False,
             "candidate_climatology_weight":float(w),
             "source_validation_blend_rmse_K":r,
             "source_validation_selected":i==winner} for i,(w,r) in enumerate(zip(WEIGHTS,rmses))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected",type=Path)
    a=p.parse_args()
    rows=evaluate(a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if a.expected:
        with a.expected.open() as f:
            expect=list(csv.DictReader(f))
        if len(expect)!=5: raise ValueError("Expected five weight choices")
        for old,new in zip(expect,rows):
            for k,v in new.items():
                if isinstance(v,float):
                    if abs(float(old[k])-v)>1e-7: raise ValueError((k,old[k],v))
                elif isinstance(v,bool):
                    if old[k].lower()!=str(v).lower(): raise ValueError((k,old[k],v))
                else:
                    if str(old[k])!=str(v):raise ValueError((k,old[k],v))
    print("Source validation Blend candidates (K):",[(r["candidate_climatology_weight"],round(r["source_validation_blend_rmse_K"],6)) for r in rows])
    print("Selected climatology weight 0.25; occupied bins 48; fallback unused")

if __name__=="__main__":
    main()
