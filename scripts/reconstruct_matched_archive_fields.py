"""Reconstruct matched CNN predictions using archived diagnostic error grids."""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

FIELDS=("time","truth_kelvin","cnn_error_kelvin","reference_error_kelvin","latitudes","longitudes")
def read(path):
    with np.load(path,allow_pickle=False) as z:
        if any(k not in z for k in FIELDS): raise ValueError("Missing archive keys")
        d={k:z[k].copy() for k in FIELDS}
    truth=d["truth_kelvin"]
    if truth.ndim!=3 or any(d[k].shape!=truth.shape for k in ("cnn_error_kelvin","reference_error_kelvin")):
        raise ValueError("Invalid grid dimensions")
    if len(d["time"])!=len(truth) or d["latitudes"].size!=truth.shape[1] or d["longitudes"].size!=truth.shape[2]:
        raise ValueError("Invalid coordinate dimensions")
    if not all(np.isfinite(d[k]).all() for k in FIELDS if k!="time"):
        raise ValueError("Nonfinite archive data")
    ts=d["time"].astype("datetime64[ns]")
    if np.isnat(ts).any() or np.any(np.diff(ts)!=np.timedelta64(6,"h")):
        raise ValueError("Invalid target timestamps")
    d["time"]=ts
    return d

def reconstruct(historical,source_mean):
    a,b=read(historical),read(source_mean)
    for k in ("time","truth_kelvin","reference_error_kelvin","latitudes","longitudes"):
        if not np.array_equal(a[k],b[k]): raise ValueError("Mismatched paired archives: "+k)
    c={"time":a["time"],"latitude":a["latitudes"],"longitude":a["longitudes"]}
    dims=("time","latitude","longitude")
    truth=a["truth_kelvin"].astype("float64")
    ds=xr.Dataset({name:(dims,truth+error) for name,error in (
        ("historical_K",a["cnn_error_kelvin"]),
        ("source_mean_K",b["cnn_error_kelvin"]),
        ("baseline_K",a["reference_error_kelvin"]))},coords=c)
    for name in ds.data_vars: ds[name].attrs["units"]="K"
    ds.attrs["time_definition"]="verifying target valid time; no shift applied"
    return ds

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--historical",required=True)
    p.add_argument("--source-mean",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=reconstruct(a.historical,a.source_mean)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    result.to_netcdf(out)
    print("Reconstructed matched-channel prediction cube:",result.sizes)
