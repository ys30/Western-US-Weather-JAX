"""Leakage-controlled month/UTC-hour climatology and persistence-blend benchmarks."""
import argparse,json
from pathlib import Path
import numpy as np
from weatherjax.data import load_era5

def benchmark(arr, timestamps, train_until="2021-01-01", validate_until="2022-01-01", lead_steps=4):
    arr=np.asarray(arr,dtype=np.float32)
    ts=np.asarray(timestamps,dtype="datetime64[h]")
    if len(arr)!=len(ts) or arr.ndim!=3: raise ValueError("Expected [time,lat,lon] temperature and matching timestamps")
    if not np.all(np.diff(ts)>np.timedelta64(0,"h")): raise ValueError("Timestamps must increase")
    a=int(np.searchsorted(ts,np.datetime64(train_until)));b=int(np.searchsorted(ts,np.datetime64(validate_until)))
    if not (lead_steps<a<b<len(arr)):raise ValueError("Invalid chronological split")
    months=(ts.astype("datetime64[M]").astype("int64")%12).astype(int)
    hours=(ts.astype("datetime64[h]").astype("int64")%24).astype(int)
    bins=months*24+hours
    training=arr[:a]
    global_mean=training.mean(axis=0)
    climatology={}
    for key in np.unique(bins[:a]):
        climatology[int(key)]=training[bins[:a]==key].mean(axis=0)
    def evaluate(lo,hi):
        indices=np.arange(lo,hi)
        truth=arr[indices]
        persistence=arr[indices-lead_steps]
        climate=np.stack([climatology.get(int(bins[i]),global_mean) for i in indices])
        return truth,persistence,climate
    yt,pv,cv=evaluate(a,b)
    grid=np.array([0,0.25,0.5,0.75,1],dtype=float)
    values=[float(np.sqrt(np.mean(((1-w)*pv+w*cv-yt)**2))) for w in grid]
    best=int(np.argmin(values));weight=float(grid[best])
    ye,pe,ce=evaluate(b,len(arr))
    def rmse(p):return float(np.sqrt(np.mean((p-ye)**2)))
    mixture=(1-weight)*pe+weight*ce
    return {"train_end":train_until,"validation_end":validate_until,"test_start":str(ts[b]),
       "climatology":"training-only month-of-year x UTC-hour, per-gridcell mean",
       "validation_grid_weights":grid.tolist(),"validation_rmse_k":values,
       "selected_climatology_weight":weight,
       "test_rmse_k":{"persistence":rmse(pe),"climatology":rmse(ce),"validation_selected_blend":rmse(mixture)},
       "test_samples":int(len(ye)),
       "note":"Climatology is computed only using training dates; blending weight selected on validation period; test year untouched."}

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--train-until",default="2021-01-01");p.add_argument("--validate-until",default="2022-01-01")
    p.add_argument("--lead-steps",type=int,default=4)
    a=p.parse_args();da=load_era5(a.data)
    report=benchmark(da.values,da.time.values,a.train_until,a.validate_until,a.lead_steps)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
