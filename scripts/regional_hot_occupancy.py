"""Descriptive regional 305 K occupancy runs from paired, archived ERA5 forecast errors."""
import argparse,csv
from pathlib import Path
import numpy as np

def occupancy_runs(truth,cnn_error,reference_error,times,threshold=305.):
    tr=np.asarray(truth);ce=np.asarray(cnn_error);re=np.asarray(reference_error)
    ts=np.asarray(times).astype("datetime64[h]")
    if tr.ndim!=3 or tr.shape!=ce.shape or tr.shape!=re.shape or len(ts)!=len(tr):
        raise ValueError("Expected aligned time/latitude/longitude arrays")
    if len(ts)<2 or not np.all(np.diff(ts)==np.timedelta64(6,"h")):
        raise ValueError("Expected continuous six-hourly times")
    if not all(np.isfinite(a).all() for a in (tr,ce,re)):
        raise ValueError("Non-finite inputs")
    mask=tr>=threshold
    occupied=mask.any(axis=(1,2))
    starts=np.flatnonzero(occupied & ~np.r_[False,occupied[:-1]])
    ends=np.flatnonzero(occupied & ~np.r_[occupied[1:],False])
    result=[]
    for i,(start,end) in enumerate(zip(starts,ends),1):
        hot=mask[start:end+1]
        cnn=np.sqrt(np.mean(ce[start:end+1][hot]**2))
        ref=np.sqrt(np.mean(re[start:end+1][hot]**2))
        result.append({"run_id":i,"start":str(ts[start]),"end":str(ts[end]),
          "steps":int(end-start+1),"hot_gridcell_times":int(hot.sum()),
          "cnn_rmse_k":float(cnn),"reference_rmse_k":float(ref),
          "cnn_minus_reference_k":float(cnn-ref)})
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("npz");p.add_argument("--output",required=True)
    a=p.parse_args()
    with np.load(a.npz) as data:
        rows=occupancy_runs(data["truth_kelvin"],data["cnn_error_kelvin"],
                            data["reference_error_kelvin"],data["time"])
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="") as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
