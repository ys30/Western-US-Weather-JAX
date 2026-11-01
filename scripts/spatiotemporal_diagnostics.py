"""Export auditable spatial and monthly errors from actual paired model predictions."""
import csv
from pathlib import Path
import numpy as np

def export_diagnostics(output,pred,baseline,truth,times,threshold=305.,latitudes=None,longitudes=None):
    """Save compact error fields and pre-aggregated metrics; never invent fields from scalar RMSE."""
    p=Path(output)
    pr=np.asarray(pred);ba=np.asarray(baseline);tr=np.asarray(truth)
    ts=np.asarray(times).astype("datetime64[h]")
    if pr.shape!=tr.shape or ba.shape!=tr.shape or tr.ndim!=3 or len(ts)!=len(tr):
        raise ValueError("Expected equally shaped [time,lat,lon] arrays and matching times")
    if not all(np.isfinite(a).all() for a in (pr,ba,tr)):
        raise ValueError("Nonfinite prediction, baseline or truth")
    if len(ts)<2 or not np.all(np.diff(ts)==np.timedelta64(6,"h")):
        raise ValueError("Expected contiguous six-hourly timestamps")
    if (latitudes is None) != (longitudes is None):
        raise ValueError('Latitude and longitude must be provided together')
    if latitudes is not None:
        latitudes=np.asarray(latitudes,dtype=float)
        longitudes=np.asarray(longitudes,dtype=float)
        if (latitudes.shape!=(tr.shape[1],) or longitudes.shape!=(tr.shape[2],)
                or not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all()):
            raise ValueError('Coordinate vectors must be finite and match grid shape')
    p.parent.mkdir(parents=True,exist_ok=True)
    err=pr-tr; referr=ba-tr
    coords=({"latitudes":latitudes,"longitudes":longitudes} if latitudes is not None else {})
    np.savez_compressed(p,time=ts,truth_kelvin=tr.astype("float32"),
        cnn_error_kelvin=err.astype("float32"),reference_error_kelvin=referr.astype("float32"),
        hot_mask=tr>=threshold,threshold_kelvin=np.float64(threshold),**coords)
    def rmse(e,mask):
        v=e[mask]
        return float(np.sqrt(np.mean(v*v))) if v.size else float("nan")
    months=ts.astype("datetime64[M]")
    with p.with_suffix(".monthly.csv").open("w",newline="") as fo:
        w=csv.writer(fo)
        w.writerow(["month","time_steps","hot_gridcell_times","cnn_rmse_k","reference_rmse_k",
                    "cnn_hot_rmse_k","reference_hot_rmse_k"])
        for m in np.unique(months):
            selected=months==m
            a=err[selected];b=referr[selected];mask=tr[selected]>=threshold
            w.writerow([str(m),int(selected.sum()),int(mask.sum()),rmse(a,np.ones_like(mask,bool)),
                        rmse(b,np.ones_like(mask,bool)),rmse(a,mask),rmse(b,mask)])
    with p.with_suffix(".grid.csv").open("w",newline="") as fo:
        w=csv.writer(fo)
        w.writerow(["lat_index","lon_index","latitude","longitude","cnn_rmse_k","reference_rmse_k","hot_sample_count",
                    "cnn_hot_rmse_k","reference_hot_rmse_k"])
        for i in range(tr.shape[1]):
            for j in range(tr.shape[2]):
                mask=tr[:,i,j]>=threshold
                w.writerow([i,j,float(latitudes[i]) if latitudes is not None else "",
                    float(longitudes[j]) if longitudes is not None else "",rmse(err[:,i,j],np.ones(len(ts),bool)),
                    rmse(referr[:,i,j],np.ones(len(ts),bool)),int(mask.sum()),
                    rmse(err[:,i,j],mask),rmse(referr[:,i,j],mask)])
