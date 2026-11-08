"""Predictor-conditioned 24h anomaly persistence and climatology-blend diagnostic.

Reuses original archived 6-hourly ERA5 input data and EXACT training-only
month x UTC-hour bins as factorial_frozen_eval.py. No CNN retraining.
Source validation is the only legitimate parameter selection population;
target optima are retrospective diagnostics and MUST NOT be test-tuned.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

WEIGHTS=np.round(np.arange(0,1.0001,.05),2)
CASES=[
    ("source_2021","source.nc","2021-01-01","2022-01-01"),
    ("southwest_2022","southwest-2022.nc","2022-01-01","2023-01-01"),
    ("southwest_2023","southwest-2023.nc","2023-01-01","2024-01-01"),
    ("plains_2022","plains-2022.nc","2022-01-01","2023-01-01"),
    ("plains_2023","plains-2023.nc","2023-01-01","2024-01-01"),
]

def read_arrays(path,start,end):
    with xr.open_dataset(path) as ds:
        v=np.asarray(ds["t2m"].values,dtype=np.float32)
        ts=np.asarray(ds.time.values).astype("datetime64[h]")
    if v.ndim!=3 or not np.isfinite(v).all() or np.any(np.diff(ts)!=np.timedelta64(6,"h")):
        raise ValueError(f"Invalid or non-contiguous ERA5 {path}")
    a=int(np.searchsorted(ts,np.datetime64(start)))
    b=int(np.searchsorted(ts,np.datetime64(end)))
    if b-a!=1460 or a<2924:
        raise ValueError(f"Expected full test year and pre-test history: {path}, {a}, {b}")
    train_end=int(np.searchsorted(ts,np.datetime64("2021-01-01"))) if "source" in path.name else a
    months=(ts.astype("datetime64[M]").astype("int64")%12).astype(int)
    hours=(ts.astype("datetime64[h]").astype("int64")%24).astype(int)
    bins=months*24+hours
    train=v[:train_end]
    climate={int(k):train[bins[:train_end]==k].mean(axis=0)
             for k in np.unique(bins[:train_end])}
    if len(climate)!=48 or not set(bins[a:b]).issubset(climate):
        raise ValueError(f"Missing observed 6-hour UTC climatology bins: {path}")
    # climatology evaluated at BOTH valid time and lagged time;
    # lagged climatology differs at month boundaries and must be explicit.
    ix=np.arange(a,b)
    target_climate=np.stack([climate[int(k)] for k in bins[ix]])
    lag_climate=np.stack([climate[int(k)] for k in bins[ix-4]])
    truth=v[ix]; lag=v[ix-4]
    return dict(truth=truth, lag=lag, clim=target_climate, lag_clim=lag_climate)

def rmse(error,mask=None):
    return float(np.sqrt(np.mean(np.square(error if mask is None else error[mask],dtype=np.float64))))

def no_intercept_slope(x,y):
    den=np.sum(np.square(x,dtype=np.float64))
    return float(np.sum(x.astype(np.float64)*y,dtype=np.float64)/den) if den>1e-10 else math.nan

def run(input_dir,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    sets={}
    for name,file,start,end in CASES:
        arr=read_arrays(input_dir/file,start,end)
        x=(arr["lag"]-arr["lag_clim"]).ravel().astype(np.float32)
        y=(arr["truth"]-arr["clim"]).ravel().astype(np.float32)
        future=arr["truth"].ravel()
        previous=arr["lag"].ravel()
        clim=arr["clim"].ravel()
        hot=previous>=305.
        if not hot.any():
            raise ValueError(f"No predictor-hot samples: {name}")
        sets[name]=dict(x=x,y=y,truth=future,prev=previous,clim=clim,hot=hot)
    decile_rows=[];summary=[];curves=[]
    for name,ds in sets.items():
        x,y=ds["x"],ds["y"]
        rho_all=no_intercept_slope(x,y)
        rho_hot=no_intercept_slope(x[ds["hot"]],y[ds["hot"]])
        edges=np.quantile(x,np.linspace(0,1,11))
        if np.unique(edges).size!=11:
            raise ValueError("Ties prevent unique predictor-decile segmentation")
        for k in range(10):
            take=(x>=edges[k])&((x<=edges[k+1]) if k==9 else (x<edges[k+1]))
            decile_rows.append(dict(case=name,bin=k+1,lag_anomaly_lower_K=float(edges[k]),
                                  lag_anomaly_upper_K=float(edges[k+1]),n=int(take.sum()),
                                  rho_24h=no_intercept_slope(x[take],y[take])))
        error0=ds["prev"]-ds["truth"]
        diff=ds["clim"]-ds["prev"]
        analytic_w=np.clip(
            -np.mean(error0.astype(np.float64)*diff.astype(np.float64))/
            np.mean(np.square(diff,dtype=np.float64)),0.,1.)
        analytic_w_hot=np.clip(
            -np.mean(error0[ds["hot"]].astype(np.float64)*diff[ds["hot"]].astype(np.float64))/
            np.mean(np.square(diff[ds["hot"]],dtype=np.float64)),0.,1.)
        rows=[]
        for w in WEIGHTS:
            err=error0+w*diff
            rows.append(dict(case=name,w=float(w),all_RMSE_K=rmse(err),lag_hot_RMSE_K=rmse(err,ds["hot"]),
                             lag_hot_signed_error_K=float(err[ds["hot"]].mean()),
                             lag_hot_B_minus_P_K=float((w*diff[ds["hot"]]).mean())))
        curves.extend(rows)
        best_all=min(rows,key=lambda z:z["all_RMSE_K"])
        best_hot=min(rows,key=lambda z:z["lag_hot_RMSE_K"])
        row25=next(v for v in rows if v["w"]==.25)
        summary.append(dict(case=name,n_all=len(x),n_lag_hot=int(ds["hot"].sum()),
                            lag_hot_percent=100*float(ds["hot"].mean()),
                            rho_all=rho_all,rho_lag_hot=rho_hot,
                            best_w_all_grid=best_all["w"],best_w_lag_hot_grid=best_hot["w"],
                            optimal_w_all_continuous=float(analytic_w),
                            optimal_w_lag_hot_continuous=float(analytic_w_hot),
                            persistence_all_RMSE_K=rows[0]["all_RMSE_K"],
                            blend025_all_RMSE_K=row25["all_RMSE_K"],
                            persistence_lag_hot_RMSE_K=rows[0]["lag_hot_RMSE_K"],
                            blend025_lag_hot_RMSE_K=row25["lag_hot_RMSE_K"],
                            blend025_minus_persistence_lag_hot_RMSE_K=row25["lag_hot_RMSE_K"]-rows[0]["lag_hot_RMSE_K"],
                            blend025_minus_persistence_lag_hot_signed_bias_K=row25["lag_hot_B_minus_P_K"]))
    # four held-out cases pooled, for descriptive synthesis only.
    pooled={k:np.concatenate([v[k] for name,v in sets.items() if name!="source_2021"]) for k in ["x","y","truth","prev","clim","hot"]}
    x,y=pooled["x"],pooled["y"]
    edges=np.quantile(x,np.linspace(0,1,11))
    for k in range(10):
        take=(x>=edges[k])&((x<=edges[k+1]) if k==9 else (x<edges[k+1]))
        decile_rows.append(dict(case="pooled_four_targets",bin=k+1,lag_anomaly_lower_K=float(edges[k]),
                               lag_anomaly_upper_K=float(edges[k+1]),n=int(take.sum()),
                               rho_24h=no_intercept_slope(x[take],y[take])))
    def csv_write(name,records):
        with (output_dir/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader();w.writerows(records)
    csv_write("predictor_anomaly_persistence_deciles.csv",decile_rows)
    csv_write("predictor_conditioned_blend_weight_curves.csv",curves)
    csv_write("predictor_conditioned_summary.csv",summary)
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("CASE | rho_all | rho_lag_hot | best_w_all | best_w_lag_hot | delta_hot_RMSE_B025-P")
    for v in summary:
        print(v["case"],round(v["rho_all"],5),round(v["rho_lag_hot"],5),
              v["best_w_all_grid"],v["best_w_lag_hot_grid"],
              round(v["blend025_minus_persistence_lag_hot_RMSE_K"],5))
    # Source validation ensures original coarse-grid selection is still .25.
    source=next(z for z in summary if z["case"]=="source_2021")
    coarse=[z for z in curves if z["case"]=="source_2021" and
            any(abs(z["w"]-k)<1e-8 for k in (0.,.25,.5,.75,1.))]
    if min(coarse,key=lambda z:z["all_RMSE_K"])["w"]!=.25:
        raise ValueError("Source validation changed relative to original published weight grid")
    return decile_rows,curves,summary

def figures(deciles,curves,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=["source_2021","southwest_2022","southwest_2023","plains_2022","plains_2023"]
    colors=["#243c5a","#237d9a","#68acb1","#bd6135","#e4a75b"]
    fig,ax=plt.subplots(figsize=(9,5.2))
    for name,color in zip(names,colors):
        rows=[r for r in deciles if r["case"]==name]
        ax.plot([r["bin"] for r in rows],[r["rho_24h"] for r in rows],marker="o",ms=4,label=name.replace("_"," "),color=color)
    ax.axhline(.75,color="#555",linestyle="--",lw=1.1,label="1 − selected w = 0.75")
    ax.set(xlabel="Decile of known 24-h-lag temperature anomaly",ylabel="No-intercept anomaly persistence slope (24 h)")
    ax.legend(fontsize=7,ncol=2,loc="best")
    ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(out/"Figure_S15_predictor_anomaly_persistence.png",dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(11,6.6),sharex=True)
    for ax,name in zip(axs.flat,names):
        r=[v for v in curves if v["case"]==name]
        ax.plot([v["w"] for v in r],[v["all_RMSE_K"] for v in r],label="All valid times",lw=1.7)
        ax.plot([v["w"] for v in r],[v["lag_hot_RMSE_K"] for v in r],label="Lagged T ≥305 K",lw=1.7)
        ax.axvline(.25,ls="--",c="#666",lw=1)
        ax.set_title(name.replace("_"," "),fontsize=9)
        ax.set_ylabel("Reference RMSE (K)")
        ax.grid(alpha=.15)
    axs.flat[-1].axis("off")
    axs.flat[0].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel("Climatology blend weight w")
    fig.tight_layout()
    fig.savefig(out/"Figure_S16_predictor_conditioned_weight_curves.png",dpi=220)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    d,c,s=run(a.input,a.output)
    figures(d,c,a.output)

if __name__=="__main__":
    main()
