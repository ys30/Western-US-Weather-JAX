"""Recompute airport elevation and time-block sensitivity for two validated cases.

Reads original successful zipped Actions artifacts, never uses synthetic
predictions, and checks zero-lapse RMSE/14-day CIs against original scorer.
"""
import argparse, io, json, hashlib
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

CASES={
    "southwest-2023": ("additional-asos-southwest-2023.zip",("ABQ","ROW")),
    "southern-plains-2022": ("additional-asos-southern-plains-2022.zip",("ABI","OKC"))
}
SEEDS=(42,7,123)
LAPSES=(0.,3.,6.5,9.8)
BLOCKS=(7,14,28)

def member(z,name):
    keys=[key for key in z.namelist() if key==name or key.endswith("/"+name)]
    if len(keys)!=1: raise ValueError(f"Not exactly one {name}: {keys}")
    return keys[0]

def calc(frame,dz,rate,block,draws):
    frame=frame.sort_values("timestamp_utc")
    dt=pd.to_datetime(frame.timestamp_utc,utc=True).dt.floor("D")
    first=dt.iloc[0]
    nd=len(pd.date_range(first,dt.iloc[-1],freq="D"))
    days=((dt-first)/pd.Timedelta(days=1)).to_numpy(dtype=int)
    observed=frame.observed_K.to_numpy(float)
    shift=-rate*dz/1000.
    h=frame.historical_K.to_numpy(float)+shift
    s=frame.source_mean_K.to_numpy(float)+shift
    base=frame.baseline_K.to_numpy(float)+shift
    rmse=lambda a:float(np.sqrt(np.mean((a-observed)**2)))
    hh,ss,bb=rmse(h),rmse(s),rmse(base)
    counts=np.bincount(days,minlength=nd)
    he=np.bincount(days,weights=(h-observed)**2,minlength=nd)
    se=np.bincount(days,weights=(s-observed)**2,minlength=nd)
    rng=np.random.default_rng(20261103)
    chunks=(nd+block-1)//block
    offset=np.arange(block)
    samples=[]
    for _ in range(draws):
        starts=rng.integers(0,nd,size=chunks)
        indices=((starts[:,None]+offset[None,:])%nd).ravel()[:nd]
        n=counts[indices].sum()
        if n==0:raise ValueError("Empty circular bootstrap draw")
        samples.append(float(np.sqrt(se[indices].sum()/n)-np.sqrt(he[indices].sum()/n)))
    lo,hi=map(float,np.quantile(samples,[.025,.975]))
    return hh,ss,bb,ss-hh,lo,hi,shift

def run(input_dir,elevation_zip,output,draws=2000):
    with ZipFile(elevation_zip) as z:
        elevations=json.loads(z.read(member(z,"station_grid_elevation.json")))
    dz={e["station"]:float(e["bilinear_station_minus_grid_m"]) for e in elevations}
    rows=[]
    for case,(filename,sites) in CASES.items():
        with ZipFile(input_dir/filename) as z:
            manifest=json.loads(z.read(member(z,"retrospective_station_results.json")))
            assert manifest["cross_seed_observation_identity_verified"]
            assert set(manifest["stations"])==set(sites)
            assert manifest["bootstrap_days"]==14 and manifest["bootstrap_draws"]==2000
            for seed in SEEDS:
                rec=next(x for x in manifest["seeds"] if x["seed"]==seed)
                raw=z.read(member(z,f"station-matched-seed-{seed}.csv"))
                if hashlib.sha256(raw).hexdigest()!=rec["matched_csv_sha256"]:
                    raise ValueError("SHA256 of matched rows mismatch")
                frame=pd.read_csv(io.BytesIO(raw))
                groups=json.loads(z.read(member(z,rec["scoring_json"])))["groups"]
                for site in sites:
                    df=frame.loc[frame.station==site].copy()
                    if len(df)<1000 or not df.timestamp_utc.is_unique:
                        raise ValueError("Missing coverage or duplicate timestamps")
                    group=next(g for g in groups if g["station"]==site)
                    for rate in LAPSES:
                        for block in BLOCKS:
                            h,s,b,d,lo,hi,shift=calc(df,dz[site],rate,block,draws)
                            if rate==0 and block==14:
                                if abs(d-group["all"]["delta_source_mean_minus_historical_rmse_K"])>1e-8:
                                    raise ValueError("Archive RMSE not recovered")
                                if abs(lo-group["ci_all"]["ci95_low_K"])>1e-8 or abs(hi-group["ci_all"]["ci95_high_K"])>1e-8:
                                    raise ValueError("Original bootstrap interval not reproduced")
                            rows.append(dict(case=case,station=site,year=manifest["year"],
                                seed=seed,matched_n=len(df),station_minus_grid_m=dz[site],
                                lapse_K_per_km=rate,temperature_adjustment_K=shift,
                                block_days=block,draws=draws,historical_rmse_K=h,
                                source_mean_rmse_K=s,baseline_rmse_K=b,
                                delta_rmse_K=d,ci95_low_K=lo,ci95_high_K=hi,
                                includes_zero=bool(lo<=0<=hi)))
    if len(rows)!=144 or len({(r["case"],r["station"],r["seed"],r["lapse_K_per_km"],r["block_days"]) for r in rows})!=144:
        raise ValueError("Missing or duplicated case/site/seed/lapse/block entries")
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest,index=False,float_format="%.10f")
    print(f"Recomputed {len(rows)} validated source-derived scenario rows at {dest}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--elevation-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.input_dir,a.elevation_zip,a.output)
