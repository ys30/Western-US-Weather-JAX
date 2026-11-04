"""Rebuild the full Southwest 2022 ASOS elevation-bootstrap sensitivity table.

Inputs are the original verified GitHub Actions ZIP artifacts. Recompute
paired annual RMSE differences by 7, 14 and 28-day circular calendar-block
bootstrap for 4 assumed lapse rates, retaining all shared station timestamps.

Examples:
  python scripts/recompute_asos_elevation_bootstrap.py \
    --asos-archive southwest-asos-results.zip \
    --elevation-archive station-grid-elevation-audit.zip \
    --output outputs/asos-elevation-bootstrap-full-72.csv
"""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

SEEDS=(42,7,123)
STATIONS=("ABQ","ROW")
LAPSE=(0.,3.,6.5,9.8)
BLOCKS=(7,14,28)
RNG_SEED=20261103

def find_member(archive, suffix):
    matches=[n for n in archive.namelist() if n==suffix or n.endswith("/"+suffix)]
    if len(matches)!=1:
        raise ValueError(f"Expected exactly one {suffix}, found {matches}")
    return matches[0]

def bootstrap(df, lapse, delta_z_m, blocks, draws):
    df=df.sort_values("timestamp_utc").copy()
    dates=pd.to_datetime(df["timestamp_utc"],utc=True).dt.floor("D")
    first=dates.iloc[0]
    n_days=len(pd.date_range(first,dates.iloc[-1],freq="D"))
    indices=((dates-first)/pd.Timedelta(days=1)).to_numpy(dtype=int)
    truth=df["observed_K"].to_numpy(dtype=float)
    correction=-lapse*delta_z_m/1000.
    h=df["historical_K"].to_numpy(dtype=float)+correction
    s=df["source_mean_K"].to_numpy(dtype=float)+correction
    day_count=np.bincount(indices,minlength=n_days)
    h_sse=np.bincount(indices,weights=(h-truth)**2,minlength=n_days)
    s_sse=np.bincount(indices,weights=(s-truth)**2,minlength=n_days)
    delta=float(np.sqrt(np.mean((s-truth)**2))-np.sqrt(np.mean((h-truth)**2)))
    rng=np.random.default_rng(RNG_SEED)
    n_starts=(n_days+blocks-1)//blocks
    offsets=np.arange(blocks,dtype=int)
    samples=[]
    for _ in range(draws):
        starts=rng.integers(0,n_days,size=n_starts)
        ids=((starts[:,None]+offsets[None,:])%n_days).ravel()[:n_days]
        count=day_count[ids].sum()
        if count==0:
            raise ValueError("Bootstrap draw had no observed reports")
        samples.append(float(np.sqrt(s_sse[ids].sum()/count)
                             -np.sqrt(h_sse[ids].sum()/count)))
    lo,hi=np.quantile(samples,[.025,.975])
    return delta,float(lo),float(hi)

def run(asos_archive, elevation_archive, output, draws=2000):
    with ZipFile(asos_archive) as az, ZipFile(elevation_archive) as ez:
        elevations=json.loads(ez.read(find_member(ez,"station_grid_elevation.json")))
        delta={row["station"]:float(row["bilinear_station_minus_grid_m"]) for row in elevations}
        if not set(STATIONS).issubset(delta):
            raise ValueError("Missing source-audited station elevation")
        frames={}
        for seed in SEEDS:
            target=f"station-matched-seed-{seed}.csv"
            data=pd.read_csv(io.BytesIO(az.read(find_member(az,target))))
            data["timestamp_utc"]=pd.to_datetime(data["timestamp_utc"],utc=True)
            frames[seed]=data
    records=[]
    baseline_keys=None
    for seed in SEEDS:
        data=frames[seed]
        keys=data[["station","timestamp_utc","observation_time_utc"]].sort_values(
            ["station","timestamp_utc"]).reset_index(drop=True)
        if baseline_keys is None:
            baseline_keys=keys
        elif not keys.equals(baseline_keys):
            raise ValueError("Non-identical observations across three seed pairs")
        for site in STATIONS:
            part=data.loc[data.station==site]
            if len(part)<1000:
                raise ValueError(f"Too few matched observations at {site}")
            for lapse in LAPSE:
                for days in BLOCKS:
                    estimate,lo,hi=bootstrap(part,lapse,delta[site],days,draws)
                    records.append({"station":site,"seed":seed,
                        "lapse_K_per_km":lapse,"block_days":days,"draws":draws,
                        "paired_delta_rmse_K":estimate,"ci95_low_K":lo,
                        "ci95_high_K":hi,"includes_zero":bool(lo<=0<=hi)})
    if len(records)!=72 or len({(r["station"],r["seed"],r["lapse_K_per_km"],r["block_days"]) for r in records})!=72:
        raise ValueError("Incomplete 72-scenario sensitivity factorial")
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out,index=False,float_format="%.10f")
    print(f"Wrote {len(records)} rows: {out}")
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--asos-archive",required=True)
    parser.add_argument("--elevation-archive",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--draws",type=int,default=2000)
    args=parser.parse_args()
    if args.draws<20:
        parser.error("draws must be >=20")
    run(args.asos_archive,args.elevation_archive,args.output,args.draws)
