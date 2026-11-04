"""Recompute retrospective IEM ASOS model error decomposition from accepted CI artifacts.

Uses the three original matched station-record artifacts. This is a descriptive
error audit, not bias correction, causality, or independent forecast validation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

CASES={
    "SW22":("southwest-2022-asos.zip",("ABQ","ROW")),
    "SW23":("southwest-2023-asos.zip",("ABQ","ROW")),
    "SP22":("southern-plains-2022-asos.zip",("ABI","OKC")),
}
SEEDS=(42,7,123)
MODELS=("historical","source_mean","baseline")

def member(z,name):
    matches=[n for n in z.namelist() if n==name or n.endswith("/"+name)]
    if len(matches)!=1:
        raise ValueError(f"Missing/ambiguous archived member {name}: {matches}")
    return matches[0]

def build(directory):
    rows=[]
    for case,(filename,sites) in CASES.items():
        with ZipFile(directory/filename) as archive:
            metadata=json.loads(archive.read(member(archive,"retrospective_station_results.json")))
            if not metadata.get("cross_seed_observation_identity_verified") or set(metadata["stations"])!=set(sites):
                raise ValueError("Archive station identity/selection mismatch")
            reference=None
            for seed in SEEDS:
                record=next(r for r in metadata["seeds"] if int(r["seed"])==seed)
                payload=archive.read(member(archive,f"station-matched-seed-{seed}.csv"))
                if hashlib.sha256(payload).hexdigest()!=record["matched_csv_sha256"]:
                    raise ValueError("Matched CSV SHA256 mismatch")
                frame=pd.read_csv(io.BytesIO(payload))
                if set(frame.station.unique())!=set(sites):
                    raise ValueError("Wrong airport stations")
                keys=frame[["station","timestamp_utc","observation_time_utc"]].sort_values(
                    ["station","timestamp_utc"]).reset_index(drop=True)
                if reference is None: reference=keys
                elif not keys.equals(reference): raise ValueError("Seeds use different verification records")
                scored=json.loads(archive.read(member(archive,record["scoring_json"])))
                for site in sites:
                    data=frame[frame.station==site]
                    truth=data.observed_K.to_numpy(float)
                    if len(data)<1000 or not np.isfinite(truth).all():
                        raise ValueError("Insufficient or nonfinite station observations")
                    group=next(g for g in scored["groups"] if g["station"]==site)
                    for model in MODELS:
                        forecast=data[f"{model}_K"].to_numpy(float)
                        error=forecast-truth
                        if not np.isfinite(error).all():
                            raise ValueError("Nonfinite forecast error")
                        rmse=float(np.sqrt(np.mean(error**2)))
                        score_name=f"{model}_rmse_K"
                        if abs(rmse-float(group["all"][score_name]))>1e-8:
                            raise ValueError(f"Archived RMSE does not reproduce: {case} {site} {seed} {model}")
                        bias=float(np.mean(error))
                        centered=float(np.sqrt(np.mean((error-bias)**2)))
                        rows.append(dict(case=case,station=site,seed=seed,model=model,
                             matched_n=len(data),rmse_K=rmse,mean_error_K=bias,
                             centered_rmse_K=centered,mae_K=float(np.mean(np.abs(error))),
                             pearson_r=float(np.corrcoef(truth,forecast)[0,1]),
                             squared_bias_fraction=bias*bias/(rmse*rmse)))
    expected=6*3*3
    if len(rows)!=expected: raise ValueError(f"Expected {expected} model/site/seed rows")
    for r in rows:
        if abs(r["rmse_K"]**2-r["mean_error_K"]**2-r["centered_rmse_K"]**2)>1e-8:
            raise ValueError("RMSE² != mean_error² + centered_RMSE²")
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    data=build(args.input_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.output,index=False,float_format="%.9f")
    print("Audited",len(data),"model × seed × station-year rows")
    print(data.query("model == 'historical'").groupby(["case","station"])
          .agg(mean_error_median_K=("mean_error_K","median"),
               squared_bias_fraction_median=("squared_bias_fraction","median")).to_string())

if __name__=="__main__":
    main()
