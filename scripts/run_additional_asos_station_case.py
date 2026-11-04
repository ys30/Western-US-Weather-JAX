"""Archive-backed retrospective ASOS validation for additional region-years.

Original frozen source-trained historical/source_mean predictions are restored
from independently hash-verified GitHub Actions archives. No model retraining.
This script deliberately keeps Southwest 2022's accepted workflow unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from accept_archived_station_sampling import validate
from reconstruct_matched_archive_fields import reconstruct
from sample_forecast_grid_at_stations import sample
from fetch_asos_station_validation import STATIONS, fetch
from match_asos_prediction_times import PRED, load_csv, match
from evaluate_paired_station_predictions import evaluate

CASES={
 "southwest-2023":{"year":2023,"stations":("ABQ","ROW"),
                   "source_run":41402185866,"source_artifact_id":13122890221},
 "southern-plains-2022":{"year":2022,"stations":("ABI","OKC"),
                          "source_run":41402189567,"source_artifact_id":13121977420},
}
SEEDS=(42,7,123)

def run(case,archive_root,output,draws=2000):
    cfg=CASES[case]
    year=cfg["year"]
    stations=[next(x for x in STATIONS if x["station"]==name) for name in cfg["stations"]]
    verified=validate(archive_root)
    output.mkdir(parents=True,exist_ok=True)
    obs_dir=output/"observations"
    obs_dir.mkdir(exist_ok=True)
    audit=[]
    for st in stations:
        path=obs_dir/f'{st["station"]}-{year}.csv'
        audit.append(fetch(st,year,path))
        time.sleep(15)
    obs=pd.concat([pd.read_csv(obs_dir/f'{st["station"]}-{year}.csv') for st in stations],ignore_index=True)
    observations_path=output/"observations_combined.csv"
    obs.to_csv(observations_path,index=False)
    o=load_csv(observations_path,("temperature_K",),"IEM ASOS")
    table=pd.DataFrame([{"station":s["station"],"station_lat":s["lat"],"station_lon":s["lon"]} for s in stations])
    matched_keys=None
    seed_records=[]
    for seed in SEEDS:
        h=archive_root/f"outputs/archive/historical/fields-{seed}.npz"
        s=archive_root/f"outputs/archive/source_mean/fields-{seed}.npz"
        ds=reconstruct(h,s)
        # 2022/2023 each have precisely 365 days of 6-hourly target-valid times.
        if ds.sizes["time"]!=1460:
            raise ValueError(f"Wrong number of target timestamps in {case}: {ds.sizes['time']}")
        times=pd.to_datetime(ds.time.values,utc=True)
        if times.min().year!=year or times.max().year!=year:
            raise ValueError(f"Archived targets do not match expected {year} year")
        predictions=sample(ds,table)
        pred_path=output/f"station-predictions-seed-{seed}.csv"
        predictions.to_csv(pred_path,index=False)
        p=load_csv(pred_path,PRED,"archived prediction")
        tolerance_counts={}
        for tolerance in (15,30,60):
            try:
                test,_=match(p,o,tolerance)
                tolerance_counts[str(tolerance)]=len(test)
            except ValueError as exc:
                if "No predictions could be matched" not in str(exc):
                    raise
                tolerance_counts[str(tolerance)]=0
        matched,counts=match(p,o,30)
        if {x["station"] for x in counts}!=set(cfg["stations"]):
            raise ValueError(f"Incomplete station coverage {counts}")
        for item in counts:
            if item["predictions"]!=1460 or item["matched"]<1000 or item["match_rate"]<0.8:
                raise ValueError(f"Station coverage failed: {item}")
        keys=matched[["station","timestamp_utc","observation_time_utc"]].sort_values(["station","timestamp_utc"]).reset_index(drop=True)
        if matched_keys is None:
            matched_keys=keys
        elif not keys.equals(matched_keys):
            raise ValueError("Inconsistent paired observation timestamps across seeds")
        matched_path=output/f"station-matched-seed-{seed}.csv"
        matched.to_csv(matched_path,index=False)
        score_path=output/f"station-scored-seed-{seed}.json"
        score=evaluate(matched_path,block_days=14,draws=draws)
        score_path.write_text(json.dumps(score,indent=2,allow_nan=False)+"\\n")
        for block in (7,28):
            sub=evaluate(matched_path,block_days=block,draws=draws)
            (output/f"station-scored-seed-{seed}-block-{block}d.json").write_text(
                json.dumps(sub,indent=2,allow_nan=False)+"\\n")
        seed_records.append({"seed":seed,"matched_rows":len(matched),
              "station_match":counts,"tolerance_matched_rows":tolerance_counts,
              "scoring_json":score_path.name,
              "matched_csv_sha256":hashlib.sha256(matched_path.read_bytes()).hexdigest()})
    manifest={"case":case,"source_archive_run":cfg["source_run"],
       "source_artifact_id":cfg["source_artifact_id"],"year":year,
       "stations":list(cfg["stations"]),"verified_archive_manifest_records":verified,
       "observation_files":audit,
       "observation_combined_sha256":hashlib.sha256(observations_path.read_bytes()).hexdigest(),
       "time_tolerance_minutes":30,
       "bootstrap_days":14,"bootstrap_sensitivity_days":[7,28],
       "bootstrap_draws":draws,
       "cross_seed_observation_identity_verified":True,
       "required_match_rate":0.8,"required_matched_per_station":1000,
       "interpretation":"Retrospective airport station agreement; potential overlap with ERA5 assimilation, not operational forecasting",
       "seeds":seed_records}
    (output/"retrospective_station_results.json").write_text(json.dumps(manifest,indent=2)+"\\n")
    print(json.dumps({"case":case,"stations":cfg["stations"],
                      "matched_by_seed":[r["matched_rows"] for r in seed_records]},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--archive-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--draws",type=int,default=2000)
    a=p.parse_args()
    if a.draws<20: p.error("At least 20 paired bootstrap draws required")
    run(a.case,a.archive_root,a.output,a.draws)
