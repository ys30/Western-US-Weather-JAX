"""Complete Southwest 2022 archived predictions versus two IEM ASOS stations.

Run after accept_archived_station_sampling.py has produced station-predictions
for seeds 42,7,123. All raw IEM responses and audit metadata stay in Actions
artifacts. This is retrospective agreement with point observations, not an
operational forecast validation or an ERA5-independent verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fetch_asos_station_validation import STATIONS, fetch
from match_asos_prediction_times import PRED, load_csv, match
from evaluate_paired_station_predictions import evaluate

def run(sampled_root,output,year=2022,draws=2000):
    output.mkdir(parents=True,exist_ok=True)
    obs_dir=output/"raw_station_observations"
    obs_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for station_id in ("ABQ","ROW"):
        station=next(s for s in STATIONS if s["station"]==station_id)
        path=obs_dir/f"{station_id}-{year}.csv"
        records.append(fetch(station,year,path))
        time.sleep(15)  # honor public IEM service limits
    observations=pd.concat([pd.read_csv(obs_dir/f"{s}-{year}.csv")
                            for s in ("ABQ","ROW")],ignore_index=True)
    obs_path=output/"station-observations-combined.csv"
    observations.to_csv(obs_path,index=False)
    o=load_csv(obs_path,("temperature_K",),"ASOS")
    result=[]
    matched_reference_keys=None
    for seed in (42,7,123):
        source=sampled_root/f"station-predictions-seed-{seed}.csv"
        p=load_csv(source,PRED,"spatial predictions")
        # Changing the tolerance affects report coverage and potentially the
        # verification population; retain counts before comparing scores.
        tolerance_counts={}
        for tolerance in (15,30,60):
            try:
                candidate,_=match(p,o,tolerance_minutes=tolerance)
                tolerance_counts[str(tolerance)]=int(len(candidate))
            except ValueError as exc:
                if "No predictions could be matched" not in str(exc):
                    raise
                tolerance_counts[str(tolerance)]=0
        matched,matching=match(p,o,tolerance_minutes=30)
        # Do not report apparent skill from a small or incomparable station subset.
        for item in matching:
            if item["predictions"]!=1460 or item["matched"]<1000 or item.get("match_rate",0)<0.80:
                raise ValueError("Station coverage gate failed: "+json.dumps(item))
        keys=matched[["station","timestamp_utc","observation_time_utc"]].sort_values(
            ["station","timestamp_utc"]).reset_index(drop=True)
        if matched_reference_keys is None:
            matched_reference_keys=keys
        elif not matched_reference_keys.equals(keys):
            raise ValueError("Different valid-time/observation keys across random seeds")
        path=output/f"station-matched-seed-{seed}.csv"
        matched.to_csv(path,index=False)
        scored=output/f"station-scored-seed-{seed}.json"
        # Paper-aligned primary paired interval.
        evaluation=evaluate(path,block_days=14,draws=draws)
        scored.write_text(json.dumps(evaluation,indent=2,allow_nan=False)+"\n")
        # Same paired observations, with alternative calendar block lengths.
        for block_days in (7,28):
            sensitivity=evaluate(path,block_days=block_days,draws=draws)
            sensitivity_path=output/f"station-scored-seed-{seed}-block-{block_days}d.json"
            sensitivity_path.write_text(json.dumps(sensitivity,indent=2,allow_nan=False)+"\n")
        result.append({"seed":seed,"matched_rows":len(matched),
                       "station_match":matching,"tolerance_matched_rows":tolerance_counts,
                       "scoring_json":scored.name,
                       "matched_csv_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    summary={"case":"Southwest 2022","source_archive_run":41400003722,
             "source_artifact_id":13122165252,
             "station_source":"Iowa Environmental Mesonet IEM ASOS/METAR archived temperatures",
             "year":year,"stations":["ABQ","ROW"],
             "observation_files":records,
             "matched_observation_sha256":hashlib.sha256(obs_path.read_bytes()).hexdigest(),
             "time_tolerance_minutes":30,
             "bootstrap_days":14,"bootstrap_sensitivity_days":[7,28],"bootstrap_draws":draws,
             "required_match_rate":0.80,"required_matched_per_station":1000,
             "cross_seed_observation_identity_verified":True,
             "interpretation":"Retrospective observational agreement; not operational or independent from ERA5 assimilation",
             "seeds":result}
    (output/"retrospective_station_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"matched_rows_by_seed":[x["matched_rows"] for x in result],
                      "stations":["ABQ","ROW"],"year":year},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--sampled-root",required=True)
    p.add_argument("--output",default="outputs/station_observation_evaluation")
    p.add_argument("--draws",type=int,default=2000)
    a=p.parse_args()
    if a.draws<20: p.error("draws must be >=20")
    run(Path(a.sampled_root),Path(a.output),draws=a.draws)
