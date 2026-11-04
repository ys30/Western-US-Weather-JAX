"""Offline regression for the ASOS station result summary's acceptance gates."""
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_asos_station_acceptance import collect, render

def run():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        seeds=[]
        for seed in (42,7,123):
            groups=[]
            matches=[]
            for station in ("ABQ","ROW"):
                n=1200
                score={"n":n,"historical_rmse_K":1.3,"source_mean_rmse_K":1.4,
                       "baseline_rmse_K":1.9,
                       "delta_source_mean_minus_historical_rmse_K":0.1}
                groups.append({"station":station,"year":2022,"n_hot":42,
                               "hot_ci_eligible":True,
                               "all":score,"observed_hot":score,
                               "ci_all":{"ci95_low_K":-0.1,"ci95_high_K":0.2},
                               "ci_hot":{"ci95_low_K":-0.2,"ci95_high_K":0.4}})
                matches.append({"station":station,"predictions":1460,
                                "matched":n,"match_rate":n/1460})
            filename=f"station-scored-seed-{seed}.json"
            (root/filename).write_text(json.dumps({"groups":groups}))
            seeds.append({"seed":seed,"scoring_json":filename,
                          "matched_rows":2400,"station_match":matches})
        (root/"retrospective_station_results.json").write_text(json.dumps({
            "cross_seed_observation_identity_verified":True,
            "time_tolerance_minutes":30,"seeds":seeds}))
        rows=collect(root)
        assert len(rows)==6
        assert len([line for line in render(rows).splitlines() if line.startswith("| ABQ")])==3
        seeds[0]["station_match"][0]["matched"]=100
        (root/"retrospective_station_results.json").write_text(json.dumps({
            "cross_seed_observation_identity_verified":True,
            "time_tolerance_minutes":30,"seeds":seeds}))
        try:
            collect(root)
        except ValueError as err:
            assert "coverage" in str(err)
        else:
            raise AssertionError("Low-coverage sample incorrectly accepted")
    print("Station summary extraction and coverage regression passed")

if __name__=="__main__":
    run()
