"""Fail-closed reporting of archived multi-region IEM ASOS cases."""
import argparse
import csv
import json
import math
from pathlib import Path

def summarize(folder):
    doc=json.loads((folder/"retrospective_station_results.json").read_text())
    if not doc.get("cross_seed_observation_identity_verified"):
        raise ValueError("Missing exact matched observation identity gate")
    if doc.get("bootstrap_days")!=14 or doc.get("bootstrap_draws",0)<2000:
        raise ValueError("Paper-aligned 14-day 2000-draw evaluation required")
    if doc["time_tolerance_minutes"]!=30:
        raise ValueError("Unexpected matching tolerance")
    expected=set(doc["stations"])
    if len(expected)!=2 or set(x["seed"] for x in doc["seeds"])!={7,42,123}:
        raise ValueError("Incomplete site/seed sample")
    records=[]
    for seed in doc["seeds"]:
        score=json.loads((folder/seed["scoring_json"]).read_text())
        groups={g["station"]:g for g in score["groups"] if int(g["year"])==int(doc["year"])}
        if set(groups)!=expected or len(score["groups"])!=len(expected):
            raise ValueError("Score group mismatch")
        matches={v["station"]:v for v in seed["station_match"]}
        if set(matches)!=expected:
            raise ValueError("Station match mismatch")
        count=0
        for site in sorted(expected):
            g=groups[site]
            m=matches[site]
            n=int(m["matched"])
            count+=n
            if n<1000 or m["match_rate"]<0.8 or g["all"]["n"]!=n:
                raise ValueError("Coverage or score count failure")
            h=float(g["all"]["historical_rmse_K"])
            s=float(g["all"]["source_mean_rmse_K"])
            b=float(g["all"]["baseline_rmse_K"])
            d=float(g["all"]["delta_source_mean_minus_historical_rmse_K"])
            if not all(math.isfinite(z) for z in (h,s,b,d)) or abs((s-h)-d)>1e-8:
                raise ValueError("Nonfinite or inconsistent model RMSE")
            ci=g["ci_all"]
            if not ci or ci["block_days"]!=14 or ci["draws"]<1600:
                raise ValueError("No primary bootstrap evidence")
            lo=float(ci["ci95_low_K"]);hi=float(ci["ci95_high_K"])
            if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:
                raise ValueError("Invalid primary confidence bounds")
            records.append(dict(case=doc["case"],year=doc["year"],station=site,
                seed=seed["seed"],matched_n=n,match_rate=m["match_rate"],
                historical_rmse_K=h,source_mean_rmse_K=s,baseline_rmse_K=b,
                delta_source_minus_historical_rmse_K=d,
                annual_ci95_low_K=lo,annual_ci95_high_K=hi,
                hot_n=g["n_hot"],hot_days=g["n_hot_days"],
                hot_ci_eligible=g["hot_ci_eligible"]))
        if count!=seed["matched_rows"]:
            raise ValueError("Seed total mismatch")
        for days in (7,28):
            p=folder/f'station-scored-seed-{seed["seed"]}-block-{days}d.json'
            if not p.is_file():
                raise ValueError(f"Missing {days}-day sensitivity JSON")
    if len(records)!=6:
        raise ValueError("Expected six accepted station×seed records")
    return records

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    rows=summarize(a.input)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print("Retrospective ASOS station comparison; not operational forecast validation.")
    for r in rows:
        print(f'{r["case"]} {r["station"]} seed={r["seed"]}: n={r["matched_n"]} '
              f'historical={r["historical_rmse_K"]:.4f} '
              f'source-mean={r["source_mean_rmse_K"]:.4f} '
              f'delta={r["delta_source_minus_historical_rmse_K"]:+.4f} K, '
              f'CI=[{r["annual_ci95_low_K"]:+.4f},{r["annual_ci95_high_K"]:+.4f}]')

if __name__=="__main__":
    main()
