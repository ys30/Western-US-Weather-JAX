"""Summarize successful retrospective ASOS evaluation artifacts without inventing skill.

Reads actual per-seed station-score JSON and the verified matched-row manifests.
Creates a concise markdown table for the Actions summary and an archival CSV.
"""
import argparse
import csv
import json
from pathlib import Path

SEEDS=(42,7,123)
STATIONS=("ABQ","ROW")
METRICS=("historical_rmse_K","source_mean_rmse_K","baseline_rmse_K",
         "delta_source_mean_minus_historical_rmse_K")

def collect(folder):
    manifest=json.loads((folder/"retrospective_station_results.json").read_text())
    if not manifest.get("cross_seed_observation_identity_verified"):
        raise ValueError("Cross-seed sample identity has not been verified")
    if manifest.get("time_tolerance_minutes")!=30:
        raise ValueError("Unexpected primary matching tolerance")
    records={int(x["seed"]):x for x in manifest["seeds"]}
    if set(records)!=set(SEEDS):
        raise ValueError("Incomplete seed set")
    rows=[]
    for seed in SEEDS:
        rec=records[seed]
        scores=json.loads((folder/rec["scoring_json"]).read_text())
        groups={(g["station"],int(g["year"])):g for g in scores["groups"]}
        if set(groups)!={(station,2022) for station in STATIONS}:
            raise ValueError(f"Wrong station/year group for seed {seed}")
        matches={x["station"]:x for x in rec["station_match"]}
        if set(matches)!=set(STATIONS):
            raise ValueError("Incomplete station match metadata")
        total=0
        for station in STATIONS:
            match=matches[station]
            n=int(match["matched"])
            total+=n
            if n<1000 or float(match["match_rate"])<0.8:
                raise ValueError(f"Insufficient station coverage: {station}")
            group=groups[(station,2022)]
            if group["all"]["n"]!=n:
                raise ValueError("Score and matching sample-size mismatch")
            all_scores=group["all"]
            if any(all_scores.get(field) is None for field in METRICS):
                raise ValueError("Missing model metrics")
            hot=group["observed_hot"]
            rows.append({
                "seed":seed,"station":station,"year":2022,
                "matched_n":n,"match_rate":float(match["match_rate"]),
                "hot_n":int(group["n_hot"]),
                "hot_ci_eligible":bool(group.get("hot_ci_eligible",False)),
                **{field:float(all_scores[field]) for field in METRICS},
                "observed_hot_delta_K":hot.get("delta_source_mean_minus_historical_rmse_K") if hot else None,
                "annual_ci95_low_K":group["ci_all"]["ci95_low_K"] if group.get("ci_all") else None,
                "annual_ci95_high_K":group["ci_all"]["ci95_high_K"] if group.get("ci_all") else None,
                "hot_ci95_low_K":group["ci_hot"]["ci95_low_K"] if group.get("ci_hot") else None,
                "hot_ci95_high_K":group["ci_hot"]["ci95_high_K"] if group.get("ci_hot") else None,
            })
        if total!=int(rec["matched_rows"]):
            raise ValueError("Mismatch with seed-level matched row count")
    return rows

def render(rows):
    md=["## Southwest 2022: archived CNN predictions vs IEM ASOS",
        "",
        "Retrospective station agreement, **not** operational forecast skill; airport reports may overlap with ERA5-assimilated observations.",
        "",
        "| Station | Seed | Matched n | Match % | Historical RMSE (K) | Source-mean RMSE (K) | Baseline RMSE (K) | Δ source-minus-historical (K) | Hot n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| {station} | {seed} | {matched_n} | {pct:.1f} | {historical_rmse_K:.4f} | {source_mean_rmse_K:.4f} | {baseline_rmse_K:.4f} | {delta_source_mean_minus_historical_rmse_K:+.4f} | {hot_n} |".format(pct=100*r["match_rate"],**r))
    md.extend(["","**Quality gates:** ≥80% and ≥1,000 matched records per station; exact cross-seed matched-time identity; common station coordinates and reference field. Confidence intervals use the archived paired seven-day bootstrap; hot CI is suppressed when sparse."])
    return "\n".join(md)+"\n"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--markdown",required=True)
    p.add_argument("--csv",required=True)
    a=p.parse_args()
    folder=Path(a.input)
    records=collect(folder)
    markdown=render(records)
    md=Path(a.markdown);md.parent.mkdir(parents=True,exist_ok=True);md.write_text(markdown)
    out=Path(a.csv);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    print(markdown)

if __name__=="__main__":
    main()
