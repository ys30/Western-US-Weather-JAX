"""Verify extracted GitHub archive and reconstruct station forecasts for three seeds.

Inputs are extracted files from the archived successful GitHub Actions run.
This script does not train models or download observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STATIONS = [("ABQ",35.0402,-106.6092),("ROW",33.2998775,-104.5294031)]

def validate(root):
    manifest=root/"outputs/archive/sha256-manifest.json"
    payload=json.loads(manifest.read_text())
    records=payload["files"]
    if len(records)!=payload["count"]:
        raise ValueError("Archive manifest record count mismatch")
    for record in records:
        p=root/record["path"]
        if not p.is_file():
            raise ValueError("Missing archived entry: "+str(p))
        digest=hashlib.sha256()
        with p.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):
                digest.update(block)
        if digest.hexdigest()!=record["sha256"] or p.stat().st_size!=record["bytes"]:
            raise ValueError("Archive SHA256/size mismatch: "+str(p))
    return len(records)

def run(root,out):
    verified=validate(root)
    out.mkdir(parents=True,exist_ok=True)
    import pandas as pd
    station_csv=out/"stations.csv"
    pd.DataFrame(STATIONS,columns=["station","station_lat","station_lon"]).to_csv(station_csv,index=False)
    results=[]
    for seed in (42,7,123):
        historical=root/f"outputs/archive/historical/fields-{seed}.npz"
        source_mean=root/f"outputs/archive/source_mean/fields-{seed}.npz"
        grid=out/f"fields-seed-{seed}.nc"
        station_pred=out/f"station-predictions-seed-{seed}.csv"
        subprocess.run([sys.executable,str(ROOT/"scripts/reconstruct_matched_archive_fields.py"),
                        "--historical",str(historical),"--source-mean",str(source_mean),
                        "--output",str(grid)],check=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/sample_forecast_grid_at_stations.py"),
                        "--netcdf",str(grid),"--stations",str(station_csv),
                        "--output",str(station_pred)],check=True)
        station=pd.read_csv(station_pred)
        expected=2*1460
        if len(station)!=expected:
            raise ValueError(f"Expected {expected} station-time rows, found {len(station)} for seed {seed}")
        row={"seed":seed,"station_rows":int(len(station)),"station_count":int(station.station.nunique()),
             "grid_sha256":hashlib.sha256(grid.read_bytes()).hexdigest(),
             "station_csv_sha256":hashlib.sha256(station_pred.read_bytes()).hexdigest()}
        results.append(row)
        grid.unlink()  # avoid uploading intermediate multi-megabyte gridded NetCDF
    output={"source_case":"Southwest 2022","source_run":41400003722,
            "source_artifact":13122165252,"verified_archive_manifest_records":verified,
            "station_sampling":"bilinear","observations_matched":False,"seeds":results}
    (out/"archive_spatial_acceptance.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive-root",default="downloaded")
    parser.add_argument("--output-dir",default="outputs/archived_station_acceptance")
    opts=parser.parse_args()
    run(Path(opts.archive_root),Path(opts.output_dir))
