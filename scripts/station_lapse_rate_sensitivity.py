"""Exploratory common lapse-rate adjustment to already matched station predictions.

Each station must have independently audited elevation source metadata.
Positive station-minus-grid height => cooler adjusted station forecast.
This is a sensitivity diagnostic, not a trained bias correction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PRED=("historical_K","source_mean_K","baseline_K")

def adjust(frame,elevations,lapse_k_per_km=6.5,delta_field="bilinear_station_minus_grid_m"):
    if not 0<=lapse_k_per_km<=12:
        raise ValueError("Invalid assumed lapse rate")
    if not set(("station","timestamp_utc","observed_K",*PRED)).issubset(frame.columns):
        raise ValueError("Missing matched temperature columns")
    if not {"station",delta_field,"station_elevation_m","era5_bilinear_elevation_m"}.issubset(elevations.columns):
        raise ValueError("Missing audited elevation fields")
    if elevations.station.duplicated().any():
        raise ValueError("Multiple elevation rows for a station")
    merged=frame.merge(elevations[["station",delta_field]],on="station",how="left",validate="many_to_one")
    if merged[delta_field].isna().any():
        raise ValueError("Missing station elevation difference")
    dz=merged[delta_field].to_numpy(float)
    if not np.isfinite(dz).all() or np.max(np.abs(dz))>3000:
        raise ValueError("Invalid station-minus-grid elevation difference")
    # Correct predictions rather than the observations, preserving verification target.
    corrected=merged.copy()
    for name in PRED:
        corrected[name]=pd.to_numeric(corrected[name],errors="raise")-lapse_k_per_km*dz/1000
    return corrected.drop(columns=[delta_field])

def self_test():
    times=pd.date_range("2022-07-01",periods=2,freq="6h",tz="UTC")
    f=pd.DataFrame({"station":["ABQ"]*2,"timestamp_utc":times,"observed_K":[300.,301.],
                    "historical_K":[301.,302.],"source_mean_K":[302.,303.],
                    "baseline_K":[303.,304.]})
    h=pd.DataFrame({"station":["ABQ"],"station_elevation_m":[1500.],
                    "era5_bilinear_elevation_m":[1400.],
                    "bilinear_station_minus_grid_m":[100.]})
    got=adjust(f,h)
    assert np.allclose(got["historical_K"],[300.35,301.35])
    assert np.allclose(got["source_mean_K"]-got["historical_K"],[1.,1.])
    assert np.allclose(adjust(f,h,0)["historical_K"],f["historical_K"])
    print("Common lapse-rate sensitivity QA passed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--matched-csv")
    p.add_argument("--elevation-json")
    p.add_argument("--output",default="outputs/station_validation/lapse_sensitivity.csv")
    p.add_argument("--lapse-rate",type=float,default=6.5)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        if not a.matched_csv or not a.elevation_json: p.error("Matched data and elevation JSON required")
        original=pd.read_csv(a.matched_csv)
        elevation=pd.DataFrame(json.loads(Path(a.elevation_json).read_text()))
        result=adjust(original,elevation,a.lapse_rate)
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(output,index=False)
        print("Wrote elevation sensitivity only:",output,"rows",len(result))
