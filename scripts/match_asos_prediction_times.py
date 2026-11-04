"""Match already spatially sampled target-valid-time predictions to IEM ASOS reports.

Predictions CSV columns: station,timestamp_utc,historical_K,source_mean_K,baseline_K
Observations CSV columns: station,timestamp_utc,temperature_K

Predictions must have been bilinearly sampled at the station's coordinates, with
a shared grid and *verifying* valid time. This module does not interpolate grids.
It produces the strict input schema for evaluate_paired_station_predictions.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRED = ("historical_K", "source_mean_K", "baseline_K")
OUTPUT = ("station", "timestamp_utc", "observed_K") + PRED


def load_csv(path, fields, label):
    frame = pd.read_csv(path)
    needed = {"station", "timestamp_utc"} | set(fields)
    absent = sorted(needed - set(frame))
    if absent:
        raise ValueError(f"{label} missing: {absent}")
    frame = frame[["station", "timestamp_utc", *fields]].copy()
    frame["station"] = frame["station"].astype(str).str.upper().str.strip()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if frame[["station", "timestamp_utc"]].isna().any().any():
        raise ValueError(f"{label}: missing station/time")
    if frame.duplicated(["station", "timestamp_utc"]).any():
        raise ValueError(f"{label}: duplicate station and timestamp")
    if not np.isfinite(frame[list(fields)].to_numpy()).all():
        raise ValueError(f"{label}: non-finite temperature")
    return frame.sort_values(["station", "timestamp_utc"]).reset_index(drop=True)


def match(predictions, observations, tolerance_minutes=30):
    if not 0 < tolerance_minutes <= 60:
        raise ValueError("tolerance must be in (0,60] minutes")
    if not set(observations.station).issubset(set(predictions.station)):
        raise ValueError("Observation contains unknown station")
    tol = pd.Timedelta(minutes=tolerance_minutes)
    output = []
    stats = []
    for station, p in predictions.groupby("station", sort=True):
        o = observations.loc[observations.station == station].copy()
        p = p.sort_values("timestamp_utc").reset_index(drop=True)
        o = o.sort_values("timestamp_utc").reset_index(drop=True)
        o = o.rename(columns={"timestamp_utc": "observation_time_utc",
                              "temperature_K": "observed_K"})
        if o.empty:
            stats.append({"station": station, "predictions": len(p), "matched": 0})
            continue
        joined = pd.merge_asof(
            p, o.drop(columns="station"),
            left_on="timestamp_utc", right_on="observation_time_utc",
            direction="nearest", tolerance=tol,
        )
        matched = joined.dropna(subset=["observed_K"]).copy()
        if matched["observation_time_utc"].duplicated().any():
            raise ValueError(f"{station}: one observation matched multiple forecasts")
        matched["time_offset_minutes"] = (
            (matched["observation_time_utc"] - matched["timestamp_utc"])
            .dt.total_seconds() / 60
        )
        stats.append({"station": station, "predictions": int(len(p)),
                      "matched": int(len(matched)),
                      "match_rate": float(len(matched) / len(p)),
                      "max_abs_offset_minutes": (
                          float(matched.time_offset_minutes.abs().max())
                          if len(matched) else None)})
        output.append(matched.loc[:, list(OUTPUT) + ["observation_time_utc", "time_offset_minutes"]])
    if output:
        result = pd.concat(output, ignore_index=True)
    else:
        result = pd.DataFrame(columns=list(OUTPUT) + ["observation_time_utc", "time_offset_minutes"])
    if result.empty:
        raise ValueError("No predictions could be matched to observations")
    return result, stats


def run(prediction_csv, observation_csv, output_csv, tolerance_minutes):
    p = load_csv(prediction_csv, PRED, "predictions")
    o = load_csv(observation_csv, ("temperature_K",), "observations")
    out, stats = match(p, o, tolerance_minutes)
    dest = Path(output_csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Keep station report timestamp/offset in the audit CSV, even though scoring
    # consumes only the six required fields.
    out.to_csv(dest, index=False)
    report = {"matching": "nearest UTC observation to prediction target valid time",
              "tolerance_minutes": tolerance_minutes, "stations": stats,
              "rows": len(out),
              "output_csv_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
              "note": "Pre-spatially-matched forecast fields; no grid interpolation was performed by this script"}
    dest.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def self_test():
    p = pd.DataFrame({"station": ["ABQ"] * 3, "timestamp_utc": pd.date_range(
        "2022-01-01", periods=3, freq="6h", tz="UTC"),
        "historical_K": [300.0] * 3, "source_mean_K": [301.0] * 3,
        "baseline_K": [299.0] * 3})
    o = pd.DataFrame({"station": ["ABQ"] * 3,
        "timestamp_utc": pd.to_datetime(["2022-01-01T00:20Z",
                                          "2022-01-01T06:40Z",
                                          "2022-01-01T12:00Z"], utc=True),
        "temperature_K": [300.5] * 3})
    matched, summary = match(p, o, 30)
    assert len(matched) == 2
    assert summary[0]["matched"] == 2
    assert matched["time_offset_minutes"].tolist() == [20, 0]
    try:
        match(p, o, 15)
    except ValueError as exc:
        assert "No predictions" not in str(exc)
        raise
    else:
        # A 15 minute tolerance yields one exact valid-time report.
        pass
    narrow, _ = match(p, o, 15)
    assert len(narrow) == 1
    o.loc[1, "timestamp_utc"] = o.loc[0, "timestamp_utc"]
    try:
        load_check = o.duplicated(["station", "timestamp_utc"])
        assert load_check.any()
    except Exception:
        raise
    print("Synthetic nearest-time matching tests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--observations")
    parser.add_argument("--output", default="outputs/station_validation/matched_station_predictions.csv")
    parser.add_argument("--tolerance-minutes", type=float, default=30)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.predictions or not args.observations:
            parser.error("--predictions and --observations required")
        print(json.dumps(run(args.predictions, args.observations, args.output, args.tolerance_minutes), indent=2))
