"""Paired retrospective ASOS station evaluation.

Input is a *pre-matched* long CSV, one row per station and UTC valid time.
Required columns:
station,timestamp_utc,observed_K,historical_K,source_mean_K,baseline_K

The two CNN fields must be sampled from the SAME target-valid-time grid,
at the SAME station coordinate as the baseline. This script does NOT
perform temporal or spatial interpolation. It rejects incomplete pairs.
Results are exploratory; METAR may be assimilated into ERA5.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("station", "timestamp_utc", "observed_K",
            "historical_K", "source_mean_K", "baseline_K")
MODELS = ("historical_K", "source_mean_K", "baseline_K")


def rmse(observed, forecast):
    return float(np.sqrt(np.mean((np.asarray(forecast) - np.asarray(observed)) ** 2)))


def metrics(frame):
    if frame.empty:
        return None
    truth = frame["observed_K"].to_numpy()
    result = {"n": int(len(frame))}
    for field in MODELS:
        prediction = frame[field].to_numpy()
        result[field.replace("_K", "_rmse_K")] = rmse(truth, prediction)
        result[field.replace("_K", "_mae_K")] = float(np.mean(np.abs(prediction - truth)))
        result[field.replace("_K", "_bias_K")] = float(np.mean(prediction - truth))
    result["delta_source_mean_minus_historical_rmse_K"] = (
        result["source_mean_rmse_K"] - result["historical_rmse_K"]
    )
    return result


def daily_block_interval(frame, days=14, draws=1000, seed=20261103):
    """Circular moving-day bootstrap; all reports from a sampled day stay together."""
    indexed = [(day, piece) for day, piece in frame.groupby(frame["timestamp_utc"].dt.floor("D"), sort=True)]
    n = len(indexed)
    if n < 2:
        return None
    # Sample consecutive *observed calendar dates* only: do not bridge missing dates.
    all_days = pd.date_range(indexed[0][0], indexed[-1][0], freq="D", tz="UTC")
    pieces = {day: part for day, part in indexed}
    if len(all_days) != n:
        # Missing days may have zero station observations. Keep empty days in the resampling frame.
        daily = [pieces.get(day) for day in all_days]
    else:
        daily = [part for _, part in indexed]
    n = len(daily)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(n))
            indices.extend((start + k) % n for k in range(days))
        sampled = [daily[i] for i in indices[:n] if daily[i] is not None]
        if not sampled:
            continue
        m = metrics(pd.concat(sampled, ignore_index=True))
        values.append(m["delta_source_mean_minus_historical_rmse_K"])
    if len(values) < max(10, draws // 2):
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return {"block_days": days, "draws": len(values),
            "ci95_low_K": float(low), "ci95_high_K": float(high)}


def evaluate(path, threshold=305.0, block_days=14, draws=1000):
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = data.loc[:, REQUIRED].copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
    for col in ("observed_K",) + MODELS:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data["station"].isna().any() or data["timestamp_utc"].isna().any():
        raise ValueError("Missing station or valid timestamp")
    if data.duplicated(["station", "timestamp_utc"]).any():
        raise ValueError("Duplicate station/valid-time rows: deduplicate upstream with an audit")
    if not np.isfinite(data[list(("observed_K",) + MODELS)].to_numpy()).all():
        raise ValueError("Missing or nonfinite prediction/observation; no unpaired filtering")
    if not data["observed_K"].between(170, 350).all():
        raise ValueError("Observed temperature is outside broad physical QA bounds")
    data["year"] = data["timestamp_utc"].dt.year
    groups = []
    for (site, year), group in data.groupby(["station", "year"], sort=True):
        hot = group.loc[group["observed_K"] >= threshold]
        groups.append({
            "station": str(site), "year": int(year),
            "all": metrics(group),
            "observed_hot": metrics(hot),
            "n_days": int(group["timestamp_utc"].dt.floor("D").nunique()),
            "n_hot": int(len(hot)),
            "ci_all": daily_block_interval(group, block_days, draws),
            "ci_hot": daily_block_interval(hot, block_days, draws) if len(hot) else None,
        })
    return {
        "schema": "station-paired-evaluation-v1",
        "threshold_observed_K": threshold,
        "comparison_sign": "source_mean minus historical: positive favors historical",
        "interpretation": "Retrospective, exploratory, station-time matched; not an operational forecast or independent ERA5 validation",
        "groups": groups,
        "pooled_descriptive_only": metrics(data),
    }


def self_test():
    import tempfile
    import os
    times = pd.date_range("2022-07-01", periods=24, freq="6h", tz="UTC")
    frame = pd.DataFrame({
        "station": ["ABQ"] * len(times),
        "timestamp_utc": times,
        "observed_K": [306.0] * len(times),
        "historical_K": [306.5] * len(times),
        "source_mean_K": [307.0] * len(times),
        "baseline_K": [308.0] * len(times),
    })
    handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    handle.close()
    try:
        frame.to_csv(handle.name, index=False)
        result = evaluate(handle.name, block_days=3, draws=80)
        assert len(result["groups"]) == 1
        group = result["groups"][0]
        assert group["n_hot"] == 24
        assert abs(group["all"]["delta_source_mean_minus_historical_rmse_K"] - 0.5) < 1e-10
        assert group["ci_hot"]["ci95_low_K"] > 0
        frame.loc[1, "timestamp_utc"] = frame.loc[0, "timestamp_utc"]
        frame.to_csv(handle.name, index=False)
        try:
            evaluate(handle.name)
        except ValueError as exc:
            assert "Duplicate" in str(exc)
        else:
            raise AssertionError("Duplicate detection failed")
    finally:
        os.remove(handle.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--output", default="outputs/station_validation/paired_evaluation.json")
    parser.add_argument("--block-days", type=int, default=14)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Self-test passed")
    else:
        if not args.csv:
            parser.error("CSV required unless --self-test")
        if args.block_days < 1 or args.draws < 20:
            parser.error("block-days >= 1 and draws >= 20 required")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate(args.csv, block_days=args.block_days, draws=args.draws)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {len(result['groups'])} station-year groups to {output}")
