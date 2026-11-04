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


def daily_block_interval(frame, days=14, draws=1000, seed=20261103, hot_threshold=None):
    """Paired circular calendar-day blocks using sufficient error statistics.

    Draw calendar days *including dates with no qualifying hot reports*. For
    each draw, calculate both model RMSEs from sum-of-squared-errors and count.
    This is algebraically identical to row concatenation, without copying
    thousands of observation frames per bootstrap draw.
    """
    if days < 1 or draws < 20:
        raise ValueError("Block length must be >=1 and draws >=20")
    if frame.empty:
        return None
    frame = frame.sort_values("timestamp_utc")
    calendar_days = frame["timestamp_utc"].dt.floor("D")
    first, last = calendar_days.iloc[0], calendar_days.iloc[-1]
    full_days = pd.date_range(first, last, freq="D")
    n = len(full_days)
    if n < 2:
        return None
    selected = (np.ones(len(frame), dtype=bool) if hot_threshold is None
                else frame["observed_K"].to_numpy() >= hot_threshold)
    if not selected.any():
        return None
    # Preserve missing calendar days as all-zero rows, never compress hot days.
    obs = frame.loc[selected, "observed_K"].to_numpy(dtype=np.float64)
    historical = frame.loc[selected, "historical_K"].to_numpy(dtype=np.float64)
    source = frame.loc[selected, "source_mean_K"].to_numpy(dtype=np.float64)
    day_index = ((calendar_days[selected] - first) / pd.Timedelta(days=1)).to_numpy(dtype=int)
    counts = np.bincount(day_index, minlength=n).astype(np.float64)
    historical_sse = np.bincount(day_index, weights=(historical-obs)**2, minlength=n)
    source_sse = np.bincount(day_index, weights=(source-obs)**2, minlength=n)
    stats = np.column_stack((counts, historical_sse, source_sse))
    rng = np.random.default_rng(seed)
    values = []
    empty_draws = 0
    blocks = (n + days - 1) // days
    offsets = np.arange(days, dtype=int)
    for _ in range(draws):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        total_count, total_h, total_s = stats[indices].sum(axis=0)
        if total_count < 1:
            empty_draws += 1
            continue
        values.append(float(np.sqrt(total_s / total_count) -
                            np.sqrt(total_h / total_count)))
    if len(values) < max(10, int(0.8 * draws)):
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return {"block_days": days, "draws": len(values),
            "empty_draws": empty_draws,
            "calendar_days": n,
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
        hot_days = int(hot["timestamp_utc"].dt.floor("D").nunique())
        hot_ci_eligible = len(hot) >= 30 and hot_days >= 10
        groups.append({
            "station": str(site), "year": int(year),
            "all": metrics(group),
            "observed_hot": metrics(hot),
            "n_days": int(group["timestamp_utc"].dt.floor("D").nunique()),
            "n_hot": int(len(hot)),
            "n_hot_days": hot_days,
            "hot_ci_eligible": bool(hot_ci_eligible),
            "ci_all": daily_block_interval(group, block_days, draws),
            "ci_hot": daily_block_interval(group, block_days, draws, hot_threshold=threshold) if hot_ci_eligible else None,
        })
    return {
        "schema": "station-paired-evaluation-v1",
        "threshold_observed_K": threshold,
        "hot_ci_minimum": {"observations": 30, "distinct_calendar_days": 10},
        "comparison_sign": "source_mean minus historical: positive favors historical",
        "interpretation": "Retrospective, exploratory, station-time matched; not an operational forecast or independent ERA5 validation",
        "groups": groups,
        "pooled_descriptive_only": metrics(data),
    }


def self_test():
    import tempfile
    import os
    times = pd.date_range("2022-07-01", periods=60, freq="6h", tz="UTC")
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
        assert group["n_hot"] == 60
        assert abs(group["all"]["delta_source_mean_minus_historical_rmse_K"] - 0.5) < 1e-10
        assert group["ci_hot"]["ci95_low_K"] > 0
        # A rare hot day must be sampled from the full calendar, not as
        # though every day were hot. Empty hot draws are explicitly tracked.
        frame["observed_K"] = [306.0] * 4 + [300.0] * (len(frame) - 4)
        frame.to_csv(handle.name, index=False)
        rare = evaluate(handle.name, block_days=1, draws=100)
        ci = rare["groups"][0]["ci_hot"]
        assert ci is None or ci["empty_draws"] > 0
        assert rare["groups"][0]["n_hot"] == 4
        assert rare["groups"][0]["ci_hot"] is None
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
