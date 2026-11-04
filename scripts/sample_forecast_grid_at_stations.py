"""Spatially sample a target-valid-time NetCDF prediction cube at ASOS stations.

Required NetCDF: time, latitude, longitude coordinates and variables
historical_K, source_mean_K, baseline_K, each with dimensions
(time, latitude, longitude). Time MUST be the forecast *target valid time*.
Required station CSV: station, station_lat, station_lon.
Outputs the exact prediction schema of match_asos_prediction_times.py.

This is spatial sampling, not external validation and not lapse-rate correction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARS = ("historical_K", "source_mean_K", "baseline_K")
COORDS = ("time", "latitude", "longitude")


def station_table(path):
    st = pd.read_csv(path)
    missing = sorted(set(("station", "station_lat", "station_lon")) - set(st))
    if missing:
        raise ValueError(f"Station table missing {missing}")
    st = st[["station", "station_lat", "station_lon"]].copy()
    st["station"] = st.station.astype(str).str.upper().str.strip()
    if st.station.eq("").any() or st.station.eq("NAN").any():
        raise ValueError("Blank station ID")
    st[["station_lat", "station_lon"]] = st[["station_lat", "station_lon"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(st[["station_lat", "station_lon"]].to_numpy()).all():
        raise ValueError("Nonfinite station coordinates")
    if not st.station_lat.between(-90, 90).all() or not st.station_lon.between(-180, 180).all():
        raise ValueError("Station latitude/longitude outside conventional bounds")
    # Historical ASOS CSVs repeat station coordinates across timestamps;
    # permit exact duplicates but reject inconsistent coordinates.
    if st.groupby("station")[["station_lat", "station_lon"]].nunique().gt(1).any().any():
        raise ValueError("Inconsistent station coordinates")
    return st.drop_duplicates("station").sort_values("station").reset_index(drop=True)


def sample(dataset, stations, method="linear"):
    if method not in ("linear", "nearest"):
        raise ValueError("method must be linear or nearest")
    for coord in COORDS:
        if coord not in dataset.coords:
            raise ValueError(f"Missing coordinate: {coord}")
    for name in VARS:
        if name not in dataset:
            raise ValueError(f"Missing prediction variable: {name}")
        if set(dataset[name].dims) != set(COORDS) or len(dataset[name].dims) != 3:
            raise ValueError(f"Unexpected dims for {name}: {dataset[name].dims}")
        if dataset[name].attrs.get("units", "K") not in ("K", "kelvin", "Kelvin"):
            raise ValueError(f"Prediction variable {name} must be Kelvin")
    if dataset.sizes["time"] < 1 or dataset.sizes["latitude"] < 2 or dataset.sizes["longitude"] < 2:
        raise ValueError("Insufficient time or spatial grid")
    ds = dataset.sortby("latitude").sortby("longitude")
    lat = ds.latitude.values.astype(float)
    lon = ds.longitude.values.astype(float)
    if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("Grid coordinates must be strictly monotonic and unique")
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("Forecast time must use datetime64 target-valid times")
    if pd.Index(ds.time.values).has_duplicates:
        raise ValueError("Duplicate target-valid times")
    # Decide longitude convention from gridded coordinates, not longitude.max alone.
    uses_360 = bool(np.all(lon >= 0) and np.any(lon > 180))
    records = []
    for st in stations.itertuples(index=False):
        qlon = float(st.station_lon) % 360 if uses_360 else float(st.station_lon)
        qlat = float(st.station_lat)
        if method == "linear" and not (lat[0] <= qlat <= lat[-1] and lon[0] <= qlon <= lon[-1]):
            raise ValueError(f"{st.station} outside grid interpolation extent")
        interp = ds[list(VARS)].interp(latitude=qlat, longitude=qlon, method=method)
        values = [np.asarray(interp[name].values, dtype=float) for name in VARS]
        if not all(v.shape == (ds.sizes["time"],) and np.isfinite(v).all() for v in values):
            raise ValueError(f"Missing or nonfinite interpolated forecasts at {st.station}")
        for i, ts in enumerate(pd.to_datetime(ds.time.values, utc=True)):
            records.append({
                "station": st.station,
                "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                **{name: float(values[j][i]) for j, name in enumerate(VARS)},
            })
    return pd.DataFrame.from_records(records)


def self_test():
    lat = np.array([36.0, 35.0])  # descending latitude
    lon = np.array([253.0, 254.0])  # 0..360 longitude
    times = pd.date_range("2022-01-01", periods=2, freq="6h")
    # f(lat,lon) = 2*lat + 3*lon; linear interpolation exact at midpoint.
    field = np.array([[2 * a + 3 * b for b in lon] for a in lat])
    ds = xr.Dataset(
        {name: (("time", "latitude", "longitude"), np.stack([field + i + k for i in range(2)]))
         for k, name in enumerate(VARS)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    st = pd.DataFrame({"station": ["ABQ"], "station_lat": [35.5], "station_lon": [-106.5]})
    result = sample(ds, st)
    assert len(result) == 2
    expected = 2 * 35.5 + 3 * 253.5
    assert np.isclose(result.loc[0, "historical_K"], expected)
    assert np.isclose(result.loc[1, "source_mean_K"], expected + 2)
    try:
        sample(ds, pd.DataFrame({"station": ["OUT"], "station_lat": [30.0], "station_lon": [-106.5]}))
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Out-of-domain bilinear sampling should fail")
    print("Spatial interpolation synthetic QA passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netcdf")
    parser.add_argument("--stations")
    parser.add_argument("--output", default="outputs/station_validation/station_predictions.csv")
    parser.add_argument("--method", choices=("linear", "nearest"), default="linear")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.netcdf or not args.stations:
        parser.error("--netcdf and --stations are required")
    stations = station_table(args.stations)
    with xr.open_dataset(args.netcdf) as ds:
        sampled = sample(ds, stations, method=args.method)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(dest, index=False)
    meta = {
        "method": args.method, "station_count": len(stations),
        "rows": len(sampled), "forecast_time": "target valid UTC time",
        "source_netcdf_sha256": hashlib.sha256(Path(args.netcdf).read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "warning": "No altitude correction; generated from already-trained forecast NetCDF fields",
    }
    dest.with_suffix(".spatial-audit.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
