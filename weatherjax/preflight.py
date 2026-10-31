"""Validate small ERA5 regional input before allocating GPU resources."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_era5

def inspect_era5(path, variable="t2m", expected_step_hours=1, lead_hours=24):
    da = load_era5(path, variable=variable)
    if da.ndim != 3:
        raise ValueError("Expected a single time/lat/lon variable; select any extra level first")
    units = str(da.attrs.get("units", "")).strip()
    if units not in ("K", "kelvin"):
        raise ValueError(f"Expected temperature in Kelvin, got units={units!r}; convert before training")
    stamps = np.asarray(da.time.values).astype("datetime64[ns]")
    if len(stamps) < 40:
        raise ValueError("At least 40 timestamps required for meaningful train/val/test smoke splits")
    intervals = np.diff(stamps).astype("timedelta64[s]").astype(np.int64)
    expected = int(expected_step_hours*3600)
    if expected <= 0 or not np.all(intervals == expected):
        raise ValueError(f"Time series must be strictly increasing and regular at {expected_step_hours}h; found {np.unique(intervals).tolist()} seconds")
    if lead_hours % expected_step_hours:
        raise ValueError("Lead horizon must be an integer multiple of timestep")
    if da.sizes["lat"] < 4 or da.sizes["lon"] < 4:
        raise ValueError("At least 4x4 spatial grid required")
    # Small preliminary scan; actual training reads all selected values into host memory.
    sample = np.asarray(da.isel(time=slice(0, min(8, da.sizes["time"]))).values)
    if not np.isfinite(sample).all():
        raise ValueError("Non-finite temperatures in input sample")
    if sample.min() < 150 or sample.max() > 350:
        raise ValueError("Unreasonable temperature range for Kelvin 2m air temperature")
    return {"input":str(path), "variable":variable, "units":"K",
            "steps":int(da.sizes["time"]), "height":int(da.sizes["lat"]), "width":int(da.sizes["lon"]),
            "timestep_hours":expected_step_hours, "lead_hours":lead_hours,
            "lead_steps":lead_hours//expected_step_hours,
            "start":str(stamps[0]), "end":str(stamps[-1]),
            "estimated_float32_field_megabytes":round(4*da.size/1e6,2),
            "validation":"metadata and first 8 frames; full missing-value check occurs at training"}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--variable",default="t2m")
    p.add_argument("--timestep-hours",type=int,default=1)
    p.add_argument("--lead-hours",type=int,default=24)
    p.add_argument("--report")
    a=p.parse_args()
    result=inspect_era5(a.path,a.variable,a.timestep_hours,a.lead_hours)
    value=json.dumps(result,indent=2)
    print(value)
    if a.report:
        Path(a.report).parent.mkdir(parents=True,exist_ok=True)
        Path(a.report).write_text(value)
if __name__=="__main__":main()
