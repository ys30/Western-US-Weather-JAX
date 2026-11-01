"""Fetch a bounded REAL ERA5 2m-temperature slice from public Google ARCO Zarr.

Do not run global downloads. Requires network and may incur egress costs.
"""
import argparse
from pathlib import Path
import xarray as xr
import numpy as np

URL="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

def fetch(output, start="2021-07-01", end="2021-07-07", stride=6,
          lat_min=31,lat_max=43,lon_min=-121,lon_max=-109):
    ds=xr.open_zarr(URL,chunks=None,storage_options={"token":"anon"})
    names=["2m_temperature","t2m","temperature_2m"]
    field=next((v for v in names if v in ds.data_vars),None)
    if field is None:
        raise ValueError("2m temperature variable not found. Available: "+", ".join(ds.data_vars))
    da=ds[field]
    rename={k:v for k,v in {"latitude":"lat","longitude":"lon","valid_time":"time"}.items() if k in da.dims}
    da=da.rename(rename)
    if da.lon.max()>180: da=da.assign_coords(lon=((da.lon+180)%360)-180).sortby("lon")
    da=da.sortby("lat").sel(time=slice(start,end),lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    da=da.isel(time=slice(None,None,stride)).transpose("time","lat","lon")
    if da.sizes["time"]<40: raise ValueError("Need >=40 timestamps; request a wider date range")
    da=da.load()
    if not np.isfinite(da.values).all(): raise ValueError("ERA5 has missing values")
    da.attrs["units"]="K" # ECMWF ERA5 t2m has Kelvin units
    da.name="t2m"
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    da.to_dataset().to_netcdf(output)
    print(f"Saved {output}, shape={da.shape}, UTC start/end={da.time.values[0]}/{da.time.values[-1]}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="input/era5_west_2021_07.nc")
    p.add_argument("--start",default="2021-07-01")
    p.add_argument("--end",default="2021-07-14")
    p.add_argument("--stride",type=int,default=6)
    p.add_argument("--lat-min",type=float,default=31)
    p.add_argument("--lat-max",type=float,default=43)
    p.add_argument("--lon-min",type=float,default=-121)
    p.add_argument("--lon-max",type=float,default=-109)
    a=p.parse_args()
    if not (a.lat_min<a.lat_max and a.lon_min<a.lon_max): raise ValueError("Invalid geographic bounds")
    if a.stride<=0: raise ValueError("stride must be positive")
    fetch(a.output,a.start,a.end,a.stride,a.lat_min,a.lat_max,a.lon_min,a.lon_max)
