"""Audit ERA5 native-grid surface geopotential and IEM ASOS station elevation.
This is a source-data acquisition and QA step, not a temperature correction.
"""
import json
import pathlib
import hashlib
import requests
import numpy as np
import pandas as pd
import xarray as xr

OUT=pathlib.Path("outputs/elevation_audit")
OUT.mkdir(parents=True,exist_ok=True)
ZARR="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SITES=[
    ("ABQ","NM_ASOS",35.0402,-106.6092),
    ("ROW","NM_ASOS",33.29988,-104.5294),
    ("ABI","TX_ASOS",32.41133,-99.68189),
    ("OKC","OK_ASOS",35.3931,-97.6007),
]
def station_meters(net,site):
    url=f"https://mesonet.agron.iastate.edu/sites/networks.php?network={net}"
    r=requests.get(url,timeout=90)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(r.text,"html.parser")
    tables=soup.select("table")
    for tab in tables:
        rows=tab.select("tr")
        for tr in rows:
            cols=[x.get_text(" ",strip=True) for x in tr.select("td")]
            if cols and cols[0].strip()==site and len(cols)>=5:
                return float(cols[4].replace(",","")),url,cols
    raise RuntimeError(f"IEM elevation not found: {net}/{site} in {len(tables)} tables")
def main():
    # Same regular 0.25 degree regridded data product as existing temperature input.
    ds=xr.open_zarr(ZARR,chunks=None,storage_options={"token":"anon"})
    key="geopotential_at_surface"
    if key not in ds:
        raise ValueError(f"Surface geopotential absent from source; candidate variables: {[x for x in ds if 'geopotential' in x]}")
    da=ds[key]
    if "time" in da.dims:
        da=da.sel(time="2022-01-01T00:00:00")
    results=[]
    for site,network,lat,lon in SITES:
        site_h,source_url,metadata=station_meters(network,site)
        longitude=(lon%360) if da.longitude.max()>180 else lon
        near=da.sel(latitude=lat,longitude=longitude,method="nearest").compute()
        surf_height=float(near.values)/9.80665
        bilinear=da.interp(latitude=lat,longitude=longitude).compute()
        bilin_height=float(bilinear.values)/9.80665
        row={"station":site,"network":network,"station_elevation_m":site_h,
             "era5_nearest_elevation_m":surf_height,"era5_bilinear_elevation_m":bilin_height,
             "nearest_station_minus_grid_m":site_h-surf_height,
             "bilinear_station_minus_grid_m":site_h-bilin_height,
             "station_lat":lat,"station_lon":lon,
             "native_gridded_geopotential_units":str(da.attrs.get("units","")),
             "era5_source":ZARR,"iem_station_metadata_source":source_url,
             "iem_station_metadata_row":metadata}
        if not all(np.isfinite([site_h,surf_height,bilin_height])): raise ValueError(row)
        if not 0<=site_h<=4000 or not -500<=surf_height<=5000: raise ValueError(row)
        results.append(row)
        print(json.dumps(row),flush=True)
    path=OUT/"station_grid_elevation.json"
    path.write_text(json.dumps(results,indent=2))
    (OUT/"provenance.json").write_text(json.dumps({
        "source":ZARR,"variable":key,"time":"2022-01-01T00:00:00",
        "geopotential_to_height":"g0=9.80665 m s^-2",
        "station_metadata":"IEM network station tables",
        "note":"Station elevations may not coincide exactly with ASOS sensor height; no lapse-rate adjustment has been applied",
        "elevation_sha256":hashlib.sha256(path.read_bytes()).hexdigest()
    },indent=2))
if __name__=="__main__": main()
