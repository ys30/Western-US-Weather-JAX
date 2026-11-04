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
def parse_station_height(html, site):
    """Require explicit Elevation [m] table header, not a fixed table index."""
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,"html.parser")
    for tab in soup.select("table"):
        rows=tab.select("tr")
        headers=[]
        for row in rows:
            cells=row.select("th")
            if cells:
                headers=[c.get_text(" ",strip=True).lower().replace(" ","") for c in cells]
                break
        if not headers:
            continue
        idx_id=next((i for i,h in enumerate(headers) if h=="id"),None)
        idx_elev=next((i for i,h in enumerate(headers)
                       if h in ("elevation[m]","elevation(m)","elevationmeters")),None)
        if idx_id is None or idx_elev is None:
            continue
        for row in rows:
            cells=[x.get_text(" ",strip=True) for x in row.select("td")]
            if len(cells)>max(idx_id,idx_elev) and cells[idx_id].upper()==site.upper():
                value=float(cells[idx_elev].replace(",",""))
                if not np.isfinite(value):
                    raise ValueError("Nonfinite IEM station elevation")
                return value,cells
    raise RuntimeError(f"Station {site} not found in a labeled elevation-meters table")


def station_meters(net,site):
    url=f"https://mesonet.agron.iastate.edu/sites/networks.php?network={net}"
    r=requests.get(url,timeout=90)
    r.raise_for_status()
    height,row=parse_station_height(r.text,site)
    return height,url,row

def self_test():
    html="""<table><tr><th>ID</th><th>Station Name</th><th>Latitude</th><th>Longitude</th><th>Elevation [m]</th></tr>
    <tr><td>ABQ</td><td>Albuquerque</td><td>35.04</td><td>-106.61</td><td>1631.4</td></tr></table>"""
    val,_=parse_station_height(html,"ABQ")
    assert abs(val-1631.4)<1e-9
    try:
        parse_station_height(html.replace("Elevation [m]","Elevation [ft]"),"ABQ")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unlabeled elevation units should fail")
    print("IEM elevation header/unit parsing regression passed")

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
if __name__=="__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
