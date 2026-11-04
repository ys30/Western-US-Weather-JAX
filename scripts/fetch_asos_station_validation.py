"""Download archived IEM airport temperature observations for retrospective station validation.
The airport reports are not guaranteed to be independent of the ERA5 observing system.
This script never selects stations or samples based on CNN prediction errors.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import requests
import pandas as pd

STATIONS = [
    {"station":"ABQ","network":"NM_ASOS","region":"southwest","name":"Albuquerque International Sunport","lat":35.0402,"lon":-106.6092},
    {"station":"ROW","network":"NM_ASOS","region":"southwest","name":"Roswell Air Center","lat":33.2998775,"lon":-104.5294031},
    {"station":"ABI","network":"TX_ASOS","region":"plains","name":"Abilene Regional Airport","lat":32.4113333,"lon":-99.6818889},
    {"station":"OKC","network":"OK_ASOS","region":"plains","name":"Will Rogers World Airport","lat":35.3931,"lon":-97.6007},
]
ENDPOINT="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def fetch(station,year,output):
    args={
        "network":station["network"],"station":station["station"],
        "data":"tmpf","year1":year,"month1":1,"day1":1,
        "year2":year+1,"month2":1,"day2":1,
        "tz":"Etc/UTC","format":"onlycomma","latlon":"no","elev":"no",
        "missing":"M","trace":"T","direct":"no","report_type":[3,4],
    }
    r=requests.get(ENDPOINT,params=args,timeout=180)
    r.raise_for_status()
    raw=r.content
    if len(raw)<200 or b"<html" in raw[:500].lower():
        raise RuntimeError(f"No CSV returned from IEM for {station['station']} {year}: {raw[:150]!r}")
    frame=pd.read_csv(io.BytesIO(raw),comment="#",low_memory=False)
    required={"station","valid","tmpf"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected IEM columns: {frame.columns.tolist()}")
    frame["valid"]=pd.to_datetime(frame["valid"],utc=True,errors="coerce")
    frame["tmpf"]=pd.to_numeric(frame["tmpf"],errors="coerce")
    frame=frame.dropna(subset=["valid","tmpf"])
    frame=frame.loc[frame["station"].astype(str).str.upper()==station["station"]]
    frame=frame.loc[frame["tmpf"].between(-40,130)]
    # Repeated routine/special reports at an identical instant are aggregated by median.
    frame=frame.groupby("valid",as_index=False)["tmpf"].median().sort_values("valid")
    start=pd.Timestamp(f"{year}-01-01",tz="UTC")
    stop=pd.Timestamp(f"{year+1}-01-01",tz="UTC")
    frame=frame.loc[(frame["valid"]>=start)&(frame["valid"]<stop)]
    if len(frame)<1000:
        raise ValueError(f"Station data insufficient: {station['station']} {year} {len(frame)} reports")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["station","region","year","timestamp_utc","temperature_K","station_lat","station_lon","source"])
        for _,rec in frame.iterrows():
            w.writerow([station["station"],station["region"],year,rec["valid"].strftime("%Y-%m-%dT%H:%M:%SZ"),round((rec["tmpf"]-32)*5/9+273.15,4),station["lat"],station["lon"],"IEM_ASOS_METAR"])
    return {"station":station["station"],"year":year,"reports":len(frame),
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"bytes":output.stat().st_size}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="outputs/station_validation")
    a=p.parse_args();dest=Path(a.output_dir)
    meta=[]
    for st in STATIONS:
        for year in (2022,2023):
            out=dest/f"{st['station']}-{year}.csv"
            info=fetch(st,year,out);meta.append(info);print(info,flush=True)
    (dest/"station_manifest.json").write_text(json.dumps({
        "source":"Iowa Environmental Mesonet (ASOS/METAR as-is archive)",
        "endpoint":ENDPOINT,"stations":STATIONS,"files":meta,
        "qa_notes":"IEM reports have limited QC; no station selected by model skill; 6-hour nearest-time matching is deferred; station observation is not necessarily independent of ERA5 assimilation.",
    },indent=2))
if __name__=="__main__":main()
