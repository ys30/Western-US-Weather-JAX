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
import time

STATIONS = [
    {"station":"ABQ","network":"NM_ASOS","region":"southwest","name":"Albuquerque International Sunport","lat":35.0402,"lon":-106.6092},
    {"station":"ROW","network":"NM_ASOS","region":"southwest","name":"Roswell Air Center","lat":33.2998775,"lon":-104.5294031},
    {"station":"ABI","network":"TX_ASOS","region":"plains","name":"Abilene Regional Airport","lat":32.4113333,"lon":-99.6818889},
    {"station":"OKC","network":"OK_ASOS","region":"plains","name":"Will Rogers World Airport","lat":35.3931,"lon":-97.6007},
]
ENDPOINT="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def parse_observations(raw,station_code,year):
    """Explicit IEM row-level QA before any report-time deduplication."""
    source=pd.read_csv(io.BytesIO(raw),comment="#",low_memory=False)
    required={"station","valid","tmpf"}
    if not required.issubset(source.columns):
        raise ValueError(f"Unexpected IEM columns: {source.columns.tolist()}")
    qa={"raw_rows":int(len(source))}
    source["station"]=source["station"].astype(str).str.upper().str.strip()
    source=source.loc[source["station"]==station_code.upper()].copy()
    qa["matching_station_rows"]=int(len(source))
    source["valid"]=pd.to_datetime(source["valid"],utc=True,errors="coerce")
    source["tmpf"]=pd.to_numeric(source["tmpf"],errors="coerce")
    qa["missing_time_rows"]=int(source["valid"].isna().sum())
    qa["missing_temperature_rows"]=int(source["tmpf"].isna().sum())
    source=source.dropna(subset=["valid","tmpf"])
    start=pd.Timestamp(f"{year}-01-01",tz="UTC")
    stop=pd.Timestamp(f"{year+1}-01-01",tz="UTC")
    qa["outside_year_rows"]=int((~source["valid"].between(start,stop,inclusive="left")).sum())
    source=source.loc[(source["valid"]>=start)&(source["valid"]<stop)]
    qa["outside_conservative_fahrenheit_range_rows"]=int((~source["tmpf"].between(-40,130)).sum())
    source=source.loc[source["tmpf"].between(-40,130)]
    qa["duplicate_report_times"]=int(source["valid"].duplicated().sum())
    source=source.groupby("valid",as_index=False)["tmpf"].median().sort_values("valid")
    qa["retained_unique_reports"]=int(len(source))
    return source,qa

def self_test():
    raw=("station,valid,tmpf\\n"+
         "ABQ,2022-01-01 00:00,32\\n"+
         "ABQ,2022-01-01 00:00,34\\n"+
         "ABQ,2022-01-01 06:00,M\\n"+
         "ABQ,2023-01-01 00:00,45\\n"+
         "ROW,2022-01-01 00:00,41\\n"+
         "ABQ,2022-01-01 12:00,140\\n").replace("\\\\n","\\n").encode()
    data,qa=parse_observations(raw,"ABQ",2022)
    assert len(data)==1 and float(data.iloc[0]["tmpf"])==33
    assert qa["duplicate_report_times"]==1
    assert qa["missing_temperature_rows"]==1
    assert qa["outside_year_rows"]==1
    assert qa["outside_conservative_fahrenheit_range_rows"]==1
    print("ASOS source parsing, duplicate and exclusion audit passed")

def fetch(station,year,output):
    args={
        "network":station["network"],"station":station["station"],
        "data":"tmpf","year1":year,"month1":1,"day1":1,
        "year2":year+1,"month2":1,"day2":1,
        "tz":"Etc/UTC","format":"onlycomma","latlon":"no","elev":"no",
        "missing":"M","trace":"T","direct":"no","report_type":[3,4],
    }
    raw=None
    for attempt in range(5):
        r=requests.get(ENDPOINT,params=args,timeout=180)
        if r.status_code==429 or r.status_code in (502,503,504):
            if attempt==4:
                r.raise_for_status()
            retry=r.headers.get("Retry-After","")
            try:
                pause=max(15,min(180,float(retry)))
            except ValueError:
                pause=min(180,25*(2**attempt))
            print(f"IEM returned {r.status_code}; polite retry in {pause:.0f}s",flush=True)
            time.sleep(pause)
            continue
        r.raise_for_status()
        raw=r.content
        break
    if raw is None:
        raise RuntimeError("No station response after retries")
    if len(raw)<200 or b"<html" in raw[:500].lower():
        raise RuntimeError(f"No CSV returned from IEM for {station['station']} {year}: {raw[:150]!r}")
    frame,qa=parse_observations(raw,station["station"],year)
    if len(frame)<1000:
        raise ValueError(f"Station data insufficient: {station['station']} {year} {len(frame)} reports; QA={qa}")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["station","region","year","timestamp_utc","temperature_K","station_lat","station_lon","source"])
        for _,rec in frame.iterrows():
            w.writerow([station["station"],station["region"],year,rec["valid"].strftime("%Y-%m-%dT%H:%M:%SZ"),round((rec["tmpf"]-32)*5/9+273.15,4),station["lat"],station["lon"],"IEM_ASOS_METAR"])
    return {"station":station["station"],"year":year,"reports":len(frame),"qa":qa,
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"bytes":output.stat().st_size}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="outputs/station_validation")
    a=p.parse_args();dest=Path(a.output_dir)
    meta=[]
    for st in STATIONS:
        for year in (2022,2023):
            out=dest/f"{st['station']}-{year}.csv"
            info=fetch(st,year,out);meta.append(info);print(info,flush=True)
            # One request every 15 seconds at most; IEM applies public-service rate limits.
            time.sleep(15)
    (dest/"station_manifest.json").write_text(json.dumps({
        "source":"Iowa Environmental Mesonet (ASOS/METAR as-is archive)",
        "endpoint":ENDPOINT,"stations":STATIONS,"files":meta,
        "qa_notes":"IEM reports have limited QC; no station selected by model skill; 6-hour nearest-time matching is deferred; station observation is not necessarily independent of ERA5 assimilation.",
    },indent=2))
if __name__=="__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
