"""Capture immutable local file digest and actual ERA5 coordinate extent."""
import argparse,hashlib,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import xarray as xr

def manifest(path):
    p=Path(path)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    with xr.open_dataset(p) as ds:
        da=ds["t2m"]
        result={"input_filename":p.name,"sha256":h.hexdigest(),"file_bytes":p.stat().st_size,
          "shape":{dim:int(da.sizes[dim]) for dim in ("time","lat","lon")},
          "latitude_min":float(da.lat.min()),"latitude_max":float(da.lat.max()),
          "longitude_min":float(da.lon.min()),"longitude_max":float(da.lon.max()),
          "time_first":str(da.time.values[0]),"time_last":str(da.time.values[-1]),
          "variable":"t2m","units":da.attrs.get("units"),
          "python_version":sys.version.split()[0],"platform":platform.platform(),
          "captured_at_utc":datetime.now(timezone.utc).isoformat()}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("file");ap.add_argument("--output",required=True)
    a=ap.parse_args()
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(manifest(a.file),indent=2)+"\n")
if __name__=="__main__":main()
