"""Check public six-station × three-seed annual ASOS figure against real CSV.

Uses static SVG geometry as an independent test that every visual circle and
95% CI whisker corresponds to the public data. No image inference or new
model performance estimate is involved.
"""
from pathlib import Path
import argparse
import csv
import xml.etree.ElementTree as ET

NS="{http://www.w3.org/2000/svg}"
X_ZERO=610.0
PIXELS_PER_K=8800.0
CASES=(("SW22","ABQ"),("SW22","ROW"),("SW23","ABQ"),("SW23","ROW"),("SP22","ABI"),("SP22","OKC"))
SEEDS=(42,7,123)

def read_rows(sw22,others):
    result={}
    for case,path in (("SW22",sw22),("OTHER",others)):
        with open(path,newline="") as h:
            for r in csv.DictReader(h):
                name="SW22" if case=="SW22" else ("SW23" if r["case"]=="southwest-2023" else "SP22")
                k=(name,r["station"],int(r["seed"]))
                if k in result: raise ValueError(f"Duplicated case/site/seed {k}")
                if case=="SW22":
                    vals=tuple(float(r[x]) for x in ("delta_source_mean_minus_historical_rmse_K","annual_ci95_low_K","annual_ci95_high_K"))
                else:
                    vals=tuple(float(r[x]) for x in ("delta_rmse_K","ci_low_K","ci_high_K"))
                result[k]=vals
    return result

def audit(svg,sw22,others):
    expected=read_rows(sw22,others)
    if len(expected)!=18: raise ValueError(f"Wrong data count {len(expected)}")
    root=ET.parse(svg).getroot()
    children=list(root)
    circles=[(i,node) for i,node in enumerate(children) if node.tag==NS+"circle"]
    if len(circles)!=18:raise ValueError(f"Wrong circle count {len(circles)}")
    for (case,site),_ in zip(CASES,range(6)):
        for seed in SEEDS:
            n=(CASES.index((case,site))*3)+SEEDS.index(seed)
            idx,circle=circles[n]
            key=case,site,seed
            value,low,high=expected[key]
            y=float(circle.attrib["cy"])
            match=[node for node in children[max(0,idx-5):idx] if node.tag==NS+"line"
                   and abs(float(node.attrib.get("y1",-1))-y)<.01
                   and abs(float(node.attrib.get("y2",-2))-y)<.01]
            if len(match)!=1:
                raise ValueError(f"Cannot locate CI whisker for {key}: {len(match)}")
            line=match[0]
            observed=(float(circle.attrib["cx"]),float(line.attrib["x1"]),float(line.attrib["x2"]))
            calculated=(X_ZERO+PIXELS_PER_K*value,X_ZERO+PIXELS_PER_K*low,X_ZERO+PIXELS_PER_K*high)
            if max(abs(x-z) for x,z in zip(observed,calculated))>.06:
                raise ValueError(f"SVG numeric mapping differs for {key}: {observed} != {calculated}")
    positives=sum(v[0]>0 for v in expected.values())
    negatives=sum(v[0]<0 for v in expected.values())
    excludes=sum(v[1]>0 or v[2]<0 for v in expected.values())
    print(f"Passed: 18 SVG points and their 36 CI endpoints match source CSV (<0.06 px).")
    print(f"Positive/negative annual deltas: {positives}/{negatives}; nominal 14-day intervals excluding zero: {excludes}/18.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--svg",type=Path,required=True)
    p.add_argument("--sw22",type=Path,required=True)
    p.add_argument("--others",type=Path,required=True)
    args=p.parse_args()
    audit(args.svg,args.sw22,args.others)
