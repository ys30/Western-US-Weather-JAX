"""Independently cross-check 9 frozen-source ERA5 observed-hot points and CIs."""
import argparse,csv,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
CASES=(("Southwest 2022","southwest-2022"),("Southwest 2023","southwest-2023"),("Southern Plains 2022","southern-plains-2022"))
SEEDS=(42,7,123); NS="{http://www.w3.org/2000/svg}"
def readarchive(folder,filename):
    z=ZipFile(folder/(filename+"-full-archive.zip"))
    names=set(z.namelist())
    def read(n):
        if n not in names:raise ValueError("Missing archive member "+n)
        return json.loads(z.read(n))
    manifest=read("outputs/archive/sha256-manifest.json")
    assert len(manifest["files"])==manifest["count"]==53
    for rec in manifest["files"]:
        data=z.read(rec["path"])
        if hashlib.sha256(data).hexdigest()!=rec["sha256"] or len(data)!=rec["bytes"]:
            raise ValueError("Failed digest: "+rec["path"])
    return z,read
def archive_rows(folder):
    rows=[]
    for case,stem in CASES:
        zipf,load=readarchive(folder,stem)
        for seed in SEEDS:
            h=load(f"outputs/archive/historical/target-{seed}.json")
            s=load(f"outputs/archive/source_mean/target-{seed}.json")
            direct=load(f"outputs/archive/direct-seed-{seed}-block-14.json")
            hv=float(h["fixed_305k_cnn_rmse_kelvin"]);sv=float(s["fixed_305k_cnn_rmse_kelvin"])
            obs=direct["results"]["observed_hot"]
            v=float(obs["source_mean_minus_historical_rmse_K"])
            l=float(obs["ci95_lower_K"]);u=float(obs["ci95_upper_K"])
            if abs((sv-hv)-v)>1e-6:raise ValueError(f"Difference mismatch: {case}/{seed}")
            if not l<=v<=u or obs["valid_draws"]!=2000 or direct["circular_block_days"]!=14:
                raise ValueError("Unexpected bootstrap")
            if h["fixed_hot_samples"]!=s["fixed_hot_samples"] or h["fixed_hot_samples"]!=obs["sample_count"]:
                raise ValueError("Inconsistent verifying hot cells")
            rows.append(dict(case=case,seed=seed,observed_hot_ERA5_cells=obs["sample_count"],
                historical_rmse_K=hv,source_mean_rmse_K=sv,source_mean_minus_historical_rmse_K=v,
                ci95_lower_K=l,ci95_upper_K=u,block_days=14,draws=2000))
        zipf.close()
    if len(rows)!=9:raise ValueError("Expected nine pairs")
    return rows
def pos(path,tag):
    root=ET.parse(path).getroot()
    return [e for e in root.iter() if e.tag==NS+tag]
def audit_plots(rows,folder):
    p1=pos(folder/"manuscript_bootstrap_public.svg","circle")
    p2=pos(folder/"paper_bootstrap_14day.svg","circle")
    lines=pos(folder/"paper_bootstrap_14day.svg","line")
    p3=pos(folder/"manuscript_performance_public.svg","circle")
    rects=[e for e in pos(folder/"manuscript_performance_public.svg","rect") if e.attrib.get("height")=="10"]
    if not all(len(a)==9 for a in (p1,p2,p3,rects)):raise ValueError("Unexpected mark count")
    for i,r in enumerate(rows):
        v=r["source_mean_minus_historical_rmse_K"]
        if abs(float(p1[i].attrib["cx"])-(525+6000*v))>.40:raise ValueError(f"Public points mismatch {i}")
        cy=float(p2[i].attrib["cy"])
        candidates=[e for e in lines if e.attrib.get("stroke-width")=="3" and abs(float(e.attrib.get("y1","-100"))-cy)<.001 and abs(float(e.attrib.get("y2","-100"))-cy)<.001]
        if len(candidates)!=1:raise ValueError(f"No unique confidence whisker at {i}")
        whisker=candidates[0]
        expect=[570+6000*v,570+6000*r["ci95_lower_K"],570+6000*r["ci95_upper_K"]]
        got=[float(p2[i].attrib["cx"]),float(whisker.attrib["x1"]),float(whisker.attrib["x2"])]
        if max(abs(a-b) for a,b in zip(expect,got))>.40:raise ValueError(f"Interval plot mismatch {i}: {expect} vs {got}")
        for value,actual in [(r["historical_rmse_K"],float(p3[i].attrib["cx"])),(r["source_mean_rmse_K"],float(rects[i].attrib["x"])+5)]:
            expected=340+(value-2.0)*(620/0.65)
            if abs(expected-actual)>.55:raise ValueError(f"Hot absolute RMSE mismatch {i}: {expected} vs {actual}")
def main():
    p=argparse.ArgumentParser();p.add_argument("--archives",type=Path,required=True)
    p.add_argument("--figures",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();rows=archive_rows(a.archives);audit_plots(rows,a.figures)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    pos=sum(r["source_mean_minus_historical_rmse_K"]>0 for r in rows)
    excl=sum(r["ci95_lower_K"]>0 or r["ci95_upper_K"]<0 for r in rows)
    print("PASS: 159 archive digests, 9 direct points, 18 hot RMSE marks and 9 intervals with 18 endpoints.")
    print(f"Historical better: {pos}/9; 14-day within-seed intervals not crossing zero: {excl}/9.")
if __name__=="__main__":main()
