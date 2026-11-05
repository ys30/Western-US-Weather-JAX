"""Audit source-selected Blend weights from the three original full archives.

Verifies every archive's SHA256/size manifest record, source and target run
metadata, and the public 18-row research table. Does not reconstruct the five
validation-score candidates, which are not included in source report JSON.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

SOURCES=(
    ("Southwest 2022",41400003722,"southwest-2022"),
    ("Southwest 2023",41402185866,"southwest-2023"),
    ("Southern Plains 2022",41402189567,"southern-plains-2022"),
)
SEEDS=(42,7,123)
MODES=("historical","source_mean")
FIELDS=("case","source_run","seed","feature_mode",
        "source_selected_climatology_weight",
        "source_best_CNN_validation_RMSE_K",
        "target_CNN_RMSE_K","target_source_weight_blend_RMSE_K")

def member(z,name):
    hits=[x for x in z.namelist() if x==name or x.endswith("/"+name)]
    if len(hits)!=1:
        raise ValueError(f"Missing/ambiguous archive member: {name}")
    return hits[0]

def load_json(z,path):
    return json.loads(z.read(member(z,path)))

def audit(folder):
    rows=[]
    for label,run,stem in SOURCES:
        path=Path(folder)/(stem+"-full-archive.zip")
        with ZipFile(path) as z:
            manifest=load_json(z,"outputs/archive/sha256-manifest.json")
            records=manifest["files"]
            if manifest["count"]!=53 or len(records)!=53:
                raise ValueError(f"Unexpected archive manifest count for {label}")
            for rec in records:
                payload=z.read(member(z,rec["path"]))
                if len(payload)!=rec["bytes"] or hashlib.sha256(payload).hexdigest()!=rec["sha256"]:
                    raise ValueError(f"SHA256/length mismatch: {rec['path']}")
            for mode in MODES:
                for seed in SEEDS:
                    src=load_json(z,f"outputs/archive/{mode}/source-{seed}.json")
                    tgt=load_json(z,f"outputs/archive/{mode}/target-{seed}.json")
                    if src["random_seed"]!=seed or src["climatology_feature_mode"]!=mode:
                        raise ValueError("Source training identity mismatch")
                    if tgt["trained_climatology_feature_mode"]!=mode:
                        raise ValueError("Target model identity mismatch")
                    w=float(src["selected_blend_climatology_weight"])
                    if w not in (0.,.25,.5,.75,1.):
                        raise ValueError("Weight outside source validation grid")
                    if abs(w-float(tgt["source_validation_blend_weight"]))>1e-8:
                        raise ValueError("Target weight does not match source")
                    if "2021-01-01" not in src["selection_period"] or "2021-12-31" not in src["selection_period"]:
                        raise ValueError("Unexpected validation chronology")
                    if tgt["protocol"]!="Frozen source CNN; target pre-test climatology; NO target weight updates":
                        raise ValueError("Unexpected transfer protocol")
                    rows.append({"case":label,"source_run":run,
                        "seed":seed,"feature_mode":mode,
                        "source_selected_climatology_weight":w,
                        "source_best_CNN_validation_RMSE_K":float(src["best_validation_rmse_kelvin"]),
                        "target_CNN_RMSE_K":float(tgt["test_rmse_kelvin"]),
                        "target_source_weight_blend_RMSE_K":float(tgt["source_weight_blend_rmse_kelvin"])})
    if len(rows)!=18 or len({(r["case"],r["seed"],r["feature_mode"]) for r in rows})!=18:
        raise ValueError("Expected exact 18-row archive case/seed/arm coverage")
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected",type=Path)
    args=p.parse_args()
    rows=audit(args.archive_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if args.expected:
        with args.expected.open() as f:
            expected=list(csv.DictReader(f))
        actual={(r["case"],r["seed"],r["feature_mode"]):r for r in rows}
        if len(expected)!=18:
            raise ValueError("Incorrect number of expected rows")
        for row in expected:
            key=row["case"],int(row["seed"]),row["feature_mode"]
            if key not in actual:
                raise ValueError("Missing public record "+str(key))
            rec=actual[key]
            for field in FIELDS:
                if field in ("case","feature_mode"):
                    if str(rec[field])!=row[field]: raise ValueError((key,field))
                else:
                    if abs(float(rec[field])-float(row[field]))>1e-8: raise ValueError((key,field))
    print("Verified 159 original manifest digests and all 18 archived source-selected Blend reports.")
    print("Chosen source climatology weights:",sorted({r["source_selected_climatology_weight"] for r in rows}))
    print("Note: CNN best_validation_rmse is NOT the Blend weight-candidate validation RMSE vector.")

if __name__=="__main__":
    main()
