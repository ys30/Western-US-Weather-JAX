"""Verify a full ERA5 GitHub Actions ZIP against its embedded SHA256 manifest."""
import argparse
import hashlib
import json
from zipfile import ZipFile

def verify_archive(path):
    with ZipFile(path) as archive:
        names=set(archive.namelist())
        manifest_name="outputs/archive/sha256-manifest.json"
        if manifest_name not in names:
            raise ValueError("Missing SHA256 manifest")
        doc=json.loads(archive.read(manifest_name))
        records=doc["files"]
        if len(records)!=doc["count"]:
            raise ValueError("Manifest count mismatch")
        mismatches=[]
        for rec in records:
            name=rec["path"]
            if name not in names:
                mismatches.append(f"missing: {name}")
                continue
            if len(archive.read(name))!=rec["bytes"]:
                mismatches.append(f"size: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest()!=rec["sha256"]:
                mismatches.append(f"sha256: {name}")
        if mismatches:
            raise ValueError("; ".join(mismatches))
        required=["input/source.nc","input/target.nc","resolved-requirements.txt","python-version.txt","git-commit.txt"]
        for name in required:
            if name not in names:
                raise ValueError(f"Missing required file: {name}")
        checkpoints=[n for n in names if n.startswith("outputs/archive/") and n.endswith(".npz") and "/seed-" in n]
        predictions=[n for n in names if n.startswith("outputs/archive/") and n.endswith(".npz") and "/fields-" in n]
        if len(checkpoints)!=6 or len(predictions)!=6:
            raise ValueError("Expected six checkpoints and six prediction archives")
        return {"files":len(names),"manifest_records":len(records),"checkpoints":len(checkpoints),"predictions":len(predictions),"verified":True}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive")
    args=p.parse_args()
    print(json.dumps(verify_archive(args.archive),indent=2))
