"""Hash reproducibility inputs, checkpoints, and result files.

Usage:
    python scripts/hash_experiment_files.py --root input --root outputs --output outputs/sha256_manifest.json

Paths are relative to the current working directory; files are streamed rather
than loaded into RAM. The output manifest is excluded from its own hash list.
"""
import argparse
import hashlib
import json
from pathlib import Path


def manifest(roots, output):
    dest = Path(output).resolve()
    files = set()
    for root in roots:
        p = Path(root)
        if not p.exists():
            raise FileNotFoundError(f"Missing hash root: {p}")
        files.update(x.resolve() for x in (p.rglob("*") if p.is_dir() else [p])
                     if x.is_file() and x.resolve() != dest)
    records = []
    for path in sorted(files):
        h = hashlib.sha256()
        with path.open("rb") as fd:
            for chunk in iter(lambda: fd.read(1024 * 1024), b""):
                h.update(chunk)
        records.append({"path": str(path.relative_to(Path.cwd().resolve()))
                        if path.is_relative_to(Path.cwd().resolve()) else str(path),
                        "bytes": path.stat().st_size, "sha256": h.hexdigest()})
    return {"algorithm": "SHA256", "files": records, "count": len(records)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", action="append", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data = manifest(a.root, a.output)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {data['count']} checksums to {output}")


if __name__ == "__main__":
    main()
