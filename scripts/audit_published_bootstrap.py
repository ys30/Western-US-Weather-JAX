"""Audit published block-bootstrap CSV without claiming inferential independence.

Usage: python scripts/audit_published_bootstrap.py results/bootstrap_block_length_sensitivity_v2.4.csv
"""
import argparse
import csv
import math
from pathlib import Path

SEEDS = {7, 42, 123}
THRESHOLDS = {303, 305, 307}
BLOCKS = {3, 7, 14, 28}
COUNTS = {303: 100982, 305: 63954, 307: 33747}

def audit(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36:
        raise ValueError("Expected exactly 36 comparisons")
    keys = set()
    points = {}
    for row in rows:
        seed = int(row["seed"])
        threshold = int(float(row["threshold_kelvin"]))
        block = int(row["block_days"])
        key = (seed, threshold, block)
        if key in keys or seed not in SEEDS or threshold not in THRESHOLDS or block not in BLOCKS:
            raise ValueError(f"Duplicate or unexpected comparison: {key}")
        keys.add(key)
        if int(row["hot_gridcell_times"]) != COUNTS[threshold]:
            raise ValueError(f"Sample count mismatch: {key}")
        if int(row["draws"]) != 2000:
            raise ValueError(f"Unexpected bootstrap draws: {key}")
        p = float(row["point_delta_k"])
        lo = float(row["ci_low_k"])
        hi = float(row["ci_high_k"])
        if not all(math.isfinite(v) for v in (p, lo, hi)) or not lo <= hi:
            raise ValueError(f"Invalid interval: {key}")
        if (lo <= 0 <= hi) == (row["ci_excludes_zero"].strip().lower() == "true"):
            raise ValueError(f"Incorrect zero-exclusion flag: {key}")
        if not lo <= p <= hi:
            raise ValueError(f"Point outside percentile CI: {key}")
        pair = (seed, threshold)
        if pair in points and not math.isclose(points[pair], p, abs_tol=1e-10):
            raise ValueError(f"Point estimate changed with block size: {pair}")
        points[pair] = p
    expected = {(seed,threshold,block) for seed in SEEDS for threshold in THRESHOLDS for block in BLOCKS}
    if keys != expected:
        raise ValueError("Incomplete comparison matrix")
    return {"comparisons": len(rows),
            "negative_interval_upper_bounds": sum(float(r["ci_high_k"]) < 0 for r in rows),
            "all_comparisons_present": True}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    print(audit(parser.parse_args().csv_path))
