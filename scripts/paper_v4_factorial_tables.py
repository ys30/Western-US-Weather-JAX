"""Rebuild paper-v4 factorial absolute-RMSE tables from archived target JSONs.

Usage: python scripts/paper_v4_factorial_tables.py --root /path/to/merged-artifacts --out results/paper-v4
Each seed archive may be extracted beneath an arbitrary parent directory. Original
relative paths outputs/factorial/{southwest,plains}-{2022,2023}/{blend,persistence}/
{historical,source_mean}/target-{seed}.json must be retained.
This script does not retrain models and does not substitute manuscript values.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

SEEDS = (7, 11, 19, 29, 37, 42, 53, 71, 97, 123)
CASES = ("southwest-2022", "southwest-2023", "plains-2022", "plains-2023")
REFS = ("persistence", "blend")
FEATURES = ("source_mean", "historical")
METRICS = {"all": "test_rmse_kelvin", "hot": "fixed_305k_cnn_rmse_kelvin"}
REFERENCE_METRICS = {"all": "source_weight_blend_rmse_kelvin",
                     "hot": "fixed_305k_blend_rmse_kelvin"}


def discover(root):
    result = {}
    for path in Path(root).rglob("target-*.json"):
        parts = path.parts
        if "factorial" not in parts:
            continue
        ix = len(parts) - 1 - parts[::-1].index("factorial")
        suffix = parts[ix + 1:]
        if len(suffix) != 4 or suffix[0] not in CASES or suffix[1] not in REFS or suffix[2] not in FEATURES:
            continue
        name = suffix[3]
        if not name.startswith("target-") or not name.endswith(".json"):
            continue
        try:
            seed = int(name[7:-5])
        except ValueError:
            continue
        key = (suffix[0], seed, suffix[1], suffix[2])
        if key in result:
            raise ValueError(f"Duplicate target report {key}: {result[key][0]} and {path}")
        data = json.loads(path.read_text())
        if data.get("factorial_reference_mode") != suffix[1]:
            raise ValueError(f"Reference mismatch: {path}")
        if data.get("trained_climatology_feature_mode") != suffix[2]:
            raise ValueError(f"Feature mismatch: {path}")
        if data.get("climatology_mode") != "target_history":
            raise ValueError(f"Not matched target-history evaluation: {path}")
        if data.get("feature_only_perturbation"):
            raise ValueError(f"Feature-only OOD sensitivity is not a matched arm: {path}")
        for keyname in (*METRICS.values(), *REFERENCE_METRICS.values()):
            value = data.get(keyname)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {keyname} in {path}")
        result[key] = (path, data)
    expected = {(c, s, r, f) for c in CASES for s in SEEDS for r in REFS for f in FEATURES}
    absent, extra = expected - set(result), set(result) - expected
    if absent or extra:
        raise ValueError(f"Expected 160 unique arm reports; missing={sorted(absent)[:15]} ({len(absent)}); "
                         f"unexpected={sorted(extra)[:15]} ({len(extra)})")
    return result


def summarize(root):
    source = discover(root)
    rows = []
    for case in CASES:
        for seed in SEEDS:
            arm = {(ref, feat): source[(case, seed, ref, feat)][1]
                   for ref in REFS for feat in FEATURES}
            for ref in REFS:
                a, b = arm[(ref, "historical")], arm[(ref, "source_mean")]
                # The matched feature ablation must NOT change the residual baseline.
                for m, key in REFERENCE_METRICS.items():
                    if not math.isclose(a[key], b[key], rel_tol=0, abs_tol=1e-5):
                        raise ValueError(f"Reference differs across feature arms: {case}, {seed}, {ref}, {m}")
                if a.get("fixed_hot_samples") != b.get("fixed_hot_samples"):
                    raise ValueError(f"Hot samples differ: {case}, {seed}, {ref}")
            for metric, key in METRICS.items():
                p_s = arm[("persistence", "source_mean")][key]
                p_h = arm[("persistence", "historical")][key]
                b_s = arm[("blend", "source_mean")][key]
                b_h = arm[("blend", "historical")][key]
                dp, db = p_s-p_h, b_s-b_h
                rows.append(dict(case=case, seed=seed, subset=metric,
                                 persistence_source_mean_K=p_s,
                                 persistence_historical_K=p_h,
                                 blend_source_mean_K=b_s,
                                 blend_historical_K=b_h,
                                 persistence_reference_K=arm[("persistence", "historical")][REFERENCE_METRICS[metric]],
                                 blend_reference_K=arm[("blend", "historical")][REFERENCE_METRICS[metric]],
                                 delta_p_K=dp, delta_b_K=db, interaction_K=db-dp,
                                 hot_samples=arm[("blend", "historical")].get("fixed_hot_samples")))
    summary = []
    for case in CASES:
        for subset in METRICS:
            group = [r for r in rows if r["case"] == case and r["subset"] == subset]
            summary.append(dict(case=case, subset=subset, n_seeds=len(group),
                                delta_p_positive=sum(r["delta_p_K"] > 0 for r in group),
                                delta_b_positive=sum(r["delta_b_K"] > 0 for r in group),
                                interaction_negative=sum(r["interaction_K"] < 0 for r in group),
                                median_delta_p_K=statistics.median(r["delta_p_K"] for r in group),
                                median_delta_b_K=statistics.median(r["delta_b_K"] for r in group),
                                median_interaction_K=statistics.median(r["interaction_K"] for r in group),
                                mean_interaction_K=statistics.mean(r["interaction_K"] for r in group)))
    return rows, summary


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted original Actions artifacts")
    parser.add_argument("--out", required=True, help="Directory for CSV and manifest")
    args = parser.parse_args()
    rows, summary = summarize(args.root)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "factorial_10seed_4case_absolute_rmse.csv", rows)
    write_csv(dest / "factorial_case_summary.csv", summary)
    (dest / "README.md").write_text(
        "# Paper v4 evidence-derived tables\n\n"
        "40 paired seed-case combinations, two subsets each, all four independently trained arms.\n"
        "Positive delta_p and delta_b mean historical channel reduces RMSE.\n"
        "Interaction = delta_b - delta_p; negative means reduced marginal benefit with blend.\n"
        "The script checks identical reference within each matched comparison.\n"
        "Seed summaries describe initialization variability, not independent climate samples.\n"
        "Bootstrap confidence intervals need the underlying time-resolved fields and separate estimation.\n"
    )
    print(f"Validated {len(rows)} case-seed-subset comparisons; wrote {dest}")


if __name__ == "__main__":
    main()
