"""Verify frozen-feature-only ERA5 experiment against the unchanged-reference control."""
import argparse
import json
import math
from pathlib import Path


def check_pair(control, altered, tol=1e-5):
    if control.get("climatology_mode") != "target_history":
        raise ValueError("Control must use target_history")
    if altered.get("climatology_mode") != "source_global_feature_only":
        raise ValueError("Treatment must use source_global_feature_only")
    if altered.get("feature_only_perturbation") is not True or altered.get("reference_held_fixed") is not True:
        raise ValueError("Feature-only treatment metadata missing")
    for field in ("target_test_start", "target_history_last_timestamp", "fixed_hot_samples"):
        if control[field] != altered[field]:
            raise ValueError(f"Misaligned target data: {field}")
    for field in ("source_weight_blend_rmse_kelvin", "fixed_305k_blend_rmse_kelvin", "source_validation_blend_weight"):
        a, b = control[field], altered[field]
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(a,b,abs_tol=tol,rel_tol=0):
            raise ValueError(f"Reference changed: {field}: {a} vs {b}")
    metrics = {}
    for field in ("test_rmse_kelvin", "fixed_305k_cnn_rmse_kelvin"):
        a,b = control[field],altered[field]
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)):
            raise ValueError(f"Missing or invalid metric: {field}")
        metrics[field] = {"control_K": a, "feature_only_K": b, "feature_only_minus_control_K": b-a}
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--control-dir",required=True)
    parser.add_argument("--feature-only-dir",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    summary={"experiment":"inference_time_climatology_channel_perturbation_NOT_matched_training",
             "not_causal_ablation":True,"seeds":{}}
    for seed in (42,7,123):
        control=json.loads((Path(args.control_dir)/f"seed-{seed}.json").read_text())
        changed=json.loads((Path(args.feature_only_dir)/f"seed-{seed}.json").read_text())
        summary["seeds"][str(seed)]=check_pair(control,changed)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
