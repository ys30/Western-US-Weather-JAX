"""Fail-closed comparison of CPU and independently hardware-audited GPU training."""
import argparse,json,math
from pathlib import Path

def compare(cpu,gpu,relative_rmse_tolerance=0.05):
    audit=gpu.get("hardware_audit",{})
    if audit.get("training_verified_on_gpu") is not True or not audit.get("gpu_devices") or not audit.get("parameter_devices"):
        raise ValueError("GPU hardware audit absent or incomplete; parity NOT verified")
    if audit.get("parameter_platforms") != ["gpu"]:
        raise ValueError("GPU parameter platform audit must contain only gpu")
    if audit.get("jax_backend") != "gpu":
        raise ValueError("JAX default backend is not GPU")
    for k in ("epochs_requested","batch_size","learning_rate","channels","random_seed","lead_steps","history_steps","hot_weight","hot_threshold_kelvin","train_end_index","validation_end_index","zero_residual","time_features","train_samples","validation_samples","test_samples"):
        if cpu.get(k)!=gpu.get(k): raise ValueError(f"Mismatched CPU/GPU experiment config: {k}")
    a=float(cpu["test_rmse_kelvin"]);b=float(gpu["test_rmse_kelvin"])
    elapsed_cpu=float(cpu["elapsed_seconds"]);elapsed_gpu=float(gpu["elapsed_seconds"])
    if not all(math.isfinite(x) and x>0 for x in (a,b,elapsed_cpu,elapsed_gpu)):
        raise ValueError("Nonfinite or nonpositive metrics")
    delta=abs(a-b)/a
    return {"hardware_verified":True,"cpu_test_rmse_k":a,"gpu_test_rmse_k":b,
        "relative_rmse_difference":delta,"rmse_parity_pass":delta<=relative_rmse_tolerance,
        "cpu_train_seconds":elapsed_cpu,"gpu_train_seconds":elapsed_gpu,
        "elapsed_speedup_cpu_over_gpu":elapsed_cpu/elapsed_gpu,
        "caveat":"Timing includes implementation-dependent JAX compilation and host/device overhead; repeated controlled measurements needed for a performance claim."}

def main():
    p=argparse.ArgumentParser();p.add_argument("cpu");p.add_argument("gpu")
    p.add_argument("--output",required=True);p.add_argument("--relative-rmse-tolerance",type=float,default=0.05)
    a=p.parse_args()
    if not 0<=a.relative_rmse_tolerance<1:p.error("Invalid tolerance")
    result=compare(json.loads(Path(a.cpu).read_text()),json.loads(Path(a.gpu).read_text()),a.relative_rmse_tolerance)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result["rmse_parity_pass"]:raise SystemExit("CPU/GPU model RMSE mismatch outside declared tolerance")
if __name__=="__main__":main()
