"""Repeat real GPU training in fresh subprocesses; report observed timing variability."""
import argparse,json,statistics,subprocess,sys
from pathlib import Path

def summarize(metrics_paths):
    rows=[]
    for p in metrics_paths:
        m=json.loads(Path(p).read_text())
        a=m.get("hardware_audit",{})
        if a.get("training_verified_on_gpu") is not True or a.get("jax_backend")!="gpu" or a.get("parameter_platforms")!=["gpu"]:
            raise ValueError("Unaudited GPU repetition")
        rows.append(m)
    if len(rows)<3: raise ValueError("At least 3 independent runs required")
    keys=("input_sha256","epochs_requested","batch_size","learning_rate","channels","random_seed","lead_steps",
          "history_steps","hot_weight","hot_threshold_kelvin","train_end_index","validation_end_index",
          "train_samples","validation_samples","test_samples")
    for k in keys:
        if not all(m.get(k)==rows[0].get(k) for m in rows):
            raise ValueError(f"GPU repetitions differ in {k}")
    times=[float(m["elapsed_seconds"]) for m in rows]
    if min(times)<=0: raise ValueError("Nonpositive elapsed time")
    rmses=[float(m["test_rmse_kelvin"]) for m in rows]
    return {"runs":len(rows),"gpu_elapsed_seconds":times,
        "elapsed_seconds_mean":statistics.mean(times),
        "elapsed_seconds_std":statistics.stdev(times),
        "elapsed_seconds_min":min(times),"elapsed_seconds_max":max(times),
        "test_rmse_kelvin":rmses,"rmse_range_kelvin":max(rmses)-min(rmses),
        "note":"Separate processes include per-run compilation. CPU/GPU speedup needs a matched CPU reference."}

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",default="outputs/gpu-repeats")
    p.add_argument("--repeats",type=int,default=3);p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--train-until",default="2021-01-01");p.add_argument("--validate-until",default="2022-01-01")
    p.add_argument("--hot-weight",type=float,default=2);p.add_argument("--hot-threshold-kelvin",type=float,default=305)
    a=p.parse_args()
    if a.repeats<3: p.error("repeats must be >= 3")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i in range(a.repeats):
        dest=out/f"run-{i+1:02d}"
        cmd=[sys.executable,"scripts/gpu_training.py","--data",a.data,"--output",str(dest),
             "--epochs",str(a.epochs),"--train-until",a.train_until,"--validate-until",a.validate_until,
             "--hot-weight",str(a.hot_weight),"--hot-threshold-kelvin",str(a.hot_threshold_kelvin)]
        subprocess.run(cmd,check=True)
        paths.append(dest/"gpu_metrics.json")
    result=summarize(paths)
    (out/"repeat-summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
