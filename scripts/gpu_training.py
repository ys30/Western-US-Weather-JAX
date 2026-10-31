"""Hardware-audited real-ERA5 JAX training entry point.

Requires an actual GPU. Exits nonzero rather than silently falling back to CPU.
"""
import argparse, json, platform
from pathlib import Path
import jax
from weatherjax.data import load_era5
from weatherjax.preflight import inspect_era5
from weatherjax.train import train
import numpy as np

def require_gpu(devices=None):
    devices=jax.devices() if devices is None else devices
    gpu=[d for d in devices if d.platform=="gpu"]
    if not gpu:
        raise RuntimeError("No JAX GPU backend detected: GPU training was NOT performed.")
    return gpu

def run(data, output, epochs=20, stride_hours=6, history_steps=4, train_until="2021-01-01", validate_until="2021-07-01", hot_weight=0.0, hot_threshold_kelvin=305.0):
    gpu=require_gpu()
    report=inspect_era5(data,expected_step_hours=stride_hours)
    da=load_era5(data)
    ts=np.asarray(da.time.values).astype("datetime64[ns]")
    te=int(np.searchsorted(ts,np.datetime64(train_until)))
    ve=int(np.searchsorted(ts,np.datetime64(validate_until)))
    with jax.default_device(gpu[0]):
        params, metrics=train(da.values,epochs=epochs,lead_steps=24//stride_hours,history_steps=history_steps,
                              zero_residual=True,time_features=True,timestamps=ts,train_end=te,val_end=ve,hot_weight=hot_weight,hot_threshold_kelvin=hot_threshold_kelvin)
        jax.block_until_ready(params)
        parameter_devices={d for p in jax.tree_util.tree_leaves(params) for d in p.devices()}
        if not parameter_devices or any(d.platform != 'gpu' for d in parameter_devices):
            raise RuntimeError(f'Trained parameters are not resident on a GPU: {sorted(map(str, parameter_devices))}')
        used=sorted(map(str, parameter_devices))
    metrics["hardware_audit"]={"jax_backend":jax.default_backend(),"gpu_devices":[str(d) for d in gpu],"parameter_devices":used,
          "jax_version":jax.__version__,"python":platform.python_version(),
          "preflight":report,"training_verified_on_gpu":True}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"gpu_metrics.json").write_text(json.dumps(metrics,indent=2))
    np.savez(output/"gpu_model_weights.npz",**{k:np.asarray(v) for k,v in params.items()})
    print(json.dumps(metrics["hardware_audit"],indent=2))
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="outputs/gpu-era5")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--stride-hours",type=int,default=6)
    p.add_argument("--history-steps",type=int,default=4)
    p.add_argument("--hot-weight",type=float,default=0.0)
    p.add_argument("--hot-threshold-kelvin",type=float,default=305.0)
    p.add_argument("--train-until",default="2021-01-01")
    p.add_argument("--validate-until",default="2021-07-01")
    a=p.parse_args()
    if 24%a.stride_hours: p.error("stride-hours must divide 24")
    run(a.data,a.output,a.epochs,a.stride_hours,a.history_steps,a.train_until,a.validate_until,a.hot_weight,a.hot_threshold_kelvin)
if __name__=="__main__": main()
