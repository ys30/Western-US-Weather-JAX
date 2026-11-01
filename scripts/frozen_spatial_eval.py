"""Evaluate source-trained CNN with frozen weights; target historical climatology allowed."""
import argparse,json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from weatherjax.data import load_era5
from weatherjax.model import predict
from scripts.heat_events import event_diagnostics,event_definition_sensitivity

def frozen_evaluate(data,times,checkpoint,test_start="2022-01-01",threshold=305.):
    a=np.asarray(data,dtype=np.float32)
    ts=np.asarray(times).astype("datetime64[h]")
    cut=int(np.searchsorted(ts,np.datetime64(test_start)))
    if a.ndim!=3 or not np.isfinite(a).all() or cut<12 or cut>=len(a):
        raise ValueError("Insufficient or invalid target data")
    if np.any(np.diff(ts).astype("timedelta64[h]").astype(int)!=6):
        raise ValueError("Expected continuous six-hourly ERA5")
    with np.load(checkpoint,allow_pickle=False) as z:
        params={k:jnp.asarray(z["param_"+k]) for k in ("w1","w2","w3","b1","b2","b3")}
        mean=float(z["source_train_mean"]);std=float(z["source_train_std"])
        weight=float(z["source_validation_selected_blend_weight"])
    if not np.isfinite([mean,std,weight]).all() or std<=0 or not 0<=weight<=1:
        raise ValueError("Invalid source metadata")
    months=(ts.astype("datetime64[M]").astype("int64")%12).astype(int)
    hours=(ts.astype("datetime64[h]").astype("int64")%24).astype(int)
    bins=months*24+hours
    climate={int(k):np.asarray(a[:cut][bins[:cut]==k].mean(axis=0),dtype=np.float32)
             for k in np.unique(bins[:cut])}
    fallback=np.asarray(a[:cut].mean(axis=0),dtype=np.float32)
    indices=np.arange(cut,len(a))
    clim=np.stack([climate.get(int(bins[i]),fallback) for i in indices])
    baseline=(1-weight)*a[indices-4]+weight*clim
    input_indices=indices[:,None]-4-np.arange(3,-1,-1)[None,:]
    frames=np.moveaxis((a[input_indices]-mean)/std,1,-1)
    hour=hours[indices].astype(np.float32)
    cycle=np.stack((np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)),axis=-1).astype(np.float32)
    cyc=np.broadcast_to(cycle[:,None,None,:],(*frames.shape[:3],2))
    x=np.concatenate((cyc,frames,(clim[...,None]-mean)/std,
                      (baseline[...,None]-mean)/std),axis=-1).astype(np.float32)
    if x.shape[-1]!=params["w1"].shape[2]:raise ValueError("Feature mismatch")
    # No training, validation selection, gradients or checkpoint updates on target.
    pred=np.concatenate([np.asarray(predict(params,jnp.asarray(x[i:i+32])))[...,0]
                         for i in range(0,len(x),32)])*std+mean
    truth=a[cut:];mask=truth>=threshold
    rmse=lambda v,m:float(np.sqrt(np.mean((v[m]-truth[m])**2))) if m.any() else None
    return {"protocol":"Frozen source CNN; target pre-test climatology; NO target weight updates",
        "strict_zero_history_transfer":False,"target_history_last_timestamp":str(ts[cut-1]),
        "target_test_start":str(ts[cut]),"source_train_normalization":True,
        "source_validation_blend_weight":weight,
        "test_rmse_kelvin":rmse(pred,np.ones_like(mask,dtype=bool)),
        "source_weight_blend_rmse_kelvin":rmse(baseline,np.ones_like(mask,dtype=bool)),
        "fixed_305k_cnn_rmse_kelvin":rmse(pred,mask),
        "fixed_305k_blend_rmse_kelvin":rmse(baseline,mask),
        "fixed_hot_samples":int(mask.sum()),
        "heat_event_diagnostics":event_diagnostics(pred,baseline,truth,ts[cut:],threshold_kelvin=threshold),
        "heat_event_definition_sensitivity":event_definition_sensitivity(pred,baseline,truth,ts[cut:],threshold_kelvin=threshold)}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True);p.add_argument("--test-start",default="2022-01-01")
    v=p.parse_args();da=load_era5(v.data)
    result=frozen_evaluate(da.values,da.time.values,v.checkpoint,v.test_start)
    out=Path(v.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
