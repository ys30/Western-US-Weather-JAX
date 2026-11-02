"""Evaluate source-trained CNN with frozen weights; target historical climatology allowed."""
import argparse,json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from weatherjax.time_audit import audit_timeline
from weatherjax.data import load_era5
from weatherjax.model import predict
from scripts.heat_events import event_diagnostics,event_definition_sensitivity
from scripts.climatology_aware_cnn import paired_time_block_ci,hot_time_block_ci

def frozen_evaluate(data,times,checkpoint,test_start="2022-01-01",threshold=305.,climatology_mode="target_history",diagnostics_output=None,latitudes=None,longitudes=None):
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
        trained_feature_mode=str(z["climatology_feature_mode"].item()) if "climatology_feature_mode" in z else "historical"
    if not np.isfinite([mean,std,weight]).all() or std<=0 or not 0<=weight<=1:
        raise ValueError("Invalid source metadata")
    months=(ts.astype("datetime64[M]").astype("int64")%12).astype(int)
    hours=(ts.astype("datetime64[h]").astype("int64")%24).astype(int)
    bins=months*24+hours
    if climatology_mode not in ("target_history","source_global_constant","source_global_feature_only"):
        raise ValueError("Invalid climatology mode")
    indices=np.arange(cut,len(a))
    if climatology_mode in ("target_history","source_global_feature_only"):
        climate={int(k):np.asarray(a[:cut][bins[:cut]==k].mean(axis=0),dtype=np.float32)
                 for k in np.unique(bins[:cut])}
        fallback=np.asarray(a[:cut].mean(axis=0),dtype=np.float32)
        clim=np.stack([climate.get(int(bins[i]),fallback) for i in indices])
    else:
        clim=np.full((len(indices),*a.shape[1:]),mean,dtype=np.float32)
    baseline=(1-weight)*a[indices-4]+weight*clim
    # Controlled inference-time feature perturbation: identical frozen weights,
    # target observations and reference; replace only the climatology input channel.
    # This is out-of-distribution sensitivity, NOT a trained matched ablation.
    feature_clim=(np.full_like(clim,mean) if (climatology_mode=="source_global_feature_only" or (climatology_mode=="target_history" and trained_feature_mode=="source_mean")) else clim)
    input_indices=indices[:,None]-4-np.arange(3,-1,-1)[None,:]
    frames=np.moveaxis((a[input_indices]-mean)/std,1,-1)
    hour=hours[indices].astype(np.float32)
    cycle=np.stack((np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)),axis=-1).astype(np.float32)
    cyc=np.broadcast_to(cycle[:,None,None,:],(*frames.shape[:3],2))
    x=np.concatenate((cyc,frames,(feature_clim[...,None]-mean)/std,
                      (baseline[...,None]-mean)/std),axis=-1).astype(np.float32)
    if x.shape[-1]!=params["w1"].shape[2]:raise ValueError("Feature mismatch")
    # No training, validation selection, gradients or checkpoint updates on target.
    pred=np.concatenate([np.asarray(predict(params,jnp.asarray(x[i:i+32])))[...,0]
                         for i in range(0,len(x),32)])*std+mean
    truth=a[cut:];mask=truth>=threshold
    if diagnostics_output is not None:
        from scripts.spatiotemporal_diagnostics import export_diagnostics
        export_diagnostics(diagnostics_output,pred,baseline,truth,ts[cut:],threshold,latitudes,longitudes)
    rmse=lambda v,m:float(np.sqrt(np.mean((v[m]-truth[m])**2))) if m.any() else None
    annual_ci=[paired_time_block_ci(pred[...,None],baseline[...,None],truth[...,None],block_steps=k,draws=1000,seed=42) for k in (4,12,28,56)]
    heat_ci=([hot_time_block_ci(pred,baseline,truth,threshold=threshold,block_steps=k,draws=1000,seed=42) for k in (4,12,28,56)] if mask.any() else [])
    return {"annual_paired_block_sensitivity":annual_ci,"fixed_heat_paired_block_sensitivity":heat_ci,
        "protocol":"Frozen source CNN; target pre-test climatology; NO target weight updates",
        "strict_zero_history_transfer":climatology_mode=="source_global_constant",
        "climatology_mode":climatology_mode,
        "trained_climatology_feature_mode":trained_feature_mode,
        "matched_training_common_reference":climatology_mode=="target_history",
        "feature_only_perturbation":climatology_mode=="source_global_feature_only",
        "reference_held_fixed":climatology_mode=="source_global_feature_only",
        "feature_perturbation_caveat":"Out-of-distribution inference sensitivity; not causal climatology ablation" if climatology_mode=="source_global_feature_only" else None,"target_history_last_timestamp":str(ts[cut-1]),
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
    p.add_argument("--climatology-mode",choices=["target_history","source_global_constant","source_global_feature_only"],default="target_history")
    p.add_argument("--diagnostics-output",default=None,help="Optional compressed arrays and monthly/gridded metrics; requires verified target data and checkpoint")
    v=p.parse_args();da=load_era5(v.data,lat_min=-90,lat_max=90,lon_min=-180,lon_max=180)
    result=frozen_evaluate(da.values,da.time.values,v.checkpoint,v.test_start,climatology_mode=v.climatology_mode,diagnostics_output=v.diagnostics_output,latitudes=da.lat.values,longitudes=da.lon.values)
    out=Path(v.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
