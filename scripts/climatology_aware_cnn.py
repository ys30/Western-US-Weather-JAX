"""Train a JAX CNN residual on a validation-selected climatology/persistence blend.

Climatology uses only training dates, blending weight uses validation dates.
The independent test year is not consulted for parameter selection.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from weatherjax.data import load_era5,prepare_splits
from weatherjax.model import init_model,predict
from scripts.climatology_baselines import benchmark
from weatherjax.train import paired_time_block_ci

def run(arr,timestamps,train_end,val_end,epochs=40,seed=42,batch=8,channels=12,lr=.001):
    ts=np.asarray(timestamps).astype('datetime64[h]')
    n=len(arr)
    a=int(np.searchsorted(ts,np.datetime64(train_end)))
    b=int(np.searchsorted(ts,np.datetime64(val_end)))
    baseline_report=benchmark(arr,ts,train_end,val_end,4)
    weight=baseline_report['selected_climatology_weight']
    months=(ts.astype('datetime64[M]').astype('int64')%12).astype(int)
    hours=(ts.astype('datetime64[h]').astype('int64')%24).astype(int)
    bins=months*24+hours
    climate={}
    for key in np.unique(bins[:a]):
        climate[int(key)]=np.asarray(arr[:a][bins[:a]==key].mean(axis=0),dtype=np.float32)
    fallback=np.asarray(arr[:a].mean(axis=0),dtype=np.float32)
    (tr,va,te),(mean,std)=prepare_splits(arr,lead_steps=4,history_steps=4,timestamps=ts,time_features=True,train_end=a,val_end=b)
    def enrich(split,lo,hi):
        x,y=split
        indices=np.arange(max(lo,7),hi)
        assert len(indices)==len(x)
        climatology=np.stack([climate.get(int(bins[i]),fallback) for i in indices])
        persistence=arr[indices-4]
        blended=(1-weight)*persistence+weight*climatology
        # Append physical-temperature climatology as a feature, then blended baseline LAST.
        # predict(params,x) adds CNN correction to x[..., -1:]
        x=np.concatenate([x,(climatology[...,None]-mean)/std,(blended[...,None]-mean)/std],axis=-1)
        return np.asarray(x,dtype=np.float32),y,np.asarray(blended,dtype=np.float32)
    xt,yt,_=enrich(tr,4,a)
    xv,yv,_=enrich(va,a,b)
    xe,ye,baseline=enrich(te,b,n)
    params=init_model(jax.random.PRNGKey(seed),channels,input_channels=xt.shape[-1],zero_residual=True)
    @jax.jit
    def update(p,x,y):
        def loss(q):return jnp.mean((predict(q,x)-y)**2)
        _,grads=jax.value_and_grad(loss)(p)
        return jax.tree_util.tree_map(lambda v,g:v-lr*g,p,grads)
    rng=np.random.default_rng(seed)
    best=float('inf');best_epoch=0;best_params=params
    start=time.perf_counter()
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/batch)))):
            params=update(params,jnp.asarray(xt[ids]),jnp.asarray(yt[ids]))
        val_pred=np.asarray(predict(params,jnp.asarray(xv)))
        vrmse=float(np.sqrt(np.mean((val_pred-yv)**2))*std)
        if vrmse<best:
            best=vrmse;best_epoch=epoch+1
            best_params=jax.tree_util.tree_map(lambda v:jnp.array(v),params)
        print(f'epoch {epoch+1}/{epochs} val_rmse_K={vrmse:.6f}',flush=True)
    pred=np.asarray(predict(best_params,jnp.asarray(xe)))[...,0]*std+mean
    truth=np.asarray(ye[...,0])*std+mean
    mask=truth>=305.0
    score=lambda z,m:float(np.sqrt(np.mean((z[m]-truth[m])**2)))
    uncertainty=[paired_time_block_ci(pred[...,None],baseline[...,None],truth[...,None],block_steps=k,draws=1000,seed=42) for k in (4,12,28,56)]
    result={'paired_block_sensitivity':uncertainty,'random_seed':seed,'test_rmse_kelvin':score(pred,np.ones_like(mask,dtype=bool)),
      'blended_baseline_rmse_kelvin':score(baseline,np.ones_like(mask,dtype=bool)),
      'fixed_305k_cnn_rmse_kelvin':score(pred,mask) if mask.any() else None,
      'fixed_305k_blend_rmse_kelvin':score(baseline,mask) if mask.any() else None,
      'fixed_hot_samples':int(mask.sum()),'selected_blend_climatology_weight':weight,
      'selection_period':str(ts[a])+' to '+str(ts[b-1]),
      'train_period':str(ts[0])+' to '+str(ts[a-1]),
      'test_period':str(ts[b])+' to '+str(ts[-1]),
      'best_validation_rmse_kelvin':best,'best_epoch':best_epoch,
      'training_elapsed_seconds':time.perf_counter()-start,
      'note':'Training-only climatology, validation-only blend weight and checkpoint, independent 2022 test. One seed only.'}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-until',default='2021-01-01');p.add_argument('--validate-until',default='2022-01-01')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();da=load_era5(a.data)
    report=run(da.values,da.time.values,a.train_until,a.validate_until,a.epochs,seed=a.seed)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
