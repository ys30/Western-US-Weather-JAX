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

def hot_time_block_ci(pred, ref, truth, threshold=305.0, block_steps=4, draws=1000, seed=42):
    """Paired time-block bootstrap of fixed-threshold hot gridcell squared errors.

    Sample ALL valid times, retaining intra-time spatial dependence. Only truth-hot
    grid cells contribute after resampling; never sample hot-only times.
    """
    p=np.asarray(pred);r=np.asarray(ref);y=np.asarray(truth)
    if p.shape!=r.shape or p.shape!=y.shape or p.ndim!=3:
        raise ValueError("Expected matched [time,lat,lon] arrays")
    if block_steps<1 or draws<1 or len(y)<2:raise ValueError("Invalid block bootstrap")
    mask=y>=threshold
    count=mask.sum(axis=(1,2)).astype(np.float64)
    if count.sum()==0:raise ValueError("No fixed-threshold hot truth observations")
    se_p=np.sum(np.where(mask,(p-y)**2,0),axis=(1,2),dtype=np.float64)
    se_r=np.sum(np.where(mask,(r-y)**2,0),axis=(1,2),dtype=np.float64)
    point=float(np.sqrt(se_p.sum()/count.sum())-np.sqrt(se_r.sum()/count.sum()))
    rng=np.random.default_rng(seed)
    values=[]
    n=len(y);k=int(np.ceil(n/block_steps))
    for _ in range(draws):
        ids=((rng.integers(0,n,k)[:,None]+np.arange(block_steps))%n).reshape(-1)[:n]
        total=count[ids].sum()
        if total>0:values.append(float(np.sqrt(se_p[ids].sum()/total)-np.sqrt(se_r[ids].sum()/total)))
    if len(values)<int(.95*draws):raise ValueError("Too many bootstrap draws with no hot observations")
    lo,hi=np.quantile(values,[.025,.975])
    return {'difference_kelvin':point,'ci95_lower_kelvin':float(lo),'ci95_upper_kelvin':float(hi),
            'block_steps':block_steps,'draws_valid':len(values),'draws_requested':draws,
            'hot_gridcell_times':int(count.sum())}

def monthly_diagnostics(pred,ref,truth,target_times,threshold=305.0):
    out=[]
    months=np.asarray(target_times).astype('datetime64[M]')
    for m in np.unique(months):
        take=months==m
        yp=truth[take]; pp=pred[take]; rr=ref[take]
        hot=yp>=threshold
        rmse=lambda z,mask:float(np.sqrt(np.mean((z[mask]-yp[mask])**2))) if mask.any() else None
        out.append({'month':str(m),'all_cnn_rmse_kelvin':rmse(pp,np.ones_like(hot,dtype=bool)),
                    'all_blend_rmse_kelvin':rmse(rr,np.ones_like(hot,dtype=bool)),
                    'fixed_hot_cnn_rmse_kelvin':rmse(pp,hot),
                    'fixed_hot_blend_rmse_kelvin':rmse(rr,hot),
                    'fixed_hot_gridcell_times':int(hot.sum())})
    return out

def run(arr,timestamps,train_end,val_end,epochs=40,seed=42,batch=8,channels=12,lr=.001,hot_weight=0.0,hot_threshold_kelvin=305.0):
    if hot_weight < 0: raise ValueError('hot_weight must be nonnegative')
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
        def loss(q):
            sq=(predict(q,x)-y)**2
            weights=1.0+hot_weight*(y>=((hot_threshold_kelvin-mean)/std))
            return jnp.sum(weights*sq)/jnp.sum(weights)
        _,grads=jax.value_and_grad(loss)(p)
        return jax.tree_util.tree_map(lambda v,g:v-lr*g,p,grads)
    rng=np.random.default_rng(seed)
    best=float('inf');best_epoch=0;best_params=params;best_hot=None
    start=time.perf_counter()
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/batch)))):
            params=update(params,jnp.asarray(xt[ids]),jnp.asarray(yt[ids]))
        val_pred=np.asarray(predict(params,jnp.asarray(xv)))
        vrmse=float(np.sqrt(np.mean((val_pred-yv)**2))*std)
        vmask=(yv[...,0]*std+mean)>=hot_threshold_kelvin
        vhot=(float(np.sqrt(np.mean((val_pred[...,0][vmask]-yv[...,0][vmask])**2))*std) if vmask.any() else None)
        if vrmse<best:
            best=vrmse;best_hot=vhot;best_epoch=epoch+1
            best_params=jax.tree_util.tree_map(lambda v:jnp.array(v),params)
        print(f'epoch {epoch+1}/{epochs} val_rmse_K={vrmse:.6f}',flush=True)
    pred=np.asarray(predict(best_params,jnp.asarray(xe)))[...,0]*std+mean
    truth=np.asarray(ye[...,0])*std+mean
    mask=truth>=hot_threshold_kelvin
    score=lambda z,m:float(np.sqrt(np.mean((z[m]-truth[m])**2)))
    uncertainty=[paired_time_block_ci(pred[...,None],baseline[...,None],truth[...,None],block_steps=k,draws=1000,seed=42) for k in (4,12,28,56)]
    hot_uncertainty=[hot_time_block_ci(pred,baseline,truth,threshold=hot_threshold_kelvin,
                      block_steps=k,draws=1000,seed=42) for k in (4,12,28,56)]
    months=monthly_diagnostics(pred,baseline,truth,ts[b:],threshold=hot_threshold_kelvin)
    result={'fixed_heat_paired_block_sensitivity':hot_uncertainty,'monthly_diagnostics':months,'paired_block_sensitivity':uncertainty,'random_seed':seed,'hot_weight':hot_weight,'hot_threshold_kelvin':hot_threshold_kelvin,'best_validation_fixed_heat_rmse_kelvin':best_hot,'test_rmse_kelvin':score(pred,np.ones_like(mask,dtype=bool)),
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
    p.add_argument('--hot-weight',type=float,default=0.0)
    p.add_argument('--hot-threshold-kelvin',type=float,default=305.0)
    a=p.parse_args();da=load_era5(a.data)
    report=run(da.values,da.time.values,a.train_until,a.validate_until,a.epochs,seed=a.seed,hot_weight=a.hot_weight,hot_threshold_kelvin=a.hot_threshold_kelvin)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
