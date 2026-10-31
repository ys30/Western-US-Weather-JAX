"""Reproducible training and held-out verification."""
import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .data import synthetic_data, load_era5, prepare_splits
from .model import init_model, predict

def paired_time_block_ci(pred, reference, truth, block_steps=4, draws=400, seed=42):
    """Paired block bootstrap over forecast valid times, preserving spatial correlation."""
    pred=np.asarray(pred); reference=np.asarray(reference); truth=np.asarray(truth)
    if pred.shape!=reference.shape or pred.shape!=truth.shape or pred.ndim<2:
        raise ValueError("Expected matching [time, ...] arrays")
    n=len(pred)
    if n<2 or block_steps<1 or draws<1: raise ValueError("Insufficient data or invalid bootstrap configuration")
    p=np.mean((pred-truth)**2,axis=tuple(range(1,pred.ndim)))
    r=np.mean((reference-truth)**2,axis=tuple(range(1,pred.ndim)))
    rng=np.random.default_rng(seed)
    k=int(np.ceil(n/block_steps))
    values=[]
    for _ in range(draws):
        starts=rng.integers(0,n,size=k)
        ids=((starts[:,None]+np.arange(block_steps))%n).reshape(-1)[:n]
        values.append(float(np.sqrt(p[ids].mean())-np.sqrt(r[ids].mean())))
    lo,hi=np.quantile(values,[.025,.975])
    return {'rmse_difference_kelvin':float(np.sqrt(p.mean())-np.sqrt(r.mean())),
            'ci95_lower_kelvin':float(lo),'ci95_upper_kelvin':float(hi),
            'bootstrap_draws':draws,'block_steps':block_steps,
            'interpretation':'Negative differences favor CNN; exploratory block bootstrap, not formal spatial inference'}

def train(arr, epochs=4, batch=8, lr=0.001, channels=12, lead_steps=4, seed=42, history_steps=4, zero_residual=False, timestamps=None, time_features=False, train_end=None, val_end=None, return_predictions=False):
    splits,(mean,std)=prepare_splits(arr,lead_steps=lead_steps,history_steps=history_steps,timestamps=timestamps,time_features=time_features,train_end=train_end,val_end=val_end)
    (xt,yt),(xv,yv),(xe,ye)=splits
    params=init_model(jax.random.PRNGKey(seed),channels,input_channels=history_steps+(2 if time_features else 0),zero_residual=zero_residual)
    @jax.jit
    def update(params,x,y):
        def loss(p): return jnp.mean((predict(p,x)-y)**2)
        val,g=jax.value_and_grad(loss)(params)
        return jax.tree_util.tree_map(lambda p,grad:p-lr*grad,params,g),val
    rng=np.random.default_rng(seed)
    history=[]; start=time.perf_counter()
    best_params=params; best_val=float('inf'); best_epoch=0
    for epoch in range(epochs):
        losses=[]
        for ids in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/batch)))):
            params,loss=update(params,jnp.asarray(xt[ids]),jnp.asarray(yt[ids]))
            losses.append(float(loss))
        val_rmse=float(np.sqrt(np.mean((np.asarray(predict(params,jnp.asarray(xv)))-yv)**2))*std)
        if val_rmse < best_val:
            best_val=val_rmse; best_epoch=epoch+1
            best_params=jax.tree_util.tree_map(lambda p: jnp.array(p),params)
        history.append({'epoch':epoch+1,'train_mse_normalized':float(np.mean(losses)),'val_rmse_kelvin':val_rmse})
        print(f'epoch {epoch+1}/{epochs} train_loss={history[-1]["train_mse_normalized"]:.4f} val_rmse_K={val_rmse:.3f}',flush=True)
    params=best_params
    pred=np.asarray(predict(params,jnp.asarray(xe)))
    model_rmse=float(np.sqrt(np.mean((pred-ye)**2))*std)
    persistence_rmse=float(np.sqrt(np.mean((xe[...,-1:]-ye)**2))*std)
    metrics={'test_rmse_kelvin':model_rmse,'persistence_test_rmse_kelvin':persistence_rmse,
             'test_mae_kelvin':float(np.mean(np.abs(pred-ye))*std),
             'train_end_index':train_end,'validation_end_index':val_end,'history_steps':history_steps,'zero_residual':zero_residual,'time_features':time_features,'best_epoch':best_epoch,'best_validation_rmse_kelvin':best_val,'test_samples':len(xe),'validation_samples':len(xv),'train_samples':len(xt),
             'elapsed_seconds':time.perf_counter()-start,'device':str(jax.devices()[0]),
             'normalization_train_mean_kelvin':mean,'normalization_train_std_kelvin':std,
             'history':history}
    if return_predictions:
        n=len(arr); b=val_end if val_end is not None else int(n*.85)
        target_indices=np.arange(max(b,lead_steps+history_steps-1),n)
        return params,metrics,{'target_indices':target_indices,'pred_kelvin':pred*std+mean,'truth_kelvin':ye*std+mean,'persistence_kelvin':xe[...,-1:]*std+mean}
    return params,metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='synthetic',help='synthetic or local .nc/.zarr ERA5 path')
    ap.add_argument('--variable',default='t2m')
    ap.add_argument('--lead-steps',type=int,default=4,help='For 6-hour data, 4 steps = 24 h; for hourly ERA5, use 24')
    ap.add_argument('--history-steps',type=int,default=4,help='Input temperature frames spaced at the data timestep')
    ap.add_argument('--time-features',action='store_true',help='Add cyclic target UTC-hour channels')
    ap.add_argument('--zero-residual',action='store_true',help='Initialize CNN output at persistence baseline')
    ap.add_argument('--train-until',help='exclusive UTC train cutoff, e.g. 2021-01-01')
    ap.add_argument('--validate-until',help='exclusive UTC validation cutoff, e.g. 2021-07-01')
    ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--channels',type=int,default=12)
    ap.add_argument('--group-metrics',action='store_true',help='Export independent test metrics by month and hot-temperature quantiles')
    ap.add_argument('--output',default='outputs/smoke')
    args=ap.parse_args()
    if args.data=='synthetic':
        arr=synthetic_data()
        timestamps=np.datetime64('2021-01-01T00')+np.arange(len(arr))*np.timedelta64(6,'h')
    else:
        da=load_era5(args.data,variable=args.variable)
        arr=da.values
        timestamps=da.time.values
    if bool(args.train_until) != bool(args.validate_until):
        ap.error('Specify both --train-until and --validate-until')
    if args.train_until:
        ts=np.asarray(timestamps).astype('datetime64[ns]')
        if not np.all(np.diff(ts)>np.timedelta64(0,'ns')):
            raise ValueError('Expected strictly increasing timestamps')
        train_end=int(np.searchsorted(ts,np.datetime64(args.train_until)))
        val_end=int(np.searchsorted(ts,np.datetime64(args.validate_until)))
    else:
        train_end=val_end=None
    result=train(arr,return_predictions=args.group_metrics,epochs=args.epochs,batch=args.batch,channels=args.channels,lead_steps=args.lead_steps,history_steps=args.history_steps,lr=args.lr,zero_residual=args.zero_residual,timestamps=timestamps,time_features=args.time_features,train_end=train_end,val_end=val_end)
    if args.group_metrics:
        params,metrics,details=result
    else:
        params,metrics=result
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.group_metrics:
        truth=details['truth_kelvin']; pred=details['pred_kelvin']; persistence=details['persistence_kelvin']
        target_times=np.asarray(timestamps)[details['target_indices']].astype('datetime64[M]')
        records=[]
        for month in np.unique(target_times):
            mask=target_times==month
            records.append({'month':str(month),'n_times':int(mask.sum()),'cnn_rmse_k':float(np.sqrt(np.mean((pred[mask]-truth[mask])**2))),'persistence_rmse_k':float(np.sqrt(np.mean((persistence[mask]-truth[mask])**2)))})
        # Exploratory only: test-set quantile computed from test truth, not usable for model tuning.
        threshold=float(np.quantile(truth,.95))
        hot=truth>=threshold
        metrics['paired_block_bootstrap']=paired_time_block_ci(pred,persistence,truth)
        metrics['group_evaluation']={'monthly':records,'hot_test_p95_kelvin':threshold,'hot_gridcell_count':int(hot.sum()),'hot_cnn_rmse_k':float(np.sqrt(np.mean((pred[hot]-truth[hot])**2))),'hot_persistence_rmse_k':float(np.sqrt(np.mean((persistence[hot]-truth[hot])**2))),'hot_quantile_note':'Descriptive test-tail metric; not a pre-registered climatic extreme threshold'}

    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez(out/'model_weights.npz',**{k:np.asarray(v) for k,v in params.items()})
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'},indent=2))
if __name__=='__main__':main()
