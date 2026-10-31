"""Reproducible training and held-out verification."""
import argparse, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .data import synthetic_data, load_era5, prepare_splits
from .model import init_model, predict

def train(arr, epochs=4, batch=8, lr=0.001, channels=12, lead_steps=4, seed=42):
    splits,(mean,std)=prepare_splits(arr,lead_steps=lead_steps)
    (xt,yt),(xv,yv),(xe,ye)=splits
    params=init_model(jax.random.PRNGKey(seed),channels)
    @jax.jit
    def update(params,x,y):
        def loss(p): return jnp.mean((predict(p,x)-y)**2)
        val,g=jax.value_and_grad(loss)(params)
        return jax.tree_util.tree_map(lambda p,grad:p-lr*grad,params,g),val
    rng=np.random.default_rng(seed)
    history=[]; start=time.perf_counter()
    for epoch in range(epochs):
        losses=[]
        for ids in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/batch)))):
            params,loss=update(params,jnp.asarray(xt[ids]),jnp.asarray(yt[ids]))
            losses.append(float(loss))
        val_rmse=float(np.sqrt(np.mean((np.asarray(predict(params,jnp.asarray(xv)))-yv)**2))*std)
        history.append({'epoch':epoch+1,'train_mse_normalized':float(np.mean(losses)),'val_rmse_kelvin':val_rmse})
        print(f'epoch {epoch+1}/{epochs} train_loss={history[-1]["train_mse_normalized"]:.4f} val_rmse_K={val_rmse:.3f}',flush=True)
    pred=np.asarray(predict(params,jnp.asarray(xe)))
    model_rmse=float(np.sqrt(np.mean((pred-ye)**2))*std)
    persistence_rmse=float(np.sqrt(np.mean((xe-ye)**2))*std)
    metrics={'test_rmse_kelvin':model_rmse,'persistence_test_rmse_kelvin':persistence_rmse,
             'test_mae_kelvin':float(np.mean(np.abs(pred-ye))*std),
             'test_samples':len(xe),'validation_samples':len(xv),'train_samples':len(xt),
             'elapsed_seconds':time.perf_counter()-start,'device':str(jax.devices()[0]),
             'normalization_train_mean_kelvin':mean,'normalization_train_std_kelvin':std,
             'history':history}
    return params,metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='synthetic',help='synthetic or local .nc/.zarr ERA5 path')
    ap.add_argument('--variable',default='t2m')
    ap.add_argument('--lead-steps',type=int,default=4,help='For 6-hour data, 4 steps = 24 h; for hourly ERA5, use 24')
    ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--channels',type=int,default=12)
    ap.add_argument('--output',default='outputs/smoke')
    args=ap.parse_args()
    if args.data=='synthetic': arr=synthetic_data()
    else:
        da=load_era5(args.data,variable=args.variable)
        arr=da.values
    params,metrics=train(arr,epochs=args.epochs,batch=args.batch,channels=args.channels,lead_steps=args.lead_steps)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez(out/'model_weights.npz',**{k:np.asarray(v) for k,v in params.items()})
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'},indent=2))
if __name__=='__main__':main()
