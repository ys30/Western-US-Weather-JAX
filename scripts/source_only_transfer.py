"""Source-only residual JAX CNN without climatology features; freeze for region transfer."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from weatherjax.data import load_era5
from weatherjax.model import init_model,predict

def evaluate(source, source_times, target, target_times, seed=42, epochs=40,
             train_end="2021-01-01", val_end="2022-01-01", test_start="2022-01-01"):
    src=np.asarray(source,dtype=np.float32); tar=np.asarray(target,dtype=np.float32)
    st=np.asarray(source_times).astype("datetime64[h]")
    tt=np.asarray(target_times).astype("datetime64[h]")
    a=int(np.searchsorted(st,np.datetime64(train_end)))
    b=int(np.searchsorted(st,np.datetime64(val_end)))
    c=int(np.searchsorted(tt,np.datetime64(test_start)))
    if not(8<a<b<len(src) and 8<c<len(tar)):raise ValueError("Invalid periods")
    mean=float(src[:a].mean());std=max(float(src[:a].std()),1e-6)
    def make(arr,times,indices):
        offsets=indices[:,None]-4-np.arange(3,-1,-1)[None,:]
        frames=np.moveaxis((arr[offsets]-mean)/std,1,-1)
        hours=(times[indices].astype("datetime64[h]").astype("int64")%24).astype(np.float32)
        cycle=np.stack((np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24)),axis=-1).astype(np.float32)
        features=np.concatenate((np.broadcast_to(cycle[:,None,None,:],(*frames.shape[:3],2)),frames,
                  ((arr[indices-4]-mean)/std)[...,None]),axis=-1).astype(np.float32)
        return features,((arr[indices]-mean)/std)[...,None]
    xt,yt=make(src,st,np.arange(7,a)); xv,yv=make(src,st,np.arange(a,b))
    xe,ye=make(tar,tt,np.arange(c,len(tar)))
    params=init_model(jax.random.PRNGKey(seed),input_channels=7,zero_residual=True)
    @jax.jit
    def step(p,x,y):
        loss=lambda q:jnp.mean((predict(q,x)-y)**2)
        grads=jax.grad(loss)(p,x,y)
        return jax.tree_util.tree_map(lambda w,g:w-.001*g,p,grads)
    rng=np.random.default_rng(seed);best=np.inf;best_params=None
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(xt)),max(1,int(np.ceil(len(xt)/8)))):
            params=step(params,jnp.asarray(xt[batch]),jnp.asarray(yt[batch]))
        val=np.concatenate([np.asarray(predict(params,jnp.asarray(xv[i:i+32]))) for i in range(0,len(xv),32)])
        e=float(np.mean((val-yv)**2))
        if e<best:best=e;best_params=jax.tree_util.tree_map(lambda x:jnp.array(x),params)
    pred=np.concatenate([np.asarray(predict(best_params,jnp.asarray(xe[i:i+32])))[...,0] for i in range(0,len(xe),32)])*std+mean
    truth=ye[...,0]*std+mean
    persistence=tar[np.arange(c,len(tar))-4]
    hot=truth>=305.
    rmse=lambda v,m:float(np.sqrt(np.mean((v[m]-truth[m])**2))) if m.any() else None
    return {"protocol":"Train on original area only: recent frames + hour + persistence, no climatology at training or inference",
            "seed":seed,"source_train_normalization":True,"target_training":False,
            "target_historical_climatology":False,"source_validation_best_mse":best,
            "target_annual_cnn_rmse_kelvin":rmse(pred,np.ones_like(hot,dtype=bool)),
            "target_annual_persistence_rmse_kelvin":rmse(persistence,np.ones_like(hot,dtype=bool)),
            "target_hot_cnn_rmse_kelvin":rmse(pred,hot),
            "target_hot_persistence_rmse_kelvin":rmse(persistence,hot),
            "hot_gridcell_times":int(hot.sum())}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--target",required=True)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--output",required=True);args=p.parse_args()
    s=load_era5(args.source);t=load_era5(args.target)
    result=evaluate(s.values,s.time.values,t.values,t.time.values,args.seed,args.epochs)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
