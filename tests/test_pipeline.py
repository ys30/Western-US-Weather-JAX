import numpy as np
import jax
import jax.numpy as jnp
from weatherjax.data import synthetic_data, prepare_splits
from weatherjax.model import init_model,predict

def test_split_no_target_overlap_and_normalize_training_only():
    arr=synthetic_data(100,8,8)
    (train,val,test),(mean,std)=prepare_splits(arr,lead_steps=4)
    assert [len(a[0]) for a in [train,val,test]]==[66,15,15]
    assert np.isclose(mean,arr[:70].mean())
    assert train[0].shape[1:]==(8,8,1)
    assert np.isfinite(test[1]).all()

def test_prediction_shape_gradient():
    p=init_model(jax.random.PRNGKey(0),4)
    x=jnp.ones((2,8,8,1))
    assert predict(p,x).shape==x.shape
    g=jax.grad(lambda params:jnp.mean(predict(params,x)**2))(p)
    assert all(np.isfinite(np.asarray(v)).all() for v in g.values())

def test_multiframe_inputs_do_not_leak_future():
    a=synthetic_data(100,8,8)
    (tr,va,te),_=prepare_splits(a,lead_steps=4,history_steps=4)
    assert tr[0].shape[-1]==4
    assert len(tr[0])==63
    assert np.allclose(tr[0][0,...,-1],(a[3]-a[:70].mean())/a[:70].std())
    assert np.allclose(tr[1][0,...,0],(a[7]-a[:70].mean())/a[:70].std())
    p=init_model(jax.random.PRNGKey(0),4,input_channels=4)
    assert predict(p,jnp.asarray(tr[0][:2])).shape==(2,8,8,1)
