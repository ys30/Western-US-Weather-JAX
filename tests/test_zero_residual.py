import numpy as np
import jax
import jax.numpy as jnp
from weatherjax.model import init_model, predict

def test_zero_residual_starts_exactly_at_persistence():
    x=jnp.arange(2*8*8*4,dtype=jnp.float32).reshape(2,8,8,4)/100
    params=init_model(jax.random.PRNGKey(7),channels=8,input_channels=4,zero_residual=True)
    np.testing.assert_allclose(np.asarray(predict(params,x)),np.asarray(x[...,-1:]),rtol=0,atol=0)
