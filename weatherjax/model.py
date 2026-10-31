"""Small convolutional network implemented in pure JAX."""
import jax
import jax.numpy as jnp

def init_model(key, channels=12):
    ks=jax.random.split(key,3)
    def w(k,shape):
        return jax.random.normal(k,shape)*jnp.sqrt(2/(shape[0]*shape[1]*shape[2]))
    return {'w1':w(ks[0],(3,3,1,channels)), 'b1':jnp.zeros((channels,)),
            'w2':w(ks[1],(3,3,channels,channels)), 'b2':jnp.zeros((channels,)),
            'w3':w(ks[2],(1,1,channels,1)), 'b3':jnp.zeros((1,))}

def conv(x,w,b):
    return jax.lax.conv_general_dilated(x,w,(1,1),'SAME',dimension_numbers=('NHWC','HWIO','NHWC'))+b

def predict(params,x):
    z=jax.nn.relu(conv(x,params['w1'],params['b1']))
    z=jax.nn.relu(conv(z,params['w2'],params['b2']))
    return x+conv(z,params['w3'],params['b3'])
