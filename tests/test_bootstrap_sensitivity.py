import numpy as np
from weatherjax.train import paired_time_block_ci

def test_block_length_sensitivity_is_deterministic_and_finite():
    truth=np.zeros((80,4,4),dtype=np.float32)
    reference=np.ones_like(truth)
    pred=np.broadcast_to(np.linspace(.7,1.2,80,dtype=np.float32)[:,None,None],truth.shape)
    for block in (4,12,28,56):
        x=paired_time_block_ci(pred,reference,truth,block_steps=block,draws=30,seed=42)
        y=paired_time_block_ci(pred,reference,truth,block_steps=block,draws=30,seed=42)
        assert x==y
        assert x["ci95_lower_kelvin"] <= x["ci95_upper_kelvin"]
