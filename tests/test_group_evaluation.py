import numpy as np
from weatherjax.data import synthetic_data
from weatherjax.train import train

def test_predictions_align_with_held_out_targets():
    data=synthetic_data(n=100,h=8,w=8)
    _,metrics,details=train(data,epochs=1,channels=4,history_steps=4,return_predictions=True,train_end=50,val_end=70)
    idx=details['target_indices']
    assert idx[0]==70 and idx[-1]==99
    np.testing.assert_allclose(details['truth_kelvin'][0,...,0],data[70],rtol=2e-6)
    np.testing.assert_allclose(details['persistence_kelvin'][0,...,0],data[66],rtol=2e-6)
    assert metrics['test_samples']==30
