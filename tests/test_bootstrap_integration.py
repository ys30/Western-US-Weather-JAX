import numpy as np
from weatherjax.data import synthetic_data
from weatherjax.train import train, paired_time_block_ci

def test_held_out_group_bootstrap_uses_kelvin_predictions():
    data=synthetic_data(n=100,h=8,w=8)
    _,metrics,details=train(data,epochs=1,channels=4,history_steps=4,return_predictions=True,train_end=50,val_end=70)
    result=paired_time_block_ci(details["pred_kelvin"],details["persistence_kelvin"],details["truth_kelvin"],draws=20)
    assert np.isfinite(result["rmse_difference_kelvin"])
    assert result["ci95_lower_kelvin"]<=result["ci95_upper_kelvin"]
