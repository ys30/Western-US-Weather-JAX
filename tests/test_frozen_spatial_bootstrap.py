import numpy as np
import jax
from scripts.frozen_spatial_eval import frozen_evaluate
from weatherjax.model import init_model

def test_frozen_bootstrap_intervals_reported(tmp_path):
    ts=np.datetime64("2020-01-01")+np.arange(200)*np.timedelta64(6,"h")
    arr=np.full((200,4,5),304.,dtype=np.float32)
    arr[148:]=306.
    weights=init_model(jax.random.PRNGKey(1),input_channels=8,zero_residual=True)
    ck=tmp_path/"checkpoint.npz"
    np.savez(ck,**{"param_"+k:np.asarray(v) for k,v in weights.items()},
             source_train_mean=np.float32(300),source_train_std=np.float32(10),
             source_validation_selected_blend_weight=np.float32(.25))
    report=frozen_evaluate(arr,ts,ck,str(ts[150]))
    assert len(report["annual_paired_block_sensitivity"])==4
    assert len(report["fixed_heat_paired_block_sensitivity"])==4
    assert all(abs(x["difference_kelvin"])<.001 for x in report["fixed_heat_paired_block_sensitivity"])
