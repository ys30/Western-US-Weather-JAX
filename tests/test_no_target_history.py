import numpy as np
import jax
from scripts.frozen_spatial_eval import frozen_evaluate
from weatherjax.model import init_model

def test_no_target_historical_climatology_ablation(tmp_path):
    ts=np.datetime64("2020-01-01")+np.arange(200)*np.timedelta64(6,"h")
    arr=np.full((200,4,5),301.,dtype=np.float32)
    weights=init_model(jax.random.PRNGKey(0),input_channels=8,zero_residual=True)
    ck=tmp_path/"source.npz"
    np.savez(ck,**{"param_"+k:np.asarray(v) for k,v in weights.items()},
             source_train_mean=np.float32(299),
             source_train_std=np.float32(10),
             source_validation_selected_blend_weight=np.float32(.25))
    r=frozen_evaluate(arr,ts,ck,str(ts[150]),climatology_mode="source_global_constant")
    assert r["strict_zero_history_transfer"]
    assert r["climatology_mode"]=="source_global_constant"
    assert r["fixed_hot_samples"]==0
    assert r["fixed_305k_cnn_rmse_kelvin"] is None
