import numpy as np
import jax
from scripts.frozen_spatial_eval import frozen_evaluate
from weatherjax.model import init_model

def test_zero_residual_source_frozen(tmp_path):
    ts=np.datetime64("2020-01-01")+np.arange(200)*np.timedelta64(6,"h")
    a=(300+np.sin(np.arange(200)/8)[:,None,None]+np.zeros((200,4,5))).astype(np.float32)
    weights=init_model(jax.random.PRNGKey(42),input_channels=8,zero_residual=True)
    ck=tmp_path/"model.npz"
    np.savez(ck,**{"param_"+k:np.asarray(v) for k,v in weights.items()},
             source_train_mean=np.float32(299),source_train_std=np.float32(10),
             source_validation_selected_blend_weight=np.float32(.25))
    result=frozen_evaluate(a,ts,ck,str(ts[150]))
    assert abs(result["test_rmse_kelvin"]-result["source_weight_blend_rmse_kelvin"])<.001
    assert result["target_history_last_timestamp"]==str(ts[149])
    assert result["strict_zero_history_transfer"] is False
