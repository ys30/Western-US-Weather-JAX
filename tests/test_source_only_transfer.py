import numpy as np
from scripts.source_only_transfer import evaluate

def test_source_only_uses_no_target_climatology():
    n=75
    ts=np.datetime64("2020-01-01")+np.arange(n)*np.timedelta64(6,"h")
    src=np.full((n,4,5),300.,dtype=np.float32)
    tar=np.full((n,3,4),302.,dtype=np.float32)
    r=evaluate(src,ts,tar,ts,seed=2,epochs=1,train_end=str(ts[32]),val_end=str(ts[48]),test_start=str(ts[50]))
    assert r["target_historical_climatology"] is False
    assert r["target_training"] is False
    assert r["target_annual_cnn_rmse_kelvin"] is not None
