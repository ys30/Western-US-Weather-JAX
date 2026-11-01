import numpy as np
from scripts.climatology_aware_cnn import run

def test_climatology_aware_cnn_smoke():
    n=120
    ts=np.datetime64('2019-01-01')+np.arange(n)*np.timedelta64(6,'h')
    a=np.full((n,8,8),290,dtype=np.float32)
    a+=np.sin(np.arange(n)[:,None,None]/5).astype(np.float32)
    r=run(a,ts,'2019-01-13','2019-01-21',epochs=1,channels=4)
    assert r['best_epoch']==1
    assert np.isfinite(r['test_rmse_kelvin'])
    assert r['selected_blend_climatology_weight'] in (0,.25,.5,.75,1)
