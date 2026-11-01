import numpy as np
from scripts.climatology_aware_cnn import hot_time_block_ci, monthly_diagnostics

def test_hot_block_ci_counts_only_truth_hot_cells():
    y=np.full((24,2,2),300.,dtype=np.float32);y[:,0,0]=310
    ref=np.full_like(y,310); pred=np.full_like(y,309)
    a=hot_time_block_ci(pred,ref,y,threshold=305,block_steps=4,draws=100,seed=1)
    assert abs(a['difference_kelvin']+1)<1e-6
    assert a['hot_gridcell_times']==24
    assert a['ci95_upper_kelvin']<0

def test_monthly_hot_empty_is_null():
    y=np.full((2,2,2),290.)
    ts=np.array(['2022-01-01','2022-02-01'],dtype='datetime64[D]')
    d=monthly_diagnostics(y,y,y,ts)
    assert len(d)==2
    assert d[0]['fixed_hot_cnn_rmse_kelvin'] is None
