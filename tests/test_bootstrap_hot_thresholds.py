import numpy as np
from scripts.bootstrap_hot_thresholds import bootstrap_thresholds

def test_truth_threshold_bootstrap():
    t=np.datetime64("2022-01-01")+np.arange(12)*np.timedelta64(1,"D")
    truth=np.full((12,2,2),310.)
    cnn=np.ones_like(truth);baseline=np.full_like(truth,2.)
    rows=bootstrap_thresholds(truth,cnn,baseline,t,42,draws=20,block_days=3)
    assert len(rows)==3
    assert all(abs(r["cnn_minus_baseline_rmse_k"]+1)<1e-12 for r in rows)
    assert all(abs(r["ci_lower_k"]+1)<1e-12 for r in rows)
