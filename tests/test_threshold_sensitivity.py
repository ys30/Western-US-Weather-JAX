import numpy as np
from scripts.threshold_sensitivity import evaluate_thresholds

def test_threshold_sensitivity():
    y=np.array([[[302.,304.,306.,308.]]])
    a=np.ones_like(y);b=np.full_like(y,2.)
    out=evaluate_thresholds(y,a,b)
    assert [r["hot_gridcell_times"] for r in out]==[3,2,1]
    assert all(r["relative_improvement_percent"]==50. for r in out)

def test_empty_subset():
    y=np.full((2,2,2),300.);a=np.ones_like(y)
    r=evaluate_thresholds(y,a,a)
    assert all(x["hot_gridcell_times"]==0 and x["cnn_rmse_kelvin"] is None for x in r)
