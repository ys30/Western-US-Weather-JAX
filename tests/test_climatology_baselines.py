import numpy as np
from scripts.climatology_baselines import benchmark

def test_leakage_free_climatology():
    ts=np.datetime64("2019-01-01")+np.arange(120)*np.timedelta64(6,"h")
    a=np.full((120,2,3),280,dtype=np.float32)
    a[80:]=320
    r=benchmark(a,ts,"2019-01-13","2019-01-21",4)
    assert r["test_samples"]==40
    assert r["test_rmse_k"]["climatology"]>0
    assert r["selected_climatology_weight"] in [0,.25,.5,.75,1]

def test_future_data_does_not_affect_validation_choice():
    ts=np.datetime64("2019-01-01")+np.arange(120)*np.timedelta64(6,"h")
    a=np.arange(120,dtype=np.float32)[:,None,None]+np.ones((120,2,3),dtype=np.float32)
    b=a.copy();b[80:]+=1000
    ra=benchmark(a,ts,"2019-01-13","2019-01-21",4)
    rb=benchmark(b,ts,"2019-01-13","2019-01-21",4)
    assert ra["selected_climatology_weight"]==rb["selected_climatology_weight"]
    assert ra["validation_rmse_k"]==rb["validation_rmse_k"]
