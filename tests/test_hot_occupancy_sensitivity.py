import numpy as np
import pytest
from scripts.hot_occupancy_sensitivity import assess

def test_run_duration_and_peak_coverage():
    ts=np.datetime64("2022-06-01")+np.arange(6)*np.timedelta64(6,"h")
    y=np.full((6,2,2),300.)
    y[0,0,0]=306.;y[1,0,1]=306.;y[2,0,:]=307.;y[5,:,:]=309.
    a=np.ones_like(y);b=np.full_like(y,2.)
    assert len(assess(y,a,b,ts))==2
    assert len(assess(y,a,b,ts,min_steps=2))==1
    assert len(assess(y,a,b,ts,min_steps=2,min_fraction=.75))==0
    assert len(assess(y,a,b,ts,min_steps=1,min_fraction=.75))==1

def test_invalid_filters_rejected():
    ts=np.datetime64("2022-06-01")+np.arange(3)*np.timedelta64(6,"h")
    y=np.full((3,2,2),300.)
    with pytest.raises(ValueError):assess(y,y,y,ts,min_fraction=1.1)
