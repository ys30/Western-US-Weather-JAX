import numpy as np
from scripts.heat_events import event_diagnostics

def test_two_day_heat_event_and_performance():
    ts=np.datetime64('2022-07-01')+np.arange(16)*np.timedelta64(6,'h')
    y=np.full((16,4,4),295.,dtype=np.float32)
    y[4:12,0,0]=310.
    b=np.full_like(y,308.);p=np.full_like(y,309.)
    r=event_diagnostics(p,b,y,ts)
    assert r['episode_count']==1
    assert r['events'][0]['days']==2
    assert r['episodes_cnn_better_hot']==1

def test_one_day_not_an_event():
    ts=np.datetime64('2022-07-01')+np.arange(12)*np.timedelta64(6,'h')
    y=np.full((12,4,4),295.,dtype=np.float32);y[4:8,0,0]=310.
    r=event_diagnostics(y,y,y,ts)
    assert r['episode_count']==0
