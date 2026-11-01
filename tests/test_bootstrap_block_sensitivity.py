import numpy as np
from scripts.bootstrap_block_sensitivity import assess

def test_block_sensitivity_constant_difference():
    days=np.datetime64("2022-01-01")+np.arange(32)*np.timedelta64(1,"D")
    y=np.full((32,2,2),309.)
    a=np.ones_like(y);b=np.full_like(y,2.)
    rows=assess(y,a,b,days,42,blocks=(3,7,14,28),draws=20)
    assert len(rows)==12
    assert all(abs(r["ci_high_k"]+1)<1e-12 for r in rows)
