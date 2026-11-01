import numpy as np
from scripts.regional_hot_occupancy import occupancy_runs

def test_occupancy_runs_not_individual_cell_episodes():
    ts=np.datetime64("2022-07-01")+np.arange(5)*np.timedelta64(6,"h")
    y=np.full((5,2,2),300.);y[0,0,0]=306;y[1,1,1]=308;y[4,0,1]=307
    cnn=np.ones_like(y);baseline=np.full_like(y,2.)
    result=occupancy_runs(y,cnn,baseline,ts)
    assert len(result)==2
    assert result[0]["steps"]==2
    assert result[0]["hot_gridcell_times"]==2
    assert result[0]["cnn_minus_reference_k"]==-1.
