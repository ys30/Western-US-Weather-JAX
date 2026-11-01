import numpy as np
from scripts.heat_events import event_definition_sensitivity

def test_sensitivity_grid_counts_and_undefined():
    times=np.datetime64("2022-07-01")+np.arange(16)*np.timedelta64(6,"h")
    truth=np.full((16,4,4),295.,dtype=np.float32)
    truth[4:12,0,0]=310.
    pred=truth.copy();base=truth.copy()
    pred[4:12,0,0]=309.
    base[4:12,0,0]=308.
    grid=event_definition_sensitivity(pred,base,truth,times)
    assert len(grid["prespecified_sensitivity_grid"])==9
    r={(row["min_region_fraction"],row["min_consecutive_days"]):row
       for row in grid["prespecified_sensitivity_grid"]}
    assert r[(0.05,2)]["episode_count"]==1
    assert r[(0.05,2)]["episode_win_fraction"]==1.
    assert r[(0.10,2)]["episode_count"]==0
    assert r[(0.10,2)]["episode_win_fraction"] is None
    assert r[(0.05,3)]["episode_count"]==0
