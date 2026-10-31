import numpy as np
import pytest
from weatherjax.train import paired_time_block_ci

def test_block_bootstrap_identical_has_zero_difference():
    truth=np.zeros((24,3,3))
    ref=np.ones_like(truth)
    out=paired_time_block_ci(ref,ref,truth,draws=50)
    assert out["rmse_difference_kelvin"]==0
    assert out["ci95_lower_kelvin"]==0
    assert out["ci95_upper_kelvin"]==0

def test_block_bootstrap_perfect_forecast_is_better():
    truth=np.zeros((24,3,3))
    reference=np.ones_like(truth)
    out=paired_time_block_ci(truth,reference,truth,draws=50)
    assert out["ci95_upper_kelvin"]<0

def test_block_bootstrap_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        paired_time_block_ci(np.zeros((10,2)),np.zeros((10,3)),np.zeros((10,2)))
