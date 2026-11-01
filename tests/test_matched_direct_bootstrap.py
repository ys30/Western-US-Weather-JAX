import numpy as np
import pytest
from scripts.matched_direct_bootstrap import compare


def sample(error):
    time=np.datetime64("2022-01-01")+np.arange(80)*np.timedelta64(6,"h")
    truth=np.full((80,2,2),306.,dtype=np.float32)
    return {"time":time,"truth_kelvin":truth,
            "cnn_error_kelvin":np.full_like(truth,error),
            "reference_error_kelvin":np.ones_like(truth),
            "threshold_kelvin":np.float64(305.)}


def test_direct_comparison_known_difference():
    result=compare(sample(2.),sample(3.),days=3,draws=100)
    for metric in ("annual","observed_hot"):
        assert result["results"][metric]["source_mean_minus_historical_rmse_K"]==pytest.approx(1.)
        assert result["results"][metric]["ci95_lower_K"]==pytest.approx(1.)


def test_alignment_check_rejects_changed_reference():
    second=sample(3.)
    second["reference_error_kelvin"][0]=2.
    with pytest.raises(ValueError,match="reference_error_kelvin"):
        compare(sample(2.),second,draws=100)


def test_misaligned_times_rejected():
    second=sample(3.)
    second["time"]=second["time"]+np.timedelta64(6,"h")
    with pytest.raises(ValueError,match="time"):
        compare(sample(2.),second,draws=100)


def test_block_lengths_do_not_change_point_estimate():
    historical=sample(2.)
    source_mean=sample(3.)
    estimates=[]
    for days in (3,7,14,28):
        result=compare(historical,source_mean,days=days,draws=100)
        estimates.append(result["results"]["observed_hot"]["source_mean_minus_historical_rmse_K"])
    assert estimates==pytest.approx([1.,1.,1.,1.])
