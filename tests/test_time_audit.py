import numpy as np
import pytest
from weatherjax.time_audit import audit_timeline,assert_source_only_inputs

def test_boundary_forecast_uses_prior_year_inputs():
    ts=np.datetime64("2020-12-27")+np.arange(40)*np.timedelta64(6,"h")
    r=audit_timeline(ts,"2021-01-01","2021-01-04")
    assert r["lead_hours"]==24
    assert r["splits"]["validation"]["first_target"]=="2021-01-01T00"
    assert r["splits"]["validation"]["first_input"]=="2020-12-30T06"
    assert r["splits"]["validation"]["target_step_minus_latest_input"]==4

def test_reject_era5_timestamp_gap():
    ts=np.datetime64("2020-01-01")+np.arange(40)*np.timedelta64(6,"h")
    ts[25:]+=np.timedelta64(6,"h")
    with pytest.raises(ValueError,match="irregular"):
        audit_timeline(ts,str(ts[20]),str(ts[29]))

def test_input_indices_never_future():
    x=assert_source_only_inputs(np.arange(7,24),40)
    assert x.shape==(17,4)
    assert np.all(x[:,-1]==np.arange(7,24)-4)
    with pytest.raises(ValueError):
        assert_source_only_inputs([3],40)
