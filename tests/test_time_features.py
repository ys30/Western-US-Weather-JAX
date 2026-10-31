import numpy as np
from weatherjax.data import synthetic_data, prepare_splits
def test_time_channels_known_without_future_temperature():
    arr=synthetic_data(n=100,h=8,w=8)
    times=np.datetime64('2021-01-01T00')+np.arange(100)*np.timedelta64(6,'h')
    (tr,va,te),_=prepare_splits(arr,lead_steps=4,history_steps=4,timestamps=times,time_features=True)
    assert tr[0].shape[-1]==6
    np.testing.assert_allclose(tr[0][0,:,:,0],-1.0,atol=1e-6)
    np.testing.assert_allclose(tr[0][0,:,:,1],0.0,atol=1e-6)
    assert np.isfinite(tr[0]).all()
