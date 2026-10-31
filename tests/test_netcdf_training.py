"""Integration test: NetCDF -> crop -> time split -> JAX training -> metrics."""
import numpy as np
import xarray as xr
from weatherjax.data import synthetic_data, load_era5
from weatherjax.preflight import inspect_era5
from weatherjax.train import train

def test_netcdf_to_training(tmp_path):
    field=synthetic_data(n=80,h=8,w=10)
    hours=np.datetime64("2020-07-01T00")+np.arange(len(field))*np.timedelta64(6,"h")
    ds=xr.Dataset({"t2m":(("valid_time","latitude","longitude"),field,{"units":"K"})},
        coords={"valid_time":hours,"latitude":np.linspace(43,35,8),
                "longitude":np.linspace(238,245,10)})
    path=tmp_path/"tiny_era5_format.nc"
    ds.to_netcdf(path)
    report=inspect_era5(path,expected_step_hours=6,lead_hours=24)
    assert report["lead_steps"]==4
    values=load_era5(path).values
    _, metrics=train(values,epochs=1,batch=8,channels=4,lead_steps=4)
    assert metrics["test_samples"]>0
    assert np.isfinite(metrics["test_rmse_kelvin"])
    assert np.isfinite(metrics["persistence_test_rmse_kelvin"])
