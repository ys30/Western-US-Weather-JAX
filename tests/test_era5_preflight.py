import numpy as np
import pytest
import xarray as xr
from weatherjax.preflight import inspect_era5

def make_input(path, units="K", step=1):
    t=np.datetime64("2020-01-01T00") + np.arange(48)*np.timedelta64(step,"h")
    da=xr.DataArray(np.full((48,6,8),291,dtype="float32"),
        dims=("time","latitude","longitude"),
        coords={"time":t,"latitude":np.linspace(35,31,6),
                "longitude":np.linspace(239,243,8)},attrs={"units":units})
    xr.Dataset({"t2m":da}).to_netcdf(path)

def test_era5_preflight_valid_0360_and_descending_lat(tmp_path):
    path=tmp_path/"era.nc"
    make_input(path)
    d=inspect_era5(path)
    assert (d["height"],d["width"],d["lead_steps"]) == (6,8,24)
    assert d["units"]=="K"

def test_reject_wrong_temperature_units(tmp_path):
    path=tmp_path/"bad.nc"
    make_input(path,units="C")
    with pytest.raises(ValueError,match="Kelvin"):
        inspect_era5(path)

def test_reject_timestep_mismatch(tmp_path):
    path=tmp_path/"6h.nc"
    make_input(path,step=6)
    with pytest.raises(ValueError,match="regular"):
        inspect_era5(path)
    assert inspect_era5(path,expected_step_hours=6)["lead_steps"]==4
