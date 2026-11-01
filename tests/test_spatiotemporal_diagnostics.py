import csv
import numpy as np
import pytest
from scripts.spatiotemporal_diagnostics import export_diagnostics

def test_diagnostics_outputs_real_paired_aggregation(tmp_path):
    ts=np.arange(8).astype("timedelta64[h]")*6+np.datetime64("2022-01-01")
    truth=np.full((8,2,3),300.,dtype="float32");truth[2:4,0,1]=310.
    pred=truth+2;ref=truth+3
    p=tmp_path/"fields.npz"
    export_diagnostics(p,pred,ref,truth,ts)
    with np.load(p) as a:
        assert a["cnn_error_kelvin"].shape==(8,2,3)
        assert int(a["hot_mask"].sum())==2
    with p.with_suffix(".monthly.csv").open() as f:
        row=list(csv.DictReader(f))[0]
    assert float(row["cnn_rmse_k"])==pytest.approx(2.)
    assert float(row["reference_hot_rmse_k"])==pytest.approx(3.)
    with p.with_suffix(".grid.csv").open() as f:
        cells=list(csv.DictReader(f))
    assert len(cells)==6
    assert int(cells[1]["hot_sample_count"])==2

def test_diagnostics_refuse_bad_alignment(tmp_path):
    ts=np.arange(4).astype("timedelta64[h]")*6+np.datetime64("2022-01-01")
    a=np.zeros((4,2,2))
    with pytest.raises(ValueError):
        export_diagnostics(tmp_path/"bad.npz",a,a,a[:3],ts)

def test_diagnostics_coordinates(tmp_path):
    ts=np.arange(4).astype("timedelta64[h]")*6+np.datetime64("2022-01-01")
    a=np.full((4,2,2),305.,dtype="float32")
    p=tmp_path/"geo.npz"
    export_diagnostics(p,a,a,a,ts,latitudes=[31.,31.25],longitudes=[-110.,-109.75])
    with np.load(p) as z:
        assert z["latitudes"].tolist()==[31.,31.25]
    with p.with_suffix(".grid.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert rows[0]["latitude"]=="31.0" and rows[0]["longitude"]=="-110.0"
    with pytest.raises(ValueError):
        export_diagnostics(tmp_path/"invalid.npz",a,a,a,ts,latitudes=[31.])
