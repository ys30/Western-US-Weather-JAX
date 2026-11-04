"""Synthetic whole-chain regression: reconstructed fields -> stations -> ASOS matching -> metrics.

Uses no external downloads and no real weather-skill claims.
"""
import tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_matched_archive_fields import reconstruct
from sample_forecast_grid_at_stations import sample
from match_asos_prediction_times import match
from evaluate_paired_station_predictions import metrics, daily_block_interval

def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        time=np.arange(np.datetime64("2022-07-01T00","h"),
                       np.datetime64("2022-07-16T00","h"),np.timedelta64(6,"h"))
        n=len(time)
        grid=np.full((n,2,2),306.,dtype=np.float32)
        common=dict(time=time,truth_kelvin=grid,reference_error_kelvin=np.full_like(grid,2.),
                    latitudes=np.array([35.,36.]),longitudes=np.array([-107.,-106.]))
        h=root/"h.npz"; s=root/"s.npz"
        np.savez_compressed(h,**common,cnn_error_kelvin=np.full_like(grid,0.5))
        np.savez_compressed(s,**common,cnn_error_kelvin=np.full_like(grid,1.))
        ds=reconstruct(h,s)
        st=pd.DataFrame({"station":["ABQ"],"station_lat":[35.5],"station_lon":[-106.5]})
        sampled=sample(ds,st)
        sampled["timestamp_utc"]=pd.to_datetime(sampled.timestamp_utc,utc=True)
        obs=pd.DataFrame({"station":["ABQ"]*n,
             "timestamp_utc":sampled.timestamp_utc+pd.Timedelta(minutes=10),
             "temperature_K":[306.]*n})
        good,counts=match(sampled,obs,30)
        assert len(good)==n and counts[0]["matched"]==n
        m=metrics(good)
        assert np.isclose(m["historical_rmse_K"],0.5)
        assert np.isclose(m["source_mean_rmse_K"],1.)
        assert np.isclose(m["baseline_rmse_K"],2.)
        assert np.isclose(m["delta_source_mean_minus_historical_rmse_K"],0.5)
        ci=daily_block_interval(good,days=2,draws=40,hot_threshold=305)
        assert ci and ci["ci95_low_K"]>0.49
        try:
            match(sampled,obs,5)
        except ValueError as exc:
            assert "No predictions" in str(exc)
        else:
            raise AssertionError("Narrow time tolerance should eliminate observations")
        print("Synthetic full NPZ-to-station QA passed")

if __name__=="__main__":
    main()
