"""Synthetic regression checks for archived NPZ conversion, no ERA5 download."""
import tempfile
from pathlib import Path
import numpy as np
from reconstruct_matched_archive_fields import reconstruct

def test_reconstruction():
    with tempfile.TemporaryDirectory() as dirname:
        a,b=(Path(dirname)/x for x in ("historical.npz","source_mean.npz"))
        time=np.array(["2022-01-01T00","2022-01-01T06"],dtype="datetime64[h]")
        truth=np.full((2,2,2),306.0,dtype=np.float32)
        common=dict(time=time,truth_kelvin=truth,reference_error_kelvin=np.ones_like(truth),
                    latitudes=np.array([35.,36.]),longitudes=np.array([-107.,-106.]))
        np.savez_compressed(a,**common,cnn_error_kelvin=np.full_like(truth,0.5))
        np.savez_compressed(b,**common,cnn_error_kelvin=np.full_like(truth,1.0))
        data=reconstruct(a,b)
        assert np.allclose(data["historical_K"],306.5)
        assert np.allclose(data["source_mean_K"],307.)
        assert np.allclose(data["baseline_K"],307.)
        bad=dict(common,reference_error_kelvin=np.full_like(truth,2.))
        np.savez_compressed(b,**bad,cnn_error_kelvin=np.ones_like(truth))
        try: reconstruct(a,b)
        except ValueError as e: assert "reference_error_kelvin" in str(e)
        else: raise AssertionError("Mismatched reference not rejected")

if __name__=="__main__":
    test_reconstruction()
    print("Archive reconstruction synthetic regression passed")
