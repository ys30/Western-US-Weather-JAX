from pathlib import Path
import numpy as np
from scripts.climatology_aware_cnn import run
from weatherjax.data import synthetic_data

def test_source_checkpoint_roundtrip(tmp_path):
    data=synthetic_data(n=180,h=6,w=7)
    ts=np.datetime64("2020-01-01")+np.arange(180)*np.timedelta64(6,"h")
    path=tmp_path/"source.npz"
    run(data,ts,str(ts[85]),str(ts[120]),epochs=1,seed=42,
        checkpoint_path=path)
    assert path.exists()
    with np.load(path) as z:
        assert set(["param_w1","param_w2","param_w3","param_b1","param_b2",
                    "param_b3","source_train_mean","source_train_std",
                    "source_validation_selected_blend_weight"]).issubset(z.files)
        assert np.isfinite(z["source_train_mean"])
