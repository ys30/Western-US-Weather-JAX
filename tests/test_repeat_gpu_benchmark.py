import json
import pytest
from scripts.repeat_gpu_benchmark import summarize

def test_repeated_gpu_summary(tmp_path):
    paths=[]
    for i,t in enumerate((20,22,21)):
        p=tmp_path/f"{i}.json"
        p.write_text(json.dumps({"input_sha256":"a"*64,"elapsed_seconds":t,"test_rmse_kelvin":3.191,
          "hardware_audit":{"training_verified_on_gpu":True,"jax_backend":"gpu","parameter_platforms":["gpu"]}}))
        paths.append(p)
    x=summarize(paths)
    assert x["runs"]==3
    assert x["elapsed_seconds_mean"]==21
    assert x["elapsed_seconds_std"]==1

def test_reject_unaudited_or_inconsistent(tmp_path):
    paths=[]
    for i in range(3):
        p=tmp_path/f"{i}.json";p.write_text(json.dumps({"input_sha256":str(i),"elapsed_seconds":20,
          "test_rmse_kelvin":3.1,"hardware_audit":{"training_verified_on_gpu":True,"jax_backend":"gpu","parameter_platforms":["gpu"]}}))
        paths.append(p)
    with pytest.raises(ValueError,match="input_sha256"):summarize(paths)
