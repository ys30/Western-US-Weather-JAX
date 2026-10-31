import pytest
from scripts.compare_gpu_cpu import compare

def valid():
    base={"epochs_requested":40,"batch_size":8,"learning_rate":0.001,"channels":12,"random_seed":42,"lead_steps":4,"history_steps":4,"hot_weight":2,"hot_threshold_kelvin":305,
        "train_end_index":100,"validation_end_index":200,"zero_residual":True,
        "time_features":True,"train_samples":90,"validation_samples":100,
        "test_samples":100,"test_rmse_kelvin":3.1,"elapsed_seconds":120}
    gpu={**base,"test_rmse_kelvin":3.11,"elapsed_seconds":50,
        "hardware_audit":{"training_verified_on_gpu":True,"gpu_devices":["GPU:0"],"parameter_devices":["GPU:0"],"parameter_platforms":["gpu"],"jax_backend":"gpu"}}
    return base,gpu

def test_parity_comparison_with_complete_audit():
    cpu,gpu=valid();result=compare(cpu,gpu)
    assert result["rmse_parity_pass"] and result["elapsed_speedup_cpu_over_gpu"]==2.4

def test_reject_gpu_without_audit():
    cpu,gpu=valid();gpu.pop("hardware_audit")
    with pytest.raises(ValueError,match="audit"): compare(cpu,gpu)

def test_reject_mismatched_splits():
    cpu,gpu=valid();gpu["validation_end_index"]=201
    with pytest.raises(ValueError,match="validation_end_index"):compare(cpu,gpu)

def test_reject_large_accuracy_difference():
    cpu,gpu=valid();gpu["test_rmse_kelvin"]=4.0
    assert compare(cpu,gpu)["rmse_parity_pass"] is False

def test_reject_false_gpu_platform_claim():
    cpu,gpu=valid();gpu["hardware_audit"]["parameter_platforms"]=["cpu"]
    with pytest.raises(ValueError,match="platform"): compare(cpu,gpu)

def test_reject_cpu_backend():
    cpu,gpu=valid();gpu["hardware_audit"]["jax_backend"]="cpu"
    with pytest.raises(ValueError,match="backend"): compare(cpu,gpu)

def test_reject_different_model_or_optimization_settings():
    cpu,gpu=valid()
    for key,new in (("channels",24),("learning_rate",0.002),("random_seed",7),("epochs_requested",20),("batch_size",16)):
        copy={**gpu,key:new}
        with pytest.raises(ValueError,match=key):compare(cpu,copy)
