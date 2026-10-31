import pytest
from scripts.gpu_training import require_gpu

class Device:
    def __init__(self,platform):
        self.platform=platform

def test_rejects_cpu_only_hardware():
    with pytest.raises(RuntimeError,match="NOT performed"):
        require_gpu([Device("cpu")])

def test_accepts_real_gpu_device_kind():
    devices=require_gpu([Device("gpu"),Device("cpu")])
    assert len(devices)==1 and devices[0].platform=="gpu"
