import pytest
from scripts.gpu_training import require_gpu

class FakeDevice:
    def __init__(self,platform): self.platform=platform

def test_requires_real_backend_not_cpu():
    with pytest.raises(RuntimeError,match="NOT performed"):
        require_gpu([FakeDevice("cpu")])

def test_gpu_filter_respects_platform():
    result=require_gpu([FakeDevice("cpu"),FakeDevice("gpu")])
    assert len(result)==1 and result[0].platform=="gpu"
