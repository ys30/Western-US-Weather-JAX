from pathlib import Path

def test_gpu_workflow_is_opt_in_and_requires_physical_gpu():
    workflow=Path(".github/workflows/era5-gpu-acceptance.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "    runs-on: [self-hosted, linux, x64, gpu]" in workflow
    assert "nvidia-smi --query-gpu=" in workflow
    assert "scripts/gpu_training.py" in workflow
    assert "scripts/compare_gpu_cpu.py" in workflow
    assert "JAX_PLATFORMS=cpu" in workflow
    assert "JAX_PLATFORMS=cuda,cpu" in workflow
