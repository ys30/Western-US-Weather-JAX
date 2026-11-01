"""Guard Colab's installation contract against NumPy/Numba regressions."""
import json
from pathlib import Path


def test_colab_smoke_uses_compatible_numpy_and_checks_dependencies():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks" / "01_colab_gpu_smoke.ipynb").read_text()
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "numpy>=1.24,<2.3" in code
    assert "-e '.[test]'" in code
    assert "pip check" in code
    assert "d.platform == 'gpu'" in code
