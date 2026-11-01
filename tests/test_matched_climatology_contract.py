"""Matched training design contract (no weather downloads)."""
from pathlib import Path
import numpy as np
import pytest
from scripts.climatology_aware_cnn import run


def test_invalid_feature_mode_rejected():
    with pytest.raises(ValueError, match="climatology_feature_mode"):
        run(np.zeros((20,4,4),dtype=np.float32),
            np.datetime64("2020-01-01")+np.arange(20)*np.timedelta64(6,"h"),
            "2020-01-03","2020-01-04",climatology_feature_mode="unsupported")


def test_matched_workflow_covers_both_arms_and_seeds():
    p=Path(__file__).resolve().parents[1] / ".github/workflows/era5-matched-climatology.yml"
    text=p.read_text()
    assert "historical source_mean" in text
    assert "42 7 123" in text
    assert "frozen_spatial_eval.py" in text
    assert "workflow_dispatch:" in text


def test_matched_checkpoint_feature_contract():
    code=(Path(__file__).resolve().parents[1] / "scripts/frozen_spatial_eval.py").read_text()
    assert 'trained_feature_mode=="source_mean"' in code
    assert 'trained_climatology_feature_mode' in code
