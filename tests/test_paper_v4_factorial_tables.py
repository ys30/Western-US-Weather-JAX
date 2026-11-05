"""Unit tests for archived-factorial report parsing and contrast signs."""
import json
from pathlib import Path

import pytest

from scripts.paper_v4_factorial_tables import CASES, FEATURES, REFS, SEEDS, summarize


def make_report(root, case, seed, ref, feat, *, reference_shift=0.0):
    # Features vary model error, but must not vary residual reference.
    base = 2.0 if ref == "persistence" else 1.8
    feature_gain = 0.1 if ref == "persistence" else 0.05
    model = base + (0 if feat == "source_mean" else -feature_gain)
    data = {
        "factorial_reference_mode": ref,
        "trained_climatology_feature_mode": feat,
        "climatology_mode": "target_history",
        "feature_only_perturbation": False,
        "test_rmse_kelvin": model + 1,
        "fixed_305k_cnn_rmse_kelvin": model,
        "source_weight_blend_rmse_kelvin": base + 1.2 + reference_shift,
        "fixed_305k_blend_rmse_kelvin": base + 0.2 + reference_shift,
        "fixed_hot_samples": 250,
    }
    p = root / "seed-artifacts" / f"seed-{seed}" / "outputs" / "factorial" / case / ref / feat / f"target-{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def populate(root):
    for case in CASES:
        for seed in SEEDS:
            for ref in REFS:
                for feat in FEATURES:
                    make_report(root, case, seed, ref, feat)


def test_complete_factorial_sign_and_counts(tmp_path):
    populate(tmp_path)
    rows, summaries = summarize(tmp_path)
    assert len(rows) == 80
    assert len(summaries) == 8
    for row in rows:
        assert row["delta_p_K"] == pytest.approx(0.1)
        assert row["delta_b_K"] == pytest.approx(0.05)
        assert row["interaction_K"] == pytest.approx(-0.05)
    assert all(s["interaction_negative"] == 10 for s in summaries)


def test_missing_arm_rejected(tmp_path):
    populate(tmp_path)
    p = tmp_path / "seed-artifacts" / f"seed-{SEEDS[0]}" / "outputs" / "factorial" / CASES[0] / REFS[0] / FEATURES[0] / f"target-{SEEDS[0]}.json"
    p.unlink()
    with pytest.raises(ValueError, match="missing="):
        summarize(tmp_path)


def test_baseline_mismatch_rejected(tmp_path):
    populate(tmp_path)
    make_report(tmp_path, CASES[0], SEEDS[0], "blend", "source_mean", reference_shift=0.2)
    with pytest.raises(ValueError, match="Reference differs"):
        summarize(tmp_path)


def test_duplicate_report_rejected(tmp_path):
    populate(tmp_path)
    make_report(tmp_path / "other", CASES[0], SEEDS[0], "blend", "source_mean")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path)
