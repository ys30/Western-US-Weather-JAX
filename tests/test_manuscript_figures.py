"""Regression: mixed frozen-adaptation and constant-feature ablations."""
import json
from scripts.manuscript_figures import records

def test_excludes_nonmatching_protocol_before_duplicate_seed_check(tmp_path):
    keys=("test_rmse_kelvin","source_weight_blend_rmse_kelvin","fixed_305k_cnn_rmse_kelvin","fixed_305k_blend_rmse_kelvin")
    for seed in (7,42,123):
        sample={k:float(seed) for k in keys}
        for folder,mode in (("a_ablation","source_global_constant"),("b_adapted","target_history")):
            d=tmp_path/folder;d.mkdir(exist_ok=True)
            (d/f"seed-{seed}.json").write_text(json.dumps(dict(sample,climatology_mode=mode)))
    result=records(tmp_path,keys)
    assert sorted(result)==[7,42,123]
    assert result[42][0]==42.
