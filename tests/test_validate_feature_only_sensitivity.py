import pytest
from scripts.validate_feature_only_sensitivity import check_pair


def sample(mode, **extra):
    x={"climatology_mode":mode,"target_test_start":"2022-01-01T00",
       "target_history_last_timestamp":"2021-12-31T18",
       "fixed_hot_samples":63954,"source_validation_blend_weight":0.25,
       "source_weight_blend_rmse_kelvin":3.79987,"fixed_305k_blend_rmse_kelvin":2.47865,
       "test_rmse_kelvin":3.79,"fixed_305k_cnn_rmse_kelvin":2.34,
       "feature_only_perturbation":mode=="source_global_feature_only",
       "reference_held_fixed":mode=="source_global_feature_only"}
    x.update(extra)
    return x


def test_identical_reference_accepted():
    result=check_pair(sample("target_history"),sample("source_global_feature_only",test_rmse_kelvin=4.2))
    assert result["test_rmse_kelvin"]["feature_only_minus_control_K"]==pytest.approx(.41)


@pytest.mark.parametrize("field,value",[
    ("source_weight_blend_rmse_kelvin",4.0),
    ("fixed_305k_blend_rmse_kelvin",3.0),
    ("fixed_hot_samples",63953),
    ("target_test_start","2023-01-01"),
])
def test_changed_reference_or_target_rejected(field,value):
    with pytest.raises(ValueError):
        check_pair(sample("target_history"),sample("source_global_feature_only",**{field:value}))


def test_wrong_protocol_rejected():
    with pytest.raises(ValueError):
        check_pair(sample("target_history"),sample("source_global_constant"))
