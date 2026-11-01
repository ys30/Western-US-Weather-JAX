from scripts.select_climate_heat import choose
def test_selects_validation_only():
    rows=[]
    for weight,validation,test in [(0,2.3,0.1),(1,2.1,9.0),(2,2.2,0.1)]:
        rows.append(dict(random_seed=42,hot_weight=weight,best_validation_fixed_heat_rmse_kelvin=validation,
        best_validation_rmse_kelvin=3.0,test_rmse_kelvin=test,fixed_305k_cnn_rmse_kelvin=test,
        blended_baseline_rmse_kelvin=3.0,fixed_305k_blend_rmse_kelvin=2.2))
    assert choose(rows)['seeds']['42']['selected_hot_weight']==1
