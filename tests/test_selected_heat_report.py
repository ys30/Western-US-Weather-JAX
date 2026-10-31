import json
from scripts.report_selected_heat import report

def test_selection_report_uses_only_selected_weight(tmp_path):
    sel=tmp_path/"selection.json";sel.write_text(json.dumps({"selected_hot_weight":2.0}))
    paths=[]
    for weight in (0,1,2):
        p=tmp_path/f"{weight}.json"
        p.write_text(json.dumps({"hot_weight":weight,"test_rmse_kelvin":3+weight/100,
           "persistence_test_rmse_kelvin":3.2,
           "fixed_heat_evaluation":{"gridcell_count":10,"threshold_kelvin":305,
              "cnn_rmse_k":2.2-weight/100,"persistence_rmse_k":2.0}}))
        paths.append(p)
    out=report(sel,paths)
    assert out["selected_hot_weight"]==2
    assert out["selected_test_heat_rmse_kelvin"]==2.18
    assert out["heat_skill_pct"]<0
