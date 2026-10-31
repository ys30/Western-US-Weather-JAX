import json
from scripts.summarize_experiment import build_report

def test_comparison_export(tmp_path):
    values=[]
    for i in range(2):
        f=tmp_path/f"{i}.json"
        f.write_text(json.dumps({"test_rmse_kelvin":3.0-i*.2,"persistence_test_rmse_kelvin":3.0,
            "best_epoch":2,"group_evaluation":{"monthly":[{"month":"2021-07","cnn_rmse_k":3.0-i*.2,"persistence_rmse_k":3.0}],
            "hot_test_p95_kelvin":305,"hot_cnn_rmse_k":4.0-i*.3,"hot_persistence_rmse_k":4.0}}))
        values.append(str(f))
    result=build_report(*values)
    assert len(result["monthly"])==2
    assert result["models"][1]["rmse_skill_pct"]>0
    assert result["hot_tail"][1]["skill_pct"]>0
