import json
from scripts.summarize_experiment import build_report

def test_fixed_heat_subsets_both_models(tmp_path):
    paths=[]
    for i in range(2):
        m={'test_rmse_kelvin':3.1,'persistence_test_rmse_kelvin':3.2,'best_epoch':5,
           'fixed_heat_evaluation':{'threshold_kelvin':305.0,'gridcell_count':22,
           'cnn_rmse_k':2.3-i*.1,'persistence_rmse_k':2.0}}
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(m));paths.append(str(p))
    result=build_report(*paths)
    assert len(result['fixed_heat'])==2
    assert result['fixed_heat'][0]['threshold_kelvin']==305
    assert result['fixed_heat'][1]['cnn_rmse_k'] < result['fixed_heat'][0]['cnn_rmse_k']
