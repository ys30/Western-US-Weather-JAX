import json
from scripts.select_heat_weight import select

def test_validation_selection_ignores_test_score(tmp_path):
    paths=[]
    for weight,val,test in [(0,2.4,0.1),(1,2.1,9.9),(2,2.3,0.2)]:
        path=tmp_path/f'{weight}.json'
        path.write_text(json.dumps({'hot_weight':weight,'best_validation_fixed_heat_rmse_kelvin':val,
            'best_validation_rmse_kelvin':3.0,'test_rmse_kelvin':test}))
        paths.append(path)
    result=select(paths)
    assert result['selected_hot_weight']==1
    assert 'test_rmse_kelvin' not in json.dumps(result)
