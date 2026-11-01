"""Select hot loss weight on validation only; report held-out tests descriptively."""
import argparse,json
from pathlib import Path

def choose(rows):
    if not rows: raise ValueError("Missing runs")
    by_seed={}
    for r in rows:
        seed=r['random_seed']
        if r.get('hot_weight') is None or r.get('best_validation_fixed_heat_rmse_kelvin') is None:raise ValueError("Missing validation heat metric")
        if seed not in by_seed:by_seed[seed]=[]
        by_seed[seed].append(r)
    weights={tuple(sorted(float(r['hot_weight']) for r in rr)) for rr in by_seed.values()}
    if len(weights)!=1:raise ValueError("Every seed must have identical candidate weights")
    winners={}
    for seed,rr in by_seed.items():
        ranked=sorted(rr,key=lambda r:(r['best_validation_fixed_heat_rmse_kelvin'],r['best_validation_rmse_kelvin'],r['hot_weight']))
        win=ranked[0]
        winners[str(seed)]={'selected_hot_weight':win['hot_weight'],
            'validation_heat_rmse_kelvin':win['best_validation_fixed_heat_rmse_kelvin'],
            'independent_test_overall_rmse_kelvin':win['test_rmse_kelvin'],
            'independent_test_fixed_heat_rmse_kelvin':win['fixed_305k_cnn_rmse_kelvin'],
            'test_blend_rmse_kelvin':win['blended_baseline_rmse_kelvin'],
            'test_blend_heat_rmse_kelvin':win['fixed_305k_blend_rmse_kelvin']}
    return {'selection_rule':'Lowest 2021 validation fixed-threshold hot RMSE per seed; no test-based selection',
        'candidate_weights':list(next(iter(weights))),'seeds':winners,
        'warning':'Per-seed selected weights may differ; 2022 metrics are post-selection evaluation, not hyperparameter selection.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('metrics',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args()
    res=choose([json.loads(Path(x).read_text()) for x in a.metrics])
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
