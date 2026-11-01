import numpy as np
from scripts.source_only_transfer import evaluate

def test_source_only_reports_paired_bootstrap():
    n=80
    ts=np.datetime64("2020-01-01")+np.arange(n)*np.timedelta64(6,"h")
    src=np.full((n,4,5),300.,dtype=np.float32)
    tar=np.full((n,3,4),306.,dtype=np.float32)
    result=evaluate(src,ts,tar,ts,seed=2,epochs=1,
                    train_end=str(ts[32]),val_end=str(ts[48]),test_start=str(ts[50]))
    assert len(result["annual_paired_block_sensitivity"])==4
    assert len(result["fixed_hot_paired_block_sensitivity"])==4
    assert all(item["ci95_lower_kelvin"]<=item["ci95_upper_kelvin"]
               for item in result["fixed_hot_paired_block_sensitivity"])
