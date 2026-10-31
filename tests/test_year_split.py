import numpy as np
from weatherjax.data import synthetic_data,prepare_splits

def test_year_split_targets_and_training_normalization():
    a=synthetic_data(n=120,h=8,w=8)
    (tr,va,te),(mean,std)=prepare_splits(a,lead_steps=4,history_steps=4,train_end=40,val_end=80)
    assert [len(x[0]) for x in (tr,va,te)]==[33,40,40]
    np.testing.assert_allclose(mean,a[:40].mean(),rtol=1e-6)
    assert np.isfinite(te[0]).all()
