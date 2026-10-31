import numpy as np
import pytest
from weatherjax.data import synthetic_data
from weatherjax.train import train

def test_heat_loss_default_matches_existing_training():
    a=synthetic_data(n=95,h=8,w=8)
    _,one=train(a,epochs=1,channels=4,history_steps=4)
    _,two=train(a,epochs=1,channels=4,history_steps=4,hot_weight=0)
    assert one["test_rmse_kelvin"]==two["test_rmse_kelvin"]

def test_heat_loss_rejects_negative_weight():
    with pytest.raises(ValueError,match="nonnegative"):
        train(synthetic_data(n=95,h=8,w=8),epochs=1,hot_weight=-1)
