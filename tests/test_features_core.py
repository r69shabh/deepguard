import pytest
import pandas as pd
import numpy as np
from deepguard.features import FeatureEngineer

def test_fe_init():
    fe = FeatureEngineer(iqr_lower=0.05, iqr_upper=0.95)
    assert fe.iqr_lower == 0.05
    assert fe.iqr_upper == 0.95
    assert not fe.is_fitted_

def test_fe_empty_df():
    fe = FeatureEngineer()
    df = pd.DataFrame()
    with pytest.raises(Exception):
        fe.fit(df)

def test_fe_check_fitted_raises():
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError):
        fe.transform(pd.DataFrame({"A": [1]}))
