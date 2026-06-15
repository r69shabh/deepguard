import pytest
import pandas as pd
import numpy as np
from deepguard.features import FeatureEngineer

def test_fit_transform_shape():
    df = pd.DataFrame({
        "Total Fwd Packets": np.random.randint(1, 100, 20),
        "Total Backward Packets": np.random.randint(1, 100, 20),
        "Subflow Fwd Bytes": np.random.randint(100, 10000, 20),
        "Total Length of Bwd Packets": np.random.randint(100, 10000, 20),
        "Average Packet Size": np.random.randint(50, 1500, 20),
        "Max Packet Length": np.random.randint(100, 2000, 20),
        "Flow IAT Std": np.random.rand(20) * 100,
        "Flow IAT Mean": np.random.rand(20) * 100,
        "Packet Length Variance": np.random.rand(20) * 500,
    })
    fe = FeatureEngineer()
    out = fe.fit_transform(df)
    assert out.shape[0] == 20
    assert out.shape[1] == len(fe.get_feature_names())

def test_transform_nan_handling():
    df_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    fe = FeatureEngineer(skew_threshold=10.0, corr_threshold=1.0)
    fe.fit(df_train)
    df_test = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    out = fe.transform(df_test)
    assert not np.isnan(out).any()
