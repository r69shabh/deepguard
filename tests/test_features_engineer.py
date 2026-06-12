import pytest
import pandas as pd
import numpy as np
from deepguard.features import FeatureEngineer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Total Fwd Packets": [10, 20, 10, 5, 100],
        "Total Backward Packets": [5, 10, 5, 2, 50],
        "Subflow Fwd Bytes": [1000, 2000, 1000, 500, 10000],
        "Total Length of Bwd Packets": [500, 1000, 500, 200, 5000],
        "Average Packet Size": [100, 100, 100, 100, 100],
        "Max Packet Length": [150, 150, 150, 150, 150],
        "Flow IAT Std": [10.0, 5.0, 12.0, 2.0, 50.0],
        "Flow IAT Mean": [100.0, 50.0, 120.0, 20.0, 500.0],
        "Packet Length Variance": [20.0, 20.0, 20.0, 20.0, 20.0],
    })

def test_engineer_features_adds_columns(sample_df):
    fe = FeatureEngineer()
    # Mock some internals to test private method
    fe._raw_feature_names = list(sample_df.columns)
    eng = fe._engineer_features(sample_df)
    assert "feat_fwd_bwd_pkt_ratio" in eng.columns
    assert "feat_payload_ratio" in eng.columns
    
def test_apply_caps(sample_df):
    fe = FeatureEngineer(iqr_lower=0.1, iqr_upper=0.9)
    fe.fit(sample_df)
    capped = fe._apply_caps(sample_df)
    assert capped["Total Fwd Packets"].max() < 100
    assert capped["Total Fwd Packets"].min() >= 5
    
def test_get_feature_names(sample_df):
    fe = FeatureEngineer()
    fe.fit(sample_df)
    names = fe.get_feature_names()
    assert isinstance(names, list)
    assert len(names) > 0
