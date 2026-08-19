import numpy as np
import pandas as pd
from anomaly_ids.features import FeatureEngineer

def test_feature_engineer_fit_transform():
    # Create some dummy benign data
    df = pd.DataFrame({
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

    fe = FeatureEngineer()
    out = fe.fit_transform(df)

    assert fe.is_fitted_
    assert out.shape[0] == 5
    assert len(fe.get_feature_names()) > 0
    assert out.shape[1] == len(fe.get_feature_names())

def test_feature_engineer_transform_only():
    df_train = pd.DataFrame({
        "Total Fwd Packets": [10, 20],
        "Total Backward Packets": [5, 10],
        "Subflow Fwd Bytes": [1000, 2000],
        "Total Length of Bwd Packets": [500, 1000],
        "Average Packet Size": [100, 100],
        "Max Packet Length": [150, 150],
        "Flow IAT Std": [10.0, 5.0],
        "Flow IAT Mean": [100.0, 50.0],
        "Packet Length Variance": [20.0, 20.0],
    })
    
    df_test = pd.DataFrame({
        "Total Fwd Packets": [15],
        "Total Backward Packets": [7],
        "Subflow Fwd Bytes": [1500],
        "Total Length of Bwd Packets": [750],
        "Average Packet Size": [100],
        "Max Packet Length": [150],
        "Flow IAT Std": [7.5],
        "Flow IAT Mean": [75.0],
        "Packet Length Variance": [20.0],
    })

    fe = FeatureEngineer()
    fe.fit(df_train)
    out = fe.transform(df_test)
    
    assert out.shape[0] == 1
    assert out.shape[1] == len(fe.get_feature_names())
