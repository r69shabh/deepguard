import pytest
import numpy as np
import pandas as pd
from deepguard.evaluate import per_class_detection_rate

def test_per_class_detection_rate_basic():
    y_true_mc = np.array(["BENIGN", "AttackA", "AttackB", "AttackA"])
    y_pred = np.array([0, 1, 0, 1])
    df = per_class_detection_rate(y_true_mc, y_pred)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    
    a_row = df[df["Attack Type"] == "AttackA"].iloc[0]
    assert a_row["Total"] == 2
    assert a_row["Detected"] == 2
    
    b_row = df[df["Attack Type"] == "AttackB"].iloc[0]
    assert b_row["Total"] == 1
    assert b_row["Detected"] == 0

def test_per_class_detection_rate_empty():
    y_true_mc = np.array(["BENIGN", "BENIGN"])
    y_pred = np.array([0, 0])
    df = per_class_detection_rate(y_true_mc, y_pred)
    assert len(df) == 0
