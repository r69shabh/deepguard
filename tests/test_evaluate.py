import numpy as np
from deepguard.evaluate import compute_metrics, per_class_detection_rate

def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_scores = np.array([0.1, 0.6, 0.8, 0.9, 0.4])
    
    metrics = compute_metrics(y_true, y_pred, y_scores)
    
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.6
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc_roc" in metrics

def test_per_class_detection_rate():
    y_true_mc = np.array(["BENIGN", "BENIGN", "DDoS", "DDoS", "DoS Hulk"])
    y_pred = np.array([0, 1, 1, 1, 0])
    
    df = per_class_detection_rate(y_true_mc, y_pred)
    
    assert len(df) == 2
    assert list(df["Attack Type"].values) == ["DDoS", "DoS Hulk"]
    
    ddos_row = df[df["Attack Type"] == "DDoS"].iloc[0]
    assert ddos_row["Total"] == 2
    assert ddos_row["Detected"] == 2
    assert ddos_row["Missed"] == 0
    
    dos_row = df[df["Attack Type"] == "DoS Hulk"].iloc[0]
    assert dos_row["Total"] == 1
    assert dos_row["Detected"] == 0
    assert dos_row["Missed"] == 1
