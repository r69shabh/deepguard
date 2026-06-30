import pytest
import numpy as np
from deepguard.evaluate import compute_metrics

def test_compute_metrics_all_correct():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y_true, y_pred, y_scores)
    assert m['accuracy'] == 1.0
    assert m['f1'] == 1.0
    assert m['auc_roc'] == 1.0
    assert m['tp'] == 2
    assert m['tn'] == 2
    assert m['fp'] == 0
    assert m['fn'] == 0

def test_compute_metrics_all_wrong():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    y_scores = np.array([0.9, 0.1])
    m = compute_metrics(y_true, y_pred, y_scores)
    assert m['accuracy'] == 0.0
    assert m['f1'] == 0.0
    assert m['auc_roc'] == 0.0
    
def test_compute_metrics_zero_division():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    y_scores = np.array([0.1, 0.2])
    m = compute_metrics(y_true, y_pred, y_scores)
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1'] == 0.0
