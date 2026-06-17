import pytest
import numpy as np
from deepguard.models import GMMDetector

def test_gmm_init():
    model = GMMDetector(n_components=5)
    assert model.n_components == 5
    assert model.covariance_type == "full"
    
def test_gmm_fit():
    X = np.random.randn(100, 5)
    model = GMMDetector(n_components=2)
    model.fit(X)
    assert model._clf is not None
    assert model.threshold is not None

def test_gmm_predict():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    model = GMMDetector(n_components=2)
    model.fit(X_train)
    preds = model.predict(X_test)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_gmm_score():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    model = GMMDetector(n_components=2)
    model.fit(X_train)
    scores = model.score(X_test)
    assert scores.shape == (10,)
    
def test_gmm_evaluate():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    y_test = np.random.randint(0, 2, 10)
    model = GMMDetector(n_components=2)
    model.fit(X_train)
    metrics = model.evaluate(X_test, y_test)
    assert "f1" in metrics
    assert "auc_roc" in metrics
