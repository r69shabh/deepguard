import pytest
import numpy as np
from deepguard.models import IsolationForestDetector

def test_iforest_init():
    model = IsolationForestDetector(n_estimators=20)
    assert model.n_estimators == 20
    
def test_iforest_fit():
    X = np.random.randn(100, 3)
    model = IsolationForestDetector(n_estimators=10)
    model.fit(X)
    assert model._clf is not None

def test_iforest_predict():
    X = np.random.randn(100, 3)
    model = IsolationForestDetector(n_estimators=10)
    model.fit(X)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)

def test_iforest_score():
    X = np.random.randn(100, 3)
    model = IsolationForestDetector(n_estimators=10)
    model.fit(X)
    scores = model.score(X[:10])
    assert scores.shape == (10,)
