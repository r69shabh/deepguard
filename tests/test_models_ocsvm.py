import pytest
import numpy as np
from deepguard.models import OCSVMDetector

def test_ocsvm_init():
    model = OCSVMDetector(nu=0.1)
    assert model.nu == 0.1
    
def test_ocsvm_fit():
    X = np.random.randn(100, 3)
    model = OCSVMDetector(train_cap=50)
    model.fit(X)
    assert model._clf is not None

def test_ocsvm_predict():
    X = np.random.randn(100, 3)
    model = OCSVMDetector(train_cap=50)
    model.fit(X)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)

def test_ocsvm_score():
    X = np.random.randn(100, 3)
    model = OCSVMDetector(train_cap=50)
    model.fit(X)
    scores = model.score(X[:10])
    assert scores.shape == (10,)
