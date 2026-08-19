import numpy as np
from anomaly_ids.models import IsolationForestDetector, OCSVMDetector, GMMDetector

def test_isolation_forest():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    
    model = IsolationForestDetector(n_estimators=10)
    model.fit(X_train)
    
    preds = model.predict(X_test)
    scores = model.score(X_test)
    
    assert preds.shape == (10,)
    assert scores.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_ocsvm():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    
    model = OCSVMDetector(train_cap=50)
    model.fit(X_train)
    
    preds = model.predict(X_test)
    scores = model.score(X_test)
    
    assert preds.shape == (10,)
    assert scores.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_gmm():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(10, 5)
    
    model = GMMDetector(n_components=2, tune_subsample=50)
    model.fit(X_train)
    
    preds = model.predict(X_test)
    scores = model.score(X_test)
    
    assert model.threshold is not None
    assert preds.shape == (10,)
    assert scores.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})
