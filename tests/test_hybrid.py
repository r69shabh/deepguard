import pytest
import numpy as np
from deepguard.models import HybridDetector, GMMDetector, LSTMAEDetector
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def dummy_hybrid():
    gmm = GMMDetector(n_components=1)
    gmm.fit(np.random.randn(10, 5))
    
    lstm = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10, 5)),
        tf.keras.layers.LSTM(4, return_sequences=False),
        tf.keras.layers.Dense(50)
    ])
    
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(np.random.randn(10, 2), np.random.randint(0, 2, 10))
    
    return HybridDetector(
        gmm_model=gmm._clf,
        lstm_ae_model=lstm,
        meta_learner=rf,
        window_size=10
    )

def test_hybrid_init(dummy_hybrid):
    assert dummy_hybrid.window_size == 10
    assert dummy_hybrid.meta_learner is not None

def test_hybrid_score(dummy_hybrid):
    X = np.random.randn(50, 5).astype(np.float32)
    scores = dummy_hybrid.score(X)
    assert scores.shape == (50,)
    assert not np.isnan(scores).any()

def test_hybrid_predict_no_threshold(dummy_hybrid):
    X = np.random.randn(50, 5).astype(np.float32)
    with pytest.raises(RuntimeError):
        dummy_hybrid.predict(X)
        
def test_hybrid_set_threshold(dummy_hybrid):
    X = np.random.randn(50, 5).astype(np.float32)
    dummy_hybrid.set_threshold(X, percentile=90)
    assert dummy_hybrid.threshold is not None
    preds = dummy_hybrid.predict(X)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})
