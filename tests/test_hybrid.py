import pytest
import numpy as np
from deepguard.models import HybridDetector, GMMDetector, LSTMAEDetector
tf = pytest.importorskip("tensorflow")
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def dummy_hybrid():
    gmm = GMMDetector(n_components=1)
    gmm.fit(np.random.randn(10, 5))
    
    lstm = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10, 5)),
        tf.keras.layers.LSTM(4, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5)),
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

def test_hybrid_predict_with_explicit_threshold(dummy_hybrid):
    X = np.random.randn(50, 5).astype(np.float32)
    preds = dummy_hybrid.predict(X, threshold=0.5)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_hybrid_evaluate(dummy_hybrid):
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    metrics = dummy_hybrid.evaluate(X, y, threshold=0.5)
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
