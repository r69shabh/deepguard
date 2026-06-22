import pytest
import numpy as np
from deepguard.models import LSTMAEDetector
import tensorflow as tf

@pytest.fixture
def seq_data():
    X_train = np.random.randn(50, 10, 5).astype(np.float32)
    X_val = np.random.randn(10, 10, 5).astype(np.float32)
    return X_train, X_val

def test_lstm_init():
    model = LSTMAEDetector(window_size=10, latent_dim=16)
    assert model.window_size == 10
    assert model.latent_dim == 16
    
def test_lstm_fit(seq_data):
    X_train, X_val = seq_data
    model = LSTMAEDetector(window_size=10, latent_dim=8)
    model.fit(X_train, X_val_seq=X_val, epochs=1, batch_size=16)
    assert model.model is not None

def test_lstm_score(seq_data):
    X_train, X_val = seq_data
    model = LSTMAEDetector(window_size=10, latent_dim=8)
    model.fit(X_train, epochs=1, batch_size=16)
    scores = model.score(X_val)
    assert scores.shape == (10,)

def test_lstm_predict_raises_no_threshold(seq_data):
    X_train, X_val = seq_data
    model = LSTMAEDetector(window_size=10, latent_dim=8)
    model.fit(X_train, epochs=1, batch_size=16)
    with pytest.raises(RuntimeError):
        model.predict(X_val)
        
def test_lstm_set_threshold(seq_data):
    X_train, X_val = seq_data
    model = LSTMAEDetector(window_size=10, latent_dim=8)
    model.fit(X_train, epochs=1, batch_size=16)
    model.set_threshold(X_val, percentile=90)
    assert model.threshold is not None
    preds = model.predict(X_val)
    assert preds.shape == (10,)
