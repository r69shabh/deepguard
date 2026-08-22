"""
Synthetic learning checks for the deep detectors.

Each test trains on a tiny benign cluster and evaluates against shifted
anomalies — a task trivially solvable if the model learns anything. These
guard against silent training collapse (the original LSTM-AE failure mode).
TF-gated: skipped automatically when TensorFlow is unavailable.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from deepguard.models_deep import (
    TransformerAEDetector,
    USADEtector,
    DeepSVDDetector,
    create_detector,
)
from deepguard.sequences import build_train_windows


def _synthetic_flow_task(n_benign=600, n_attack=120, seed=0):
    rng = np.random.RandomState(seed)
    X_benign = rng.normal(0, 1.0, size=(n_benign, 6)).astype(np.float32)
    X_attack = (rng.normal(0, 1.0, size=(n_attack, 6)) + np.array([4, -4, 3, -3, 5, -5])).astype(np.float32)
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    return X_benign, X_attack, y


def _window_task(X_benign, X_attack, W=8):
    Xtr = build_train_windows(X_benign, window_size=W, stride=2)
    _, cov_tr = None, None
    # eval windows: benign windows + attack windows
    from deepguard.sequences import build_eval_windows

    win_b, cov_b = build_eval_windows(X_benign[:200], W, stride=W)
    win_a, cov_a = build_eval_windows(X_attack, W, stride=W)
    X_seq = np.concatenate([win_b, win_a])
    y_seq = np.concatenate([np.zeros(len(win_b), dtype=int), np.ones(len(win_a), dtype=int)])
    return Xtr, X_seq, y_seq


def test_transformer_ae_learns_synthetic_anomalies():
    Xb, Xa, _ = _synthetic_flow_task()
    Xtr, X_seq, y_seq = _window_task(Xb, Xa)
    det = TransformerAEDetector(window_size=8, d_model=16, num_heads=2,
                                latent_dim=8, dropout_rate=0.0, lr=1e-3)
    det.fit(Xtr, epochs=5, batch_size=64, patience=5)
    det.set_threshold(Xtr, percentile=90)
    m = det.evaluate(X_seq, y_seq)
    assert m["auc"] > 0.85, f"Transformer-AE failed to learn: {m}"


def test_usad_learns_synthetic_anomalies():
    Xb, Xa, _ = _synthetic_flow_task()
    Xtr, X_seq, y_seq = _window_task(Xb, Xa)
    det = USADEtector(window_size=8, hidden_dim=32, latent_dim=8, alpha=0.5, lr=1e-3)
    det.fit(Xtr, epochs=3, batch_size=64)
    det.set_threshold(Xtr, percentile=90)
    m = det.evaluate(X_seq, y_seq)
    assert m["auc"] > 0.80, f"USAD failed to learn: {m}"


def test_deep_svdd_learns_synthetic_anomalies():
    Xb, Xa, y = _synthetic_flow_task()
    det = DeepSVDDetector(hidden_dim=32, latent_dim=4, lr=1e-3, epochs=10, batch_size=128)
    det.fit(Xb[:500], Xb[500:])
    det.set_threshold(Xb[:500], percentile=95)
    m = det.evaluate(np.concatenate([Xb[500:], Xa]),
                     np.concatenate([np.zeros(100, int), np.ones(len(Xa), int)]))
    assert m["auc"] > 0.9, f"DeepSVDD failed to learn: {m}"


def test_lstm_ae_score_flows_shapes_and_learning():
    from deepguard.models import LSTMAEDetector

    Xb, Xa, y = _synthetic_flow_task(n_benign=400, n_attack=80)
    Xtr = build_train_windows(Xb, window_size=8, stride=2)
    det = LSTMAEDetector(window_size=8, latent_dim=8, dropout_rate=0.0, lr=1e-3)
    det.fit(Xtr, epochs=5, batch_size=64, patience=5)

    pool = np.concatenate([Xb[:200], Xa])
    yy = np.concatenate([np.zeros(200, int), np.ones(len(Xa), int)])
    scores = det.score_flows(pool, stride=1)
    assert scores.shape == (len(pool),)
    assert np.isfinite(scores).all()

    det.set_flow_threshold(Xb[200:], percentile=95, stride=1)
    m = det.evaluate_flows(pool, yy, stride=1)
    assert m["auc"] > 0.85, f"LSTM-AE flow-level scoring failed: {m}"


def test_factory_roundtrip(tmp_path):
    Xb, Xa, _ = _synthetic_flow_task(n_benign=300, n_attack=50)
    det = create_detector("deep_svdd", hidden_dim=16, latent_dim=4, epochs=3)
    det.fit(Xb)
    p = tmp_path / "svdd"
    det.save(p)
    loaded = create_detector("deep_svdd").load(p)
    np.testing.assert_allclose(det.score(Xa), loaded.score(Xa), rtol=1e-4)


def test_factory_unknown_name_raises():
    with pytest.raises(KeyError):
        create_detector("nope")
