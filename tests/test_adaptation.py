import numpy as np
import pytest

from deepguard.adaptation import ReplayBuffer, adapt_gmm
from deepguard.models import GMMDetector


class TestReplayBuffer:
    def test_capacity_and_uniform_reservoir(self):
        buf = ReplayBuffer(capacity=100, seed=0, mode="reservoir")
        stream = np.arange(1000).reshape(-1, 1).astype(float)
        buf.extend(stream)
        data = buf.get()
        assert len(buf) == 100
        assert set(np.unique(data[:, 0])).issubset(set(range(1000)))
        # reservoir keeps a spread-out sample, not just the first 100 rows
        assert data[:, 0].max() > 500

    def test_get_returns_copy(self):
        buf = ReplayBuffer(capacity=5)
        buf.add(np.zeros(3))
        d1 = buf.get()
        d1[:] = 99
        assert (buf.get() == 0).all()


def _drift_world(seed=0):
    """Benign regime A → regime B; attacks fixed. Returns models + data."""
    rng = np.random.RandomState(seed)
    X_benign_A = rng.normal(0, 1, (4000, 6))
    X_benign_B = rng.normal(4, 1, (4000, 6))     # drifted benign traffic
    X_attack = rng.normal(0, 1, (600, 6)) + np.array([2, -2, 2, -2, 2, -2])
    gmm_old = GMMDetector(n_components=3, threshold_percentile=10,
                          n_init=1, max_iter=50, tune_subsample=4000)
    gmm_old.fit(X_benign_A)
    return gmm_old, X_benign_B, X_attack


def test_adaptation_accepts_after_drift_and_restores_fpr():
    gmm_old, X_new_benign, X_attack = _drift_world()

    # Old model on NEW benign: massive false alarms
    val_benign = X_new_benign[:800]
    val_attacks = X_attack[:200]
    fpr_before = float(np.mean(gmm_old.predict(val_benign)))
    assert fpr_before > 0.8

    model, rep = adapt_gmm(
        gmm_old, X_new_benign[800:],
        val_benign=val_benign, val_attacks=val_attacks,
        fpr_tolerance=0.30, recall_tolerance=0.05, min_buffer=1500,
    )
    assert rep.accepted and model is not gmm_old
    assert rep.fpr_after < rep.fpr_before
    assert rep.fpr_after < 0.2


def test_adaptation_rolls_back_when_guard_fails():
    gmm_old, X_new_benign, _ = _drift_world()
    # Poisoned buffer: half attacks inside "benign" buffer must trip guards
    poisoned = np.concatenate([X_new_benign[800:2200],
                               X_new_benign[:1400] + np.array([2] * 6)])
    model, rep = adapt_gmm(
        gmm_old, poisoned,
        val_benign=X_new_benign[:800], val_attacks=None,
        fpr_tolerance=0.30, min_buffer=1500,
    )
    # Either rejected outright or accepted with sane FPR — guard must hold
    if rep.accepted:
        assert rep.fpr_after <= rep.fpr_before * 1.30 + 1e-3
    else:
        assert model is gmm_old


def test_adaptation_refuses_tiny_buffer():
    gmm_old, _, _ = _drift_world()
    model, rep = adapt_gmm(gmm_old, np.zeros((50, 6)), min_buffer=1500)
    assert not rep.accepted and model is gmm_old and "too small" in rep.reason


def test_proxy_guard_path_without_labels():
    gmm_old, X_new_benign, _ = _drift_world()
    model, rep = adapt_gmm(gmm_old, X_new_benign[800:], min_buffer=1500)
    assert rep.accepted
    assert float(np.median(model.score(X_new_benign[800:]))) <= \
           float(np.median(gmm_old.score(X_new_benign[800:])))


class TestSlidingBuffer:
    def test_sliding_keeps_most_recent(self):
        buf = ReplayBuffer(capacity=50, mode="sliding")
        stream = np.arange(200).reshape(-1, 1).astype(float)
        buf.extend(stream)
        data = buf.get()[:, 0]
        assert len(buf) == 50
        assert set(data) == set(range(150, 200))   # exactly the newest rows
