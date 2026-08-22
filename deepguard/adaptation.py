"""
deepguard/adaptation.py
=======================
Online adaptation after drift alarms.

Components
----------
ReplayBuffer : reservoir-sampled store of recent high-confidence benign flows
    (processed feature space). Reservoir sampling keeps a uniform sample of
    the stream under bounded memory.

adapt_gmm : refits a GMMDetector on the buffer and ACCEPTS or ROLLS BACK the
    new model based on validation guards:
      - with labeled validation data: FPR on benign must not grow beyond
        ``fpr_tolerance`` (relative), recall must not drop beyond
        ``recall_tolerance``;
      - without labels (proxy guard): the adapted model must score the recent
        benign buffer at least as low as the old model did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer (Vitter's Algorithm R)
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Bounded store of recently-admitted benign flows.

    mode="sliding" (default): keeps the MOST RECENT ``capacity`` rows —
        the right bias for adaptation, where the new regime must dominate.
    mode="reservoir": uniform sample of the whole history (Vitter's Algorithm
        R) — useful when the stream is stationary and you want long-run
        coverage instead of recency.
    """

    def __init__(self, capacity: int = 20_000, seed: int = 42,
                 mode: str = "sliding") -> None:
        if mode not in ("sliding", "reservoir"):
            raise ValueError("mode must be 'sliding' or 'reservoir'")
        self.capacity = int(capacity)
        self.mode = mode
        self._rng = np.random.RandomState(seed)
        self._store = np.zeros((0, 0))
        self._seen = 0

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x).reshape(1, -1)
        if self._store.size == 0:
            self._store = np.empty((self.capacity, x.shape[1]), dtype=np.float64)
            self._store[:] = np.nan
            self._filled = 0

        if self._filled < self.capacity:
            self._store[self._filled] = x[0]
            self._filled += 1
        elif self.mode == "sliding":
            # ring buffer: overwrite oldest
            self._store[self._seen % self.capacity] = x[0]
        else:  # reservoir
            j = self._rng.randint(0, self._seen + 1)
            if j < self.capacity:
                self._store[j] = x[0]
        self._seen += 1

    def extend(self, X: np.ndarray) -> None:
        for row in np.asarray(X):
            self.add(row)

    def get(self) -> np.ndarray:
        return self._store[: self._filled].copy()

    def clear(self) -> None:
        """Drop all contents (e.g., on regime change: pre-shift rows would
        dilute the new-regime sample the adapter needs)."""
        if self._store.size:
            self._store[:] = np.nan
        self._filled = 0
        self._seen = 0

    def __len__(self) -> int:
        return self._filled


# ──────────────────────────────────────────────────────────────────────────────
# GMM adaptation with rollback guards
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptationReport:
    accepted: bool
    reason: str
    n_buffer: int
    fpr_before: Optional[float] = None
    fpr_after: Optional[float] = None
    recall_before: Optional[float] = None
    recall_after: Optional[float] = None


def adapt_gmm(
    gmm_detector,
    X_buffer: np.ndarray,
    val_benign: Optional[np.ndarray] = None,
    val_attacks: Optional[np.ndarray] = None,
    fpr_tolerance: float = 0.30,
    recall_tolerance: float = 0.05,
    min_buffer: int = 1500,
    seed: int = 42,
) -> Tuple[object, AdaptationReport]:
    """
    Refit ``gmm_detector``'s architecture on buffered benign flows; gate by
    validation guards; roll back to the original model when guards fail.

    Returns (model_to_use, AdaptationReport).
    """
    from deepguard.models import GMMDetector

    X_buffer = np.asarray(X_buffer)
    if len(X_buffer) < max(min_buffer, 10):
        return gmm_detector, AdaptationReport(
            accepted=False, reason="buffer too small", n_buffer=len(X_buffer)
        )

    # Rebuild with the same hyperparameters as the live model
    candidate = GMMDetector(
        n_components=gmm_detector.n_components,
        covariance_type=gmm_detector.covariance_type,
        threshold_percentile=gmm_detector.threshold_percentile,
        n_init=gmm_detector.n_init,
        max_iter=gmm_detector.max_iter,
        random_state=seed,
        tune_subsample=gmm_detector.tune_subsample,
    )
    candidate.fit(X_buffer)

    report = AdaptationReport(accepted=False, reason="", n_buffer=len(X_buffer))

    if val_benign is not None and len(val_benign):
        fpr_before = _fpr(gmm_detector, val_benign)
        fpr_after = _fpr(candidate, val_benign)
        report.fpr_before, report.fpr_after = fpr_before, fpr_after

        recall_ok = True
        if val_attacks is not None and len(val_attacks):
            rec_before = _recall(gmm_detector, val_attacks, val_benign)
            rec_after = _recall(candidate, val_attacks, val_benign)
            report.recall_before, report.recall_after = rec_before, rec_after
            recall_ok = rec_after >= rec_before - recall_tolerance

        fpr_ok = fpr_after <= max(fpr_before * (1 + fpr_tolerance),
                                  fpr_before + 1e-3)

        # Saturation guard: if validation benign is fully flagged before AND
        # after, the refit learned nothing useful (buffer lacks new-regime
        # evidence) — reject and wait for more data.
        if fpr_before >= 0.999 and fpr_after >= 0.999:
            report.reason = ("validation benign still fully flagged; "
                             "insufficient new-regime evidence")
            return gmm_detector, report

        if not fpr_ok:
            report.reason = f"FPR guard failed ({fpr_before:.4f} → {fpr_after:.4f})"
            return gmm_detector, report
        if not recall_ok:
            report.reason = (
                f"Recall guard failed "
                f"({report.recall_before:.4f} → {report.recall_after:.4f})"
            )
            return gmm_detector, report

        report.accepted = True
        report.reason = "validated on labeled holdout"
        return candidate, report

    # ── Proxy guard (no labeled data): adapted model must fit the recent
    # benign buffer at least as tightly as the stale model did.
    s_old = gmm_detector.score(X_buffer)
    s_new = candidate.score(X_buffer)
    if float(np.median(s_new)) <= float(np.median(s_old)):
        report.accepted = True
        report.reason = "proxy guard (buffer median score improved)"
    else:
        report.reason = "proxy guard failed"
    return (candidate if report.accepted else gmm_detector), report


def _fpr(model, X_benign: np.ndarray) -> float:
    return float(np.mean(model.predict(X_benign)))


def _recall(model, X_attack: np.ndarray, X_benign: np.ndarray) -> float:
    """Attack recall using each model's own threshold."""
    return float(np.mean(model.predict(X_attack)))
