"""
deepguard/monitors.py
=====================
Concept-drift detection over anomaly-score streams.

Detectors
---------
PageHinckleyDriftDetector : incremental one-sided test for sustained mean
    increases (e.g., rising anomaly scores / alert rate). Faithful to the
    formulation used in river: running mean, cumulative deviation sum minus
    margin λ, alarm when (sum − min) exceeds τ.

ADWINDetector : adaptive-windowing (simplified exact ADWIN0). Keeps a capped
    window of recent values; at each check it searches for a split whose
    sub-window means differ by more than the Hoeffding bound ε and drops the
    older sub-window. A cut means "the recent stream is statistically
    different from what came before" → drift signal.

KSWindowDriftDetector : two-sample Kolmogorov–Smirnov test between the most
    recent window of scores and a frozen baseline window; alarms when p < α.

ScoreStreamMonitor : orchestrates detectors over a live score stream and
    aggregates alarms. Used by pipelines/monitor.py and reusable in production.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Page-Hinckley
# ──────────────────────────────────────────────────────────────────────────────

class PageHinckleyDriftDetector:
    """
    Incremental Page–Hinckley test for monotonic mean drift.

    Detects *increases* by default (appropriate for anomaly-score streams,
    where benign-traffic drift pushes scores up).

    Parameters
    ----------
    tau : float — alarm threshold on the cumulative-deviation statistic.
        Should scale well above the natural excursion size σ²/(2λ).
    lamda : float — drift margin subtracted per step. Pins the statistic on
        stationary streams; choose λ ≈ σ/10 for roughly unit-variance scores.
    min_instances : int — warm-up samples before any alarm can fire.
    """

    def __init__(self, tau: float = 40.0, lamda: float = 0.1,
                 min_instances: int = 150) -> None:
        self.tau = float(tau)
        self.lamda = float(lamda)
        self.min_instances = int(min_instances)
        self._n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._min_sum = 0.0

    def update(self, x: float) -> bool:
        """Feed one value; returns True exactly on the step an alarm fires."""
        x = float(x)
        self._n += 1
        if self._n == 1:
            self._mean = x
        else:
            self._mean += (x - self._mean) / self._n
        if self._n <= self.min_instances:
            return False

        # one-sided: detect upward shifts
        self._sum += (x - self._mean) - self.lamda
        self._min_sum = min(self._min_sum, self._sum)
        if self._sum - self._min_sum > self.tau:
            # Full re-anchor: after a level shift the old mean is obsolete;
            # keeping it would re-alarm indefinitely. Warm-up re-applies,
            # giving a natural refractory period while the new regime settles.
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Re-arm after an alarm or adaptation (fully re-anchors the mean)."""
        self._n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._min_sum = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ADWIN (simplified exact ADWIN0)
# ──────────────────────────────────────────────────────────────────────────────

class ADWINDetector:
    """
    Adaptive Windowing (ADWIN0) with a hard cap on window length.

    At every ``cadence``-th sample the full window is scanned for a split
    point where the two sub-window means differ by at least

        ε(n0, n1) = sqrt( (1 / (2·m)) · ln(4/δ) ),   m = 1/(1/n0 + 1/n1)

    On such a split the OLDER sub-window is dropped (the stream has changed).
    The detector reports drift on steps where at least one cut occurred.

    Parameters
    ----------
    delta : float — confidence parameter (smaller = fewer false alarms).
        A Bonferroni budget over all split checks performed since the last
        cut keeps the family-wise error bounded by ``delta``.
    max_window : int — hard cap; beyond it the oldest values are discarded
        (bounded-memory approximation of ADWIN).
    cadence : int — run the (O(n)) split scan only every k-th sample.
    min_window : int — no splits are considered below this width.
    """

    def __init__(self, delta: float = 0.0005, max_window: int = 5000,
                 cadence: int = 50, min_window: int = 150) -> None:
        self.delta = float(delta)
        self.max_window = int(max_window)
        self.cadence = int(cadence)
        self.min_window = int(min_window)
        self._window: List[float] = []
        self._t = 0
        self._checks = 0
        self._pending_cut = False
        self.width: int = 0

    def _epsilon(self, n0: int, n1: int) -> float:
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        # Bonferroni: budget δ across every split comparison made so far
        budget = 4.0 * max(1, self._checks) / self.delta
        return float(np.sqrt((1.0 / (2.0 * m)) * np.log(budget)))

    def update(self, x: float) -> bool:
        """Feed one value; returns True on steps where a cut occurred."""
        self._window.append(float(x))
        self._t += 1

        if len(self._window) > self.max_window:
            self._window = self._window[-self.max_window:]
        self.width = len(self._window)

        if self._t % self.cadence != 0 or len(self._window) < self.min_window:
            return False

        arr = np.asarray(self._window, dtype=np.float64)
        prefix_sum = np.cumsum(arr)
        total = prefix_sum[-1]

        candidates = range(self.min_window, len(arr) - self.min_window + 1)
        n_candidates = max(1, len(list(candidates)) if not isinstance(candidates, range)
                           else (len(arr) - 2 * self.min_window + 1))
        self._checks += max(0, n_candidates)

        for k in range(self.min_window, len(arr) - self.min_window + 1):
            s0 = prefix_sum[k - 1]
            s1 = total - s0
            n0, n1 = k, len(arr) - k
            diff = abs(s0 / n0 - s1 / n1)
            if diff >= self._epsilon(n0, n1):
                # Two-scan confirmation: a genuine regime change persists to
                # the next scan; a noise cut usually does not.
                if not self._pending_cut:
                    self._pending_cut = True
                    self._checks = 0
                    return False
                self._window = arr[k:].tolist()   # keep only the new regime
                self.width = len(self._window)
                self._checks = 0
                self._pending_cut = False
                return True
        self._pending_cut = False
        return False

    def reset(self) -> None:
        self._window.clear()
        self._t = 0
        self._checks = 0
        self.width = 0


# ──────────────────────────────────────────────────────────────────────────────
# KS window-vs-baseline monitor
# ──────────────────────────────────────────────────────────────────────────────

class KSWindowDriftDetector:
    """
    Two-sample KS test between the latest ``window`` scores and a frozen
    baseline distribution of scores (typically from clean calibration data).

    Parameters
    ----------
    baseline : array-like — reference scores.
    alpha : float — significance level; p-value below α ⇒ drift.
    window : int — size of the rolling recent window.
    cadence : int — run the test every k-th sample.
    """

    def __init__(self, baseline: np.ndarray, alpha: float = 0.001,
                 window: int = 1000, cadence: int = 25) -> None:
        from scipy import stats as _stats  # noqa: F401  (import guard)

        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.alpha = float(alpha)
        self.window = int(window)
        self.cadence = int(cadence)
        self._recent: deque = deque(maxlen=window)
        self._t = 0
        self.last_pvalue: Optional[float] = None

    def set_baseline(self, scores: np.ndarray) -> None:
        """Refresh the reference distribution (after adaptation)."""
        self.baseline = np.asarray(scores, dtype=np.float64)

    def update(self, x: float) -> bool:
        from scipy.stats import ks_2samp

        self._recent.append(float(x))
        self._t += 1
        if self._t % self.cadence != 0 or len(self._recent) < self.window:
            return False
        result = ks_2samp(self.baseline, np.asarray(self._recent))
        self.last_pvalue = float(result.pvalue)
        return self.last_pvalue < self.alpha


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftEvent:
    step: int
    detectors: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)


class ScoreStreamMonitor:
    """
    Watches an anomaly-score stream with several detectors and aggregates
    their alarms into DriftEvents.

    Parameters
    ----------
    threshold : float — decision boundary used to derive the alert indicator.
    ph_tau, ph_lambda : Page-Hinckley settings (score level).
    ph_alert_tau : PH threshold for the binary alert-rate series.
    use_adwin, adwin_kwargs : enable/configure the adaptive-window detector.
    ks_baseline : optional baseline scores enabling the KS detector.
    ks_alpha, ks_window : KS detector settings.
    """

    def __init__(
        self,
        threshold: float,
        ph_tau: float = 40.0,
        ph_lambda: float = 0.1,
        ph_alert_tau: float = 15.0,
        use_adwin: bool = True,
        adwin_delta: float = 0.002,
        adwin_max_window: int = 5000,
        ks_baseline: Optional[np.ndarray] = None,
        ks_alpha: float = 0.001,
        ks_window: int = 1000,
    ) -> None:
        self.threshold = float(threshold)
        self.ph_scores = PageHinckleyDriftDetector(tau=ph_tau, lamda=ph_lambda)
        self.ph_alerts = PageHinckleyDriftDetector(tau=ph_alert_tau, lamda=0.005)
        self.adwin = (
            ADWINDetector(delta=adwin_delta, max_window=adwin_max_window)
            if use_adwin else None
        )
        self.ks = (
            KSWindowDriftDetector(ks_baseline, alpha=ks_alpha, window=ks_window)
            if ks_baseline is not None else None
        )
        self.events: List[DriftEvent] = []
        self.step = 0

    def feed(self, score: float, flagged: Optional[bool] = None) -> Optional[DriftEvent]:
        """
        Feed one anomaly score (and optionally its known flag); returns a
        DriftEvent when at least one detector fired on this step.
        """
        self.step += 1
        is_alert = bool(flagged) if flagged is not None else (score >= self.threshold)

        fired: List[str] = []
        if self.ph_scores.update(score):
            fired.append("ph_score")
        if self.ph_alerts.update(1.0 if is_alert else 0.0):
            fired.append("ph_alert_rate")
        if self.adwin is not None and self.adwin.update(score):
            fired.append("adwin")
        if self.ks is not None and self.ks.update(score):
            fired.append("ks_window")

        event = None
        if fired:
            event = DriftEvent(
                step=self.step,
                detectors=fired,
                details={"score": float(score),
                         "ks_pvalue": self.ks.last_pvalue if self.ks else float("nan")},
            )
            self.events.append(event)
        return event

    def rearm(self, new_threshold: Optional[float] = None,
              recent_benign_scores: Optional[np.ndarray] = None) -> None:
        """Re-arm all detectors after adaptation (optionally recalibrated)."""
        if new_threshold is not None:
            self.threshold = float(new_threshold)
        self.ph_scores.reset()
        self.ph_alerts.reset()
        if self.adwin is not None:
            self.adwin.reset()
        if self.ks is not None and recent_benign_scores is not None:
            self.ks.set_baseline(recent_benign_scores)
