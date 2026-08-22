import numpy as np
import pytest

from deepguard.monitors import (
    ADWINDetector,
    KSWindowDriftDetector,
    PageHinckleyDriftDetector,
    ScoreStreamMonitor,
)


class TestPageHinckley:
    def test_no_alarm_on_stationary_stream(self):
        rng = np.random.RandomState(0)
        ph = PageHinckleyDriftDetector(tau=40.0, lamda=0.1)
        alarms = [ph.update(x) for x in rng.normal(0, 1, 3000)]
        assert not any(alarms)

    def test_alarms_after_mean_shift(self):
        rng = np.random.RandomState(1)
        ph = PageHinckleyDriftDetector(tau=40.0, lamda=0.1)
        stream = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
        alarm_steps = [i for i, x in enumerate(stream) if ph.update(x)]
        assert len(alarm_steps) >= 1
        assert 500 <= min(alarm_steps) <= 700  # detects soon after the shift


class TestADWIN:
    def test_cut_after_distribution_change(self):
        rng = np.random.RandomState(2)
        ad = ADWINDetector(delta=0.0005, max_window=2000, cadence=20, min_window=150)
        stream = np.concatenate([rng.normal(0, 1, 400), rng.normal(4, 1, 500)])
        cuts = [i for i, x in enumerate(stream) if ad.update(x)]
        assert cuts and cuts[0] > 400          # only reacts to real change
        assert cuts[0] <= 700                  # ...and promptly
        assert ad.width < 800                  # window shrank (old regime dropped)

    def test_stationary_stream_rarely_cuts(self):
        """Bounded-memory ADWIN keeps a small nonzero FP rate; a false alarm
        is cheap because adaptation is validation-gated downstream."""
        rng = np.random.RandomState(3)
        ad = ADWINDetector(delta=0.0005, max_window=2000, cadence=20, min_window=150)
        cuts = sum(ad.update(x) for x in rng.normal(0, 1, 1500))
        assert cuts <= 1


class TestKSWindow:
    def test_detects_swapped_distribution(self):
        rng = np.random.RandomState(4)
        baseline = rng.normal(0, 1, 2000)
        ks = KSWindowDriftDetector(baseline, alpha=0.001, window=300, cadence=10)
        fired = any(ks.update(x) for x in rng.normal(6, 1, 900))
        assert fired
        assert ks.last_pvalue is not None and ks.last_pvalue < 0.01

    def test_quiet_for_matching_distribution(self):
        rng = np.random.RandomState(5)
        baseline = rng.normal(0, 1, 2000)
        ks = KSWindowDriftDetector(baseline, alpha=0.001, window=300, cadence=10)
        fired = any(ks.update(x) for x in rng.normal(0, 1, 900))
        assert not fired


class TestScoreStreamMonitor:
    def _run(self, monitor, scores):
        events = []
        for s in scores:
            ev = monitor.feed(s)
            if ev is not None:
                events.append(ev)
        return events

    def test_end_to_end_drift_and_rearm(self):
        rng = np.random.RandomState(6)
        baseline = rng.normal(0.2, 0.05, 2000)
        m = ScoreStreamMonitor(threshold=0.9, ks_baseline=baseline,
                               ks_alpha=0.001, ks_window=300,
                               adwin_delta=0.0005, adwin_max_window=2000)
        stationary = rng.normal(0.2, 0.05, 1200)
        drifted = rng.normal(1.4, 0.05, 1200)   # benign traffic drifts up
        events = self._run(m, np.concatenate([stationary, drifted]))
        assert events
        first = min(e.step for e in events)
        assert 1100 <= first <= 1450            # near the shift point

        m.rearm(new_threshold=0.9, recent_benign_scores=rng.normal(1.4, 0.05, 1000))
        post = self._run(m, rng.normal(1.4, 0.05, 800))
        assert all(e.step > 2400 for e in post) or not post  # quiet after rearm
