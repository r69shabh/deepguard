import numpy as np
import pandas as pd
import pytest

from deepguard.robustness import mimicry_evasion, noise_fpr, robustness_report


def _task(seed=0):
    rng = np.random.RandomState(seed)
    ref = np.zeros(4)
    X_benign = rng.normal(0, 1, (400, 4))
    # scorer: distance from benign reference — mimicry must reduce it
    scorer = lambda Z: ((np.asarray(Z) - ref) ** 2).sum(axis=1)  # noqa: E731
    X_attack = rng.normal(0, 1, (100, 4)) + 5.0
    return X_benign, X_attack, scorer


def test_mimicry_reduces_scores_monotonically():
    _, X_attack, scorer = _task()
    df = mimicry_evasion(X_attack, scorer, np.zeros(4), threshold=10.0,
                         alphas=(0.0, 0.5, 1.0))
    assert (df.mean_score.diff().dropna() < 0).all()
    assert df.detection_rate.iloc[-1] < df.detection_rate.iloc[0]
    assert df.detection_rate.iloc[-1] == pytest.approx(0.0)


def test_mimicry_alpha_zero_is_identity():
    _, X_attack, scorer = _task()
    df = mimicry_evasion(X_attack, scorer, np.zeros(4), threshold=1.0,
                         alphas=(0.0,))
    expected = float(scorer(X_attack).mean())
    assert df.mean_score.iloc[0] == pytest.approx(expected)


def test_noise_fpr_increases_with_sigma():
    X_benign, _, scorer = _task()
    df = noise_fpr(X_benign, scorer, threshold=2.0,
                   sigmas=(0.0, 0.5, 2.0), n_repeats=3)
    assert df.fpr_mean.is_monotonic_increasing
    assert df.fpr_clean.nunique() == 1


def test_robustness_report_writes_csvs(tmp_path):
    X_benign, X_attack, scorer = _task()
    rep = robustness_report(X_benign, X_attack, scorer, threshold=10.0,
                            out_prefix=tmp_path / "rob")
    assert isinstance(rep["mimicry"], pd.DataFrame)
    assert (tmp_path / "rob_mimicry.csv").exists()
    assert (tmp_path / "rob_noise.csv").exists()
