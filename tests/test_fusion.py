import numpy as np
import pytest

from deepguard.fusion import (
    calibrate_scores,
    build_meta_features,
    select_meta_learner,
    WeightedAverageClassifier,
)


def _synthetic_scores(n=400, seed=0):
    """Two base scores with different separations."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    s0 = rng.normal(2.0 * y, 1.0)          # strong
    s1 = rng.normal(0.5 * y, 1.5)          # weak
    return np.column_stack([s0, s1]), y


def test_calibrate_scores_monotonic_and_bounded():
    X, y = _synthetic_scores()
    cal, iso = calibrate_scores(X[:, 0], y, X[:, 0])
    assert ((cal >= 0) & (cal <= 1)).all()
    # calibrated scores must preserve ordering (isotonic)
    order_raw = np.argsort(X[:, 0])
    assert np.all(np.diff(cal[order_raw]) >= -1e-9)


def test_build_meta_features_shape_and_calibration_fit_isolation():
    X, y = _synthetic_scores(200)
    meta_idx = np.arange(100)
    eval_idx = np.arange(100, 200)
    X_meta_eval, cals = build_meta_features(X[meta_idx], y[meta_idx], X[eval_idx])
    assert X_meta_eval.shape == (100, 4)  # [raw | calibrated] per base
    assert len(cals) == 2
    assert ((X_meta_eval[:, 2:] >= 0) & (X_meta_eval[:, 2:] <= 1)).all()


def test_select_meta_learner_picks_something_sensible():
    X, y = _synthetic_scores(300)
    best_name, est, rows = select_meta_learner(X, y, candidates=("rf", "lr", "weighted"),
                                               cv_folds=3, seed=1)
    assert best_name in {"rf", "lr", "weighted"}
    assert len(rows) == 3
    proba = est.predict_proba(X)[:, 1]
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, proba) > 0.8


def test_weighted_average_classifier_finds_good_alpha():
    X, y = _synthetic_scores(300)
    wac = WeightedAverageClassifier().fit(X, y)
    assert wac.alpha > 0.5  # s0 is the informative one
    p = wac.predict_proba(X)[:, 1]
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, p) > 0.85
