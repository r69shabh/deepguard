"""
deepguard/fusion.py
===================
Leak-free score fusion for the hybrid detector (Phase 3 upgrade).

Protocol
--------
1. Base detectors produce per-flow scores on the labeled pool.
2. The pool is split into disjoint meta-fit / eval-only parts
   (see deepguard.utils.labeled_holdout_split).
3. Per-base-score calibrators (isotonic regression) are fitted on the
   META-FIT part only, then applied everywhere.
4. Meta-feature matrix = [raw scores | calibrated scores].
5. Candidate meta-learners (rf / lr / weighted) are compared by stratified
   CV *within the meta-fit part*, ranked by PR-AUC; the winner is refit on
   the full meta-fit part.
6. Final metrics are computed once, on eval-only.

No eval-only row ever influences calibrators or the meta-learner.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_scores(
    fit_scores: np.ndarray,
    y_fit: np.ndarray,
    apply_scores: np.ndarray,
) -> Tuple[np.ndarray, IsotonicRegression]:
    """
    Fit isotonic calibration on (fit_scores, y_fit); transform apply_scores.

    Returns (calibrated_apply, calibrator).
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(fit_scores, y_fit)
    return iso.predict(apply_scores), iso


def build_meta_features(
    raw_fit: np.ndarray,
    y_fit: np.ndarray,
    raw_apply: np.ndarray,
) -> Tuple[np.ndarray, List[IsotonicRegression]]:
    """
    Calibrate each base-score column on the meta-fit part and build the
    meta-feature matrix [raw | calibrated] for both fit and apply parts.

    Parameters
    ----------
    raw_fit : (n_fit, k) raw base scores for the meta-fit part.
    y_fit : (n_fit,) labels of the meta-fit part.
    raw_apply : (n_apply, k) raw base scores for any other part (e.g., eval).

    Returns
    -------
    X_meta_apply : (n_apply, 2k)
    calibrators : list of k fitted IsotonicRegression objects (for persistence).
    """
    raw_fit = np.atleast_2d(np.asarray(raw_fit, dtype=np.float64))
    if raw_fit.shape[0] == 1:
        raw_fit = raw_fit.T
    raw_apply = np.atleast_2d(np.asarray(raw_apply, dtype=np.float64))
    if raw_apply.shape[0] == 1 and raw_apply.shape[1] != raw_fit.shape[1]:
        raw_apply = raw_apply.T

    cal_parts, calibrators = [], []
    for j in range(raw_fit.shape[1]):
        cal_fit_j, iso = calibrate_scores(raw_fit[:, j], y_fit, raw_fit[:, j])
        cal_app_j, _ = calibrate_scores(raw_fit[:, j], y_fit, raw_apply[:, j])
        cal_parts.append(cal_app_j)
        calibrators.append(iso)

    X_meta = np.column_stack([raw_apply] + cal_parts)
    return X_meta, calibrators


# ──────────────────────────────────────────────────────────────────────────────
# Weighted-average candidate (fitted: alpha chosen on training fold)
# ──────────────────────────────────────────────────────────────────────────────

class WeightedAverageClassifier:
    """alpha * s0 + (1 - alpha) * s1 with alpha selected on the fit set."""

    def __init__(self, alphas: Optional[np.ndarray] = None) -> None:
        self.alphas = alphas if alphas is not None else np.round(np.arange(0.0, 1.01, 0.05), 2)
        self.alpha = 0.5

    def _score(self, X: np.ndarray, alpha: float) -> np.ndarray:
        s = alpha * X[:, 0]
        for j in range(1, X.shape[1]):
            s = s + (1.0 - alpha) / (X.shape[1] - 1) * X[:, j]
        return s

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedAverageClassifier":
        best_ap, best_a = -1.0, 0.5
        for a in self.alphas:
            ap = average_precision_score(y, self._score(X, float(a)))
            if ap > best_ap:
                best_ap, best_a = ap, float(a)
        self.alpha = best_a
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._score(X, self.alpha)
        return np.column_stack([1 - p, p])


def _make_candidate(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    if name == "weighted":
        return WeightedAverageClassifier()
    raise KeyError(f"Unknown meta-learner candidate '{name}'")


# ──────────────────────────────────────────────────────────────────────────────
# Candidate selection via CV inside the meta-fit part
# ──────────────────────────────────────────────────────────────────────────────

def select_meta_learner(
    X_meta_fit: np.ndarray,
    y_fit: np.ndarray,
    candidates: Tuple[str, ...] = ("rf", "lr", "weighted"),
    cv_folds: int = 3,
    seed: int = 42,
) -> Tuple[str, object, List[Dict]]:
    """
    Compare candidates by stratified CV on the meta-fit part (PR-AUC),
    refit the winner on all of it.

    Returns (best_name, fitted_best, cv_rows).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: List[Dict] = []

    for cand in candidates:
        fold_scores = []
        for tr_idx, va_idx in skf.split(X_meta_fit, y_fit):
            est = _make_candidate(cand, seed)
            est.fit(X_meta_fit[tr_idx], y_fit[tr_idx])
            proba = est.predict_proba(X_meta_fit[va_idx])[:, 1]
            fold_scores.append(average_precision_score(y_fit[va_idx], proba))
        rows.append({
            "candidate": cand,
            "cv_pr_auc_mean": float(np.mean(fold_scores)),
            "cv_pr_auc_std": float(np.std(fold_scores)),
        })

    best_row = max(rows, key=lambda r: r["cv_pr_auc_mean"])
    best_name = best_row["candidate"]
    best = _make_candidate(best_name, seed).fit(X_meta_fit, y_fit)
    return best_name, best, rows


# ──────────────────────────────────────────────────────────────────────────────
# FusionModel — persisted inference wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FusionModel:
    """
    End-to-end hybrid scorer: GMM + sequence-detector flow scores → calibrated
    meta-features → meta-learner probability.

    The heavy base models (GMM pickle, sequence detector artifacts) are loaded
    separately and injected, mirroring HybridDetector's convention.
    """

    def __init__(
        self,
        gmm_detector,
        seq_detector,
        calibrators: Optional[List[IsotonicRegression]] = None,
        meta_learner=None,
        alpha: float = 0.7,
        threshold: float = 0.5,
        seq_stride: int = 1,
    ) -> None:
        self.gmm_detector = gmm_detector
        self.seq_detector = seq_detector
        self.calibrators = calibrators or []
        self.meta_learner = meta_learner
        self.alpha = alpha
        self.threshold = threshold
        self.seq_stride = seq_stride

    # -- base scores ----------------------------------------------------------
    def gmm_scores(self, X: np.ndarray) -> np.ndarray:
        # GMMDetector.score() already returns -log p(x)
        return self.gmm_detector.score(X)

    def seq_scores(self, X: np.ndarray) -> np.ndarray:
        return self.seq_detector.score_flows(X, stride=self.seq_stride)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.gmm_scores(X), self.seq_scores(X)])

    # -- fusion ---------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_scores(X)
        if self.meta_learner is not None and self.calibrators:
            cols = [raw[:, j] for j in range(raw.shape[1])]
            cal_cols = [
                np.asarray(iso.predict(raw[:, j]), dtype=np.float64)
                for j, iso in enumerate(self.calibrators)
            ]
            X_meta = np.column_stack(cols + cal_cols)
            return self.meta_learner.predict_proba(X_meta)[:, 1]
        # fallback: weighted average of min-max normalised raw scores
        def norm(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo + 1e-8)

        a = self.alpha
        return a * norm(raw[:, 0]) + (1 - a) * norm(raw[:, 1])

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.score(X) >= t).astype(int)

    # -- persistence ------------------------------------------------------------
    def save(self, path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, str(path) + "_meta.pkl")
        joblib.dump(
            {
                "calibrators": self.calibrators,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "seq_stride": self.seq_stride,
            },
            str(path) + "_fusion.pkl",
        )

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path],
        gmm_detector,
        seq_detector,
    ) -> "FusionModel":
        path = pathlib.Path(path)
        cfg = joblib.load(str(path) + "_fusion.pkl")
        meta_path = pathlib.Path(str(path) + "_meta.pkl")
        meta = joblib.load(meta_path) if meta_path.exists() else None
        return cls(
            gmm_detector=gmm_detector,
            seq_detector=seq_detector,
            calibrators=cfg["calibrators"],
            meta_learner=meta,
            alpha=cfg["alpha"],
            threshold=cfg["threshold"],
            seq_stride=cfg["seq_stride"],
        )
