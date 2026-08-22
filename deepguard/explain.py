"""
deepguard/explain.py
====================
Model-agnostic per-flow explanation via counterfactual attribution.

For a flagged flow x and an anomaly scorer S, the contribution of feature j is

    φ_j = S(x) − S(x with feature j replaced by its benign reference value)

A large positive φ_j means "this flow looks anomalous largely because feature j
deviates from benign behaviour". This works for ANY scorer (GMM, fusion
meta-learner, Deep SVDD) with no gradients or model internals required, and it
uses only information available at inference time (benign reference values).

Reference values are the FeatureEngineer's training medians mapped through the
same preprocessing — i.e., the processed-space value a typical benign flow has.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def counterfactual_attribution(
    x_processed: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    benign_reference: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """
    Per-feature attribution for ONE preprocessed flow.

    Parameters
    ----------
    x_processed : (n_features,) processed feature vector of the flow.
    scorer : callable mapping (1, n_features) → (1,) anomaly score
             (higher = more anomalous). Must be deterministic.
    benign_reference : (n_features,) processed-space benign reference point
                       (e.g., per-feature train medians in processed space).
    feature_names : names aligned with the processed columns.

    Returns
    -------
    list of (feature_name, contribution) sorted by |contribution| descending.
    """
    x = np.asarray(x_processed, dtype=np.float64)
    ref = np.asarray(benign_reference, dtype=np.float64)
    base = float(scorer(x.reshape(1, -1))[0])

    out: List[Tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        x_cf = x.copy()
        x_cf[j] = ref[j]
        s_cf = float(scorer(x_cf.reshape(1, -1))[0])
        out.append((name, base - s_cf))

    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out


def explain_flow(
    x_processed: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    benign_reference: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, List]:
    """
    Convenience wrapper returning a dict suitable for CSV columns:
      {"top_features": [...], "contributions": [...]}
    """
    attribs = counterfactual_attribution(x_processed, scorer, benign_reference,
                                         feature_names)[:top_k]
    return {
        "top_features": [a[0] for a in attribs],
        "contributions": [float(a[1]) for a in attribs],
    }


def benign_reference_point(
    X_benign_processed: np.ndarray,
) -> np.ndarray:
    """Per-feature median of processed benign flows — the reference point."""
    X = np.asarray(X_benign_processed)
    return np.median(X, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Meta-learner permutation importance
# ──────────────────────────────────────────────────────────────────────────────

def permutation_importance_scores(
    estimator,
    X_meta: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """
    Permutation importance (mean PR-AUC drop) for the meta-learner's features.

    Parameters
    ----------
    estimator : fitted object exposing predict_proba(X)[:, 1].
    X_meta : (n, k) meta-feature matrix.
    y : labels of that same (eval-only) part.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.RandomState(seed)
    base_p = estimator.predict_proba(X_meta)[:, 1]
    base_ap = average_precision_score(y, base_p)

    drops: List[Tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        vals = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X_meta))
            Xp = X_meta.copy()
            Xp[:, j] = X_meta[perm, j]
            p = estimator.predict_proba(Xp)[:, 1]
            vals.append(base_ap - average_precision_score(y, p))
        drops.append((name, float(np.mean(vals))))

    drops.sort(key=lambda t: t[1], reverse=True)
    return drops
