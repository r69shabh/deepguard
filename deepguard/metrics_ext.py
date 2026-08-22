"""
deepguard/metrics_ext.py
========================
Rigorous evaluation utilities beyond ROC-AUC/F1:

- fpr_at_tpr        : operating-point metric (e.g., FPR @ 95% TPR)
- bootstrap_ci      : confidence intervals for any score-based metric
- per_attack_table  : recall / AUC broken down by attack type
- prevalence_sweep  : metrics under realistic (low) attack prevalence

Rationale: CICIDS-2017's ~17% attack rate is unrealistically dense; production
networks see far fewer attacks. ROC-AUC is optimistic under extreme class
imbalance, so we report PR-AUC, FPR@TPR, and a prevalence sweep alongside.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# ──────────────────────────────────────────────────────────────────────────────
# FPR at target TPR
# ──────────────────────────────────────────────────────────────────────────────

def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    False-positive rate achieved when the threshold is set to reach
    ``target_tpr`` true-positive rate.

    Returns 1.0 if the target TPR is unreachable (degenerate scores).
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr_arr, target_tpr, side="left")
    if idx >= len(fpr_arr):
        return 1.0
    return float(fpr_arr[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Percentile-bootstrap confidence interval for a score-based metric.

    Parameters
    ----------
    y_true, scores : arrays of equal length.
    metric_fn : callable(y, scores) -> float. Defaults to ROC-AUC.
    n_boot : number of stratified resamples.
    alpha : significance level; returns (1-alpha) interval.
    seed : RNG seed.

    Returns
    -------
    dict with keys: estimate, ci_lower, ci_upper.
    """
    if metric_fn is None:
        metric_fn = lambda y, s: roc_auc_score(y, s)  # noqa: E731

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.RandomState(seed)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    point = float(metric_fn(y_true, scores))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        take = np.concatenate([
            pos_idx[rng.randint(0, len(pos_idx), len(pos_idx))],
            neg_idx[rng.randint(0, len(neg_idx), len(neg_idx))],
        ])
        try:
            stats[b] = metric_fn(y_true[take], scores[take])
        except ValueError:
            stats[b] = np.nan

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "estimate": point,
        "ci_lower": float(np.nanpercentile(stats, lo_q)),
        "ci_upper": float(np.nanpercentile(stats, hi_q)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-attack-type breakdown
# ──────────────────────────────────────────────────────────────────────────────

def per_attack_table(
    y_true: np.ndarray,
    attack_types: np.ndarray,
    scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Detection quality per attack type: count, mean score, AUC (one-vs-benign),
    and recall@threshold when ``y_pred`` is provided.
    """
    y_true = np.asarray(y_true).astype(int)
    attack_types = np.asarray(attack_types)
    rows: List[Dict] = []

    benign_scores = scores[y_true == 0]
    for atype in sorted(set(attack_types[y_true == 1].tolist())):
        mask = (attack_types == atype) & (y_true == 1)
        type_scores = scores[mask]
        one_v_rest_y = np.concatenate([np.ones(mask.sum(), dtype=int),
                                       np.zeros(len(benign_scores), dtype=int)])
        one_v_rest_s = np.concatenate([type_scores, benign_scores])
        try:
            auc = float(roc_auc_score(one_v_rest_y, one_v_rest_s))
        except ValueError:
            auc = float("nan")
        row: Dict[str, object] = {
            "attack_type": atype,
            "count": int(mask.sum()),
            "mean_score": float(type_scores.mean()),
            "auc_vs_benign": auc,
        }
        if y_pred is not None:
            row["detection_rate"] = float(y_pred[mask].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("count", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Low-prevalence protocol
# ──────────────────────────────────────────────────────────────────────────────

def subsample_to_prevalence(
    y_true: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample the benign class so attacks make up ``prevalence`` of the pool.
    All attacks are kept.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    rng = np.random.RandomState(seed)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_neg_keep = min(len(neg_idx), max(1, int(len(pos_idx) * (1 - prevalence) / max(prevalence, 1e-9))))
    keep_neg = neg_idx[rng.choice(len(neg_idx), size=n_neg_keep, replace=False)]
    keep = np.concatenate([pos_idx, keep_neg])
    return y_true[keep], scores[keep]


def prevalence_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    prevalences: Tuple[float, ...] = (0.20, 0.10, 0.05, 0.01),
    seed: int = 42,
) -> pd.DataFrame:
    """
    PR-AUC / FPR@95%TPR as the effective attack prevalence drops toward
    realistic enterprise rates. ROC-AUC included for reference only — it is
    prevalence-invariant and therefore not informative here.
    """
    rows = []
    for prev in prevalences:
        y_sub, s_sub = subsample_to_prevalence(y_true, scores, prev, seed=seed)
        try:
            pr = float(average_precision_score(y_sub, s_sub))
        except ValueError:
            pr = float("nan")
        rows.append({
            "prevalence": prev,
            "n": len(y_sub),
            "n_attacks": int(y_sub.sum()),
            "pr_auc": pr,
            "fpr_at_95tpr": fpr_at_tpr(y_sub, s_sub, 0.95),
            # Baseline PR-AUC of a random scorer equals the prevalence itself
            "random_baseline_pr_auc": prev,
        })
    return pd.DataFrame(rows)
