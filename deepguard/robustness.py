"""
deepguard/robustness.py
=======================
Adversarial robustness experiments for the anomaly detectors.

Two threat models:

1. Mimicry evasion — an attacker who knows the benign profile perturbs attack
   flows toward the benign reference point (feature-wise interpolation). We
   measure how much the anomaly score / detection rate degrades as the attacker
   is allowed to move closer to "normal" (α = 0 → no change, α = 1 → full
   replacement by the benign reference).

2. Noise-induced false positives — benign flows perturbed with Gaussian noise
   of increasing magnitude; we measure how quickly FPR grows. A robust
   operational detector keeps FPR low under sensor/feature noise.

All experiments run in PROCESSED feature space against any scorer callable,
so they apply uniformly to GMM, fusion, and deep models.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1. Mimicry evasion
# ──────────────────────────────────────────────────────────────────────────────

def mimicry_evasion(
    X_attack: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    benign_reference: np.ndarray,
    threshold: float,
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """
    Interpolate attack flows toward the benign reference and re-score.

        x'(α) = x + α · (ref − x)      for each feature

    Parameters
    ----------
    X_attack : (n, k) processed attack flows.
    scorer : callable (n', k) → (n',) anomaly scores.
    benign_reference : (k,) processed-space benign reference.
    threshold : decision boundary on scores.
    alphas : attacker strengths to sweep.

    Returns
    -------
    DataFrame: alpha, mean_score, detection_rate (fraction above threshold).
    """
    Xa = np.asarray(X_attack, dtype=np.float64)
    ref = np.asarray(benign_reference, dtype=np.float64)

    rows: List[Dict] = []
    for a in alphas:
        X_evade = Xa + a * (ref[None, :] - Xa)
        s = np.asarray(scorer(X_evade))
        rows.append({
            "alpha": float(a),
            "mean_score": float(s.mean()),
            "detection_rate": float((s >= threshold).mean()),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Noise robustness of benign traffic
# ──────────────────────────────────────────────────────────────────────────────

def noise_fpr(
    X_benign: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    sigmas: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add Gaussian noise (σ scaled by per-feature std of benign data) to benign
    flows and measure the resulting false-positive rate.

    Returns DataFrame: sigma, fpr_mean, fpr_std across repeats.
    """
    Xb = np.asarray(X_benign, dtype=np.float64)
    stds = Xb.std(axis=0)
    stds[stds == 0] = 1.0
    rng = np.random.RandomState(seed)

    rows = []
    base_fpr = float((np.asarray(scorer(Xb)) >= threshold).mean())
    for sigma in sigmas:
        fprs = []
        for _ in range(n_repeats):
            Xn = Xb + rng.normal(0.0, sigma * stds[None, :], size=Xb.shape)
            s = np.asarray(scorer(Xn))
            fprs.append(float((s >= threshold).mean()))
        rows.append({
            "sigma": float(sigma),
            "fpr_mean": float(np.mean(fprs)),
            "fpr_std": float(np.std(fprs)),
            "fpr_clean": base_fpr,
        })
    return pd.DataFrame(rows)


def robustness_report(
    X_benign: np.ndarray,
    X_attack: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    out_prefix,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Run both experiments and write CSVs next to ``out_prefix``."""
    from pathlib import Path

    ref = np.median(np.asarray(X_benign, dtype=np.float64), axis=0)
    ev_df = mimicry_evasion(X_attack, scorer, ref, threshold)
    nz_df = noise_fpr(X_benign, scorer, threshold, seed=seed)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(str(out_prefix) + "_mimicry.csv", index=False)
    nz_df.to_csv(str(out_prefix) + "_noise.csv", index=False)
    return {"mimicry": ev_df, "noise": nz_df}
