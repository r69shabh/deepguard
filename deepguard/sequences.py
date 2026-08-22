"""
deepguard/sequences.py
======================
Sliding-window construction for sequence anomaly detectors.

Design decisions (fixing the Phase-2 failures of the original pipeline):

1. Training windows are built from BENIGN rows only. The original notebook
   sliced windows across the whole (shuffled) test pool, so "benign" training
   windows could contain attack flows and temporal structure was destroyed
   by shuffling.

2. Evaluation produces FLOW-LEVEL scores, not window-level scores. A window
   MSE averaged over W×F elements dilutes a single anomalous flow across the
   whole window (the reason LSTM-AE scored AUC ≈ 0.52 on DoS Hulk). Instead,
   detectors reconstruct every timestep and we map each timestep's error back
   to the flow that produced it, averaging over the windows that cover it.

3. Window labels are reported for reference but flow-level labels remain the
   primary evaluation target, aligned 1:1 with GMM / DeepSVDD scores.
"""

from __future__ import annotations

import numpy as np


def build_train_windows(
    X_benign: np.ndarray,
    window_size: int,
    stride: int = 1,
    max_windows: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Sliding windows over benign-only flows, in row order.

    Parameters
    ----------
    X_benign : (n_flows, n_features) — benign-only feature matrix.
    window_size : int — flows per window.
    stride : int — step between window starts.
    max_windows : int, optional — subsample to this many windows (seeded).
    seed : int — RNG seed for subsampling.

    Returns
    -------
    np.ndarray (n_windows, window_size, n_features), float32.
    """
    X = np.asarray(X_benign, dtype=np.float32)
    n = len(X)
    if n < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} benign flows, got {n}."
        )
    starts = range(0, n - window_size + 1, max(1, stride))
    windows = np.stack([X[t : t + window_size] for t in starts])

    if max_windows is not None and len(windows) > max_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(windows), max_windows, replace=False)
        windows = windows[np.sort(idx)]
    return windows.astype(np.float32)


def build_eval_windows(
    X: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Sliding windows over any flow pool with an explicit flow-coverage map.

    Parameters
    ----------
    X : (n_flows, n_features)
    window_size : int
    stride : int

    Returns
    -------
    windows : (n_windows, window_size, n_features) float32
    coverage : list of lists — coverage[i] holds the flow indices inside
               window i. Together they define the window→flow scatter used
               by score_flows().
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n < window_size:
        # Degenerate input: return a single zero-padded window whose coverage
        # is all flows, so scoring still yields one score per flow.
        pad = np.zeros((window_size - n, X.shape[1]), dtype=np.float32)
        return np.stack([np.concatenate([X, pad])]), [list(range(n))]

    starts = range(0, n - window_size + 1, max(1, stride))
    windows, coverage = [], []
    for t in starts:
        windows.append(X[t : t + window_size])
        coverage.append(list(range(t, t + window_size)))
    return np.stack(windows).astype(np.float32), coverage


def scatter_window_errors_to_flows(
    per_timestep_err: np.ndarray,
    coverage: list[list[int]],
    n_flows: int,
) -> np.ndarray:
    """
    Map per-window per-timestep errors to per-flow scores.

    Parameters
    ----------
    per_timestep_err : (n_windows, window_size) — reconstruction error of each
        timestep within each window (already reduced over features).
    coverage : window → flow-index lists, as produced by build_eval_windows().
    n_flows : int — total number of flows (for uncovered tails).

    Returns
    -------
    (n_flows,) float array — mean error over all windows covering the flow;
    flows covered by no window (can happen with stride > 1) get the global mean.
    """
    err = np.asarray(per_timestep_err, dtype=np.float64)
    flow_sum = np.zeros(n_flows, dtype=np.float64)
    flow_cnt = np.zeros(n_flows, dtype=np.int64)

    for w_idx, flow_ids in enumerate(coverage):
        if w_idx >= len(err):
            break
        errs = err[w_idx]
        for k, f in enumerate(flow_ids):
            if k >= len(errs):
                break
            flow_sum[f] += errs[k]
            flow_cnt[f] += 1

    scores = np.where(flow_cnt > 0, flow_sum / np.maximum(flow_cnt, 1), np.nan)
    if np.isnan(scores).any():
        global_mean = float(np.nanmean(scores))
        scores = np.where(np.isnan(scores), global_mean, scores)
    return scores.astype(np.float64)


def window_labels_from_flows(
    y_flow: np.ndarray,
    coverage: list[list[int]],
    min_attack_frac: float = 0.0,
) -> np.ndarray:
    """
    Binary label per window: 1 if the fraction of attack flows it contains
    exceeds ``min_attack_frac`` (default: any single attack flow ⇒ positive).
    """
    y = np.asarray(y_flow).astype(int)
    labels = np.zeros(len(coverage), dtype=int)
    for i, flow_ids in enumerate(coverage):
        if not flow_ids:
            continue
        frac = y[flow_ids].mean()
        labels[i] = int(frac > min_attack_frac)
    return labels
