"""
pipelines/monitor.py
====================
Streaming simulation: drift monitoring + adaptive retraining of the GMM.

Streams the labeled pool through the live detector; a ScoreStreamMonitor
watches anomaly scores / alert rate; on drift alarms the model is refit on a
ReplayBuffer of high-confidence benign flows and accepted or rolled back via
validation guards. Optionally injects synthetic covariate drift mid-stream to
exercise the full loop without real drifting data.

Outputs results/drift_events.csv and logs the timeline.

Usage
-----
    python main.py monitor
    python main.py monitor --inject-drift-after 20000 --drift-shift 3.0
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import deque

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.adaptation import ReplayBuffer, adapt_gmm
from deepguard.models import GMMDetector
from deepguard.monitors import ScoreStreamMonitor
from deepguard.utils import (
    ensure_dirs,
    labeled_holdout_split,
    load_config,
    set_seed,
    setup_logging,
)

logger = setup_logging()


def run(config_path: str = "configs/default.yaml",
        inject_drift_after: int | None = None,
        drift_shift: float = 3.0,
        max_steps: int | None = None) -> None:
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    root = pathlib.Path(__file__).parent.parent
    prep_dir = root / cfg["paths"]["prep_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    results_dir = root / cfg["paths"]["results_dir"]
    ensure_dirs(results_dir)

    mon_cfg = cfg.get("monitor", {})
    ad_cfg = cfg.get("adapt", {})

    logger.info("Loading preprocessed arrays …")
    X_train = np.load(prep_dir / "X_train.npy")
    X_test = np.load(prep_dir / "X_test.npy")
    y_test = np.load(prep_dir / "y_test.npy")

    # Reserve a labeled validation slice to gate adaptation.
    # With injected drift, validation must reflect the POST-drift world, so
    # we hold out the tail of the pool (shifted consistently) instead of
    # splitting the static distribution.
    if inject_drift_after is not None:
        n_val = max(150, int(len(X_test) * float(ad_cfg.get("validation_fraction", 0.10))))
        X_stream_all = X_test[:-n_val]
        X_val_raw, y_val_raw = X_test[-n_val:], y_test[-n_val:]
        stds = X_stream_all.std(axis=0)
        _shift = lambda A: A + drift_shift * stds[None, :]  # noqa: E731
        X_val = np.concatenate([_shift(X_val_raw[y_val_raw == 0]),
                                X_val_raw[y_val_raw == 1]])
        y_val = np.concatenate([np.zeros(int((y_val_raw == 0).sum()), int),
                                np.ones(int((y_val_raw == 1).sum()), int)])
    else:
        val_idx, stream_idx_all = labeled_holdout_split(
            y_test, meta_frac=float(ad_cfg.get("validation_fraction", 0.10)), seed=seed
        )
        X_val, y_val = X_test[val_idx], y_test[val_idx]
        X_stream_all = X_test[stream_idx_all]
        stds = None

    gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")

    # Baseline scores for the KS detector: clean benign calibration slice
    n_cal = min(5000, len(X_train))
    baseline_scores = gmm.score(X_train[:n_cal])
    threshold = -float(np.load(models_dir / "model_a_threshold.npy"))

    monitor = ScoreStreamMonitor(
        threshold=threshold,
        ph_tau=float(mon_cfg.get("ph_tau", 5.0)),
        ph_lambda=float(mon_cfg.get("ph_lambda", 0.5)),
        ph_alert_tau=float(mon_cfg.get("ph_alert_tau", 3.0)),
        use_adwin=True,
        adwin_delta=float(mon_cfg.get("adwin_delta", 0.002)),
        adwin_max_window=int(mon_cfg.get("adwin_max_window", 5000)),
        ks_baseline=baseline_scores,
        ks_alpha=float(mon_cfg.get("ks_alpha", 0.001)),
        ks_window=min(int(mon_cfg.get("ks_window", 1000)), n_cal),
    )

    buffer = ReplayBuffer(capacity=int(ad_cfg.get("buffer_capacity", 20_000)),
                          seed=seed,
                          mode=str(ad_cfg.get("buffer_mode", "sliding")))
    fpr_tol = float(ad_cfg.get("fpr_tolerance", 0.30))
    rec_tol = float(ad_cfg.get("recall_tolerance", 0.05))
    min_buffer = int(ad_cfg.get("min_buffer_to_adapt", 1500))
    attempt_interval = int(ad_cfg.get("min_attempt_interval", 100))
    val_benign = X_val[y_val == 0]
    val_attacks = X_val[y_val == 1]

    # Feature stds for synthetic covariate drift injection
    stds = X_stream_all.std(axis=0) if inject_drift_after is not None else None

    n_steps = len(X_stream_all) if max_steps is None else min(max_steps, len(X_stream_all))
    rows = []
    adaptations = []
    last_attempt = -attempt_interval
    last_attempt_buffer = 0
    pending_adaptation = False      # alarm fired; retry once buffer fills
    logged_starvation = False
    recent_scores = deque(maxlen=500)
    admit_q = float(ad_cfg.get("admit_quantile_under_drift", 0.40))
    logger.info(f"Streaming {n_steps:,} flows "
                f"({'with injected drift after step %d, shift=%.1fσ' % (inject_drift_after, drift_shift) if inject_drift_after else 'natural distribution'})")

    for i in range(n_steps):
        x = X_stream_all[i]
        if inject_drift_after is not None and i >= inject_drift_after:
            x = x + drift_shift * stds

        score = float(gmm.score(x.reshape(1, -1))[0])
        flagged = bool(gmm.predict(x.reshape(1, -1))[0])

        event = monitor.feed(score, flagged)
        if event is not None and not pending_adaptation:
            # First alarm of this episode: reset the buffer so the adapter
            # learns purely from post-change evidence (pre-shift rows would
            # dilute the new-regime sample).
            buffer.clear()
            logged_starvation = False
        if event is not None:
            pending_adaptation = True
        recent_scores.append(score)

        # Buffer admission: always take flows the current model considers
        # benign. Once a drift alarm is pending, also admit the lower tail of
        # the stream — the new benign regime — even though the stale model
        # flags it; otherwise the buffer starves exactly when it matters.
        admitted = not flagged
        if not admitted and pending_adaptation and len(recent_scores) >= 50:
            admit_cutoff = np.percentile(np.asarray(recent_scores), admit_q * 100)
            admitted = score <= admit_cutoff
        if admitted:
            buffer.add(x)
        growth_needed = max(20, min_buffer // 4)
        can_attempt = (pending_adaptation
                       and len(buffer) >= min_buffer
                       and len(buffer) >= last_attempt_buffer + growth_needed
                       and i - last_attempt >= 10)
        if (pending_adaptation and not can_attempt
                and len(buffer) < min_buffer and not logged_starvation):
            logger.info(f"  [step {i + 1}] drift suspected — buffer warming up "
                        f"({len(buffer)}/{min_buffer}); adaptation deferred")
            logged_starvation = True

        if can_attempt:
            last_attempt = i
            last_attempt_buffer = len(buffer)
            logged_starvation = False
            pending_adaptation = False   # consumed; a NEW alarm re-arms it
            logger.info(f"  [step {i + 1}] attempting adaptation (buffer={len(buffer)})")
            new_model, rep = adapt_gmm(
                gmm, buffer.get(),
                val_benign=val_benign, val_attacks=val_attacks,
                fpr_tolerance=fpr_tol, recall_tolerance=rec_tol,
                min_buffer=min_buffer, seed=seed,
            )
            detectors_tag = "deferred" if event is None else "+".join(event.detectors)
            adaptations.append({
                "step": i + 1, "detectors": detectors_tag,
                **{k: v for k, v in rep.__dict__.items()},
            })
            logger.info(
                f"    adaptation: {'ACCEPTED' if rep.accepted else 'REJECTED'} "
                f"({rep.reason}, buffer={rep.n_buffer})"
            )
            rows.append({
                "step": i + 1, "detectors": detectors_tag,
                "score": score,
                "adapted": rep.accepted, "reason": rep.reason,
                "fpr_before": rep.fpr_before, "fpr_after": rep.fpr_after,
                "recall_before": rep.recall_before, "recall_after": rep.recall_after,
            })
            if rep.accepted:
                gmm = new_model
                threshold = candidate_threshold(gmm)
                recent_benign = gmm.score(buffer.get()[-min(5000, len(buffer)):])
                monitor.rearm(new_threshold=threshold,
                              recent_benign_scores=recent_benign)
            else:
                # Re-arm the pending flag so a later alarm (or growth) retries
                # once meaningfully more evidence has arrived.
                pending_adaptation = True

    out_csv = results_dir / "drift_events.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    n_adapted = sum(1 for a in adaptations if a["accepted"])
    logger.info(f"{len(adaptations)} adaptation attempts · {n_adapted} applied · report → {out_csv}")


def candidate_threshold(gmm: GMMDetector) -> float:
    """Flag boundary on the -log p scale for a fitted GMMDetector."""
    return -float(gmm.threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift monitoring + adaptation simulation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inject-drift-after", type=int, default=None,
                        help="Step after which a synthetic covariate shift is applied")
    parser.add_argument("--drift-shift", type=float, default=3.0,
                        help="Shift magnitude in feature-std units")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.inject_drift_after, args.drift_shift, args.max_steps)
