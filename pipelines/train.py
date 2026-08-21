"""
pipelines/train.py
==================
End-to-end training pipeline for all three model phases.

Phase 1: Isolation Forest, OC-SVM, GMM baselines (flow level).
Phase 2: configurable sequence detector (LSTM-AE / Transformer-AE / USAD),
         trained on benign-only sliding windows built internally; evaluated
         at FLOW level via per-timestep error mapping.
Phase 3: leak-free fusion — isotonic calibration + CV-selected meta-learner,
         fitted exclusively on the meta-fit split; metrics on eval-only.

Usage
-----
    python main.py preprocess
    python main.py train
    python main.py train --phases 1 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.evaluate import compute_metrics
from deepguard.fusion import FusionModel, WeightedAverageClassifier, select_meta_learner
from deepguard.models import GMMDetector, IsolationForestDetector, OCSVMDetector
from deepguard.models_deep import create_detector, load_detector
from deepguard.sequences import build_eval_windows, build_train_windows, window_labels_from_flows
from deepguard.utils import (
    ensure_dirs,
    labeled_holdout_split,
    load_config,
    set_seed,
    setup_logging,
)

logger = setup_logging()

SEQ_MODEL_CONFIG_KEYS = {
    "lstm_ae": ("lstm_ae", ["window_size", "latent_dim", "dropout_rate", "lr", "threshold_pct"]),
    "transformer_ae": ("transformer_ae", ["window_size", "d_model", "num_heads",
                                          "latent_dim", "dropout_rate", "lr", "threshold_pct"]),
    "usad": ("usad", ["window_size", "hidden_dim", "latent_dim", "alpha", "lr", "threshold_pct"]),
}


def _load_arrays(prep_dir: pathlib.Path):
    """Load preprocessed numpy arrays from outputs/preprocessing/."""
    logger.info(f"Loading preprocessed arrays from {prep_dir}")
    X_train = np.load(prep_dir / "X_train.npy")
    X_test  = np.load(prep_dir / "X_test.npy")
    y_test  = np.load(prep_dir / "y_test.npy")
    logger.info(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}  |  y_test : {y_test.shape}")
    return X_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Baseline + GMM
# ──────────────────────────────────────────────────────────────────────────────

def train_phase1(cfg: dict, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 models_dir: pathlib.Path, results_dir: pathlib.Path) -> GMMDetector:
    """Train Isolation Forest, OCSVM, and GMM. Save artifacts and metrics CSV."""

    logger.info("=" * 60)
    logger.info("PHASE 1 — Classical ML Baseline")
    logger.info("=" * 60)

    rows = []

    logger.info("Training Isolation Forest …")
    t0 = time.time()
    if_cfg = cfg.get("isolation_forest", {})
    ifd = IsolationForestDetector(
        n_estimators=if_cfg.get("n_estimators", 50),
        contamination=if_cfg.get("contamination", 0.15),
        max_features=if_cfg.get("max_features", 0.5),
    )
    ifd.fit(X_train)
    m = ifd.evaluate(X_test, y_test)
    logger.info(f"  IF   F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  PR={m['pr_auc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "Isolation Forest", **m})

    logger.info("Training One-Class SVM …")
    t0 = time.time()
    ocsvm_cfg = cfg.get("ocsvm", {})
    ocd = OCSVMDetector(
        nu=ocsvm_cfg.get("nu", 0.2),
        kernel=ocsvm_cfg.get("kernel", "rbf"),
        gamma=ocsvm_cfg.get("gamma", 0.1),
        train_cap=ocsvm_cfg.get("train_cap", 8000),
    )
    ocd.fit(X_train)
    m = ocd.evaluate(X_test, y_test)
    logger.info(f"  OCSVM F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  PR={m['pr_auc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "One-Class SVM", **m})

    logger.info("Training Gaussian Mixture Model (Model A) …")
    t0 = time.time()
    gmm_cfg = cfg.get("gmm", {})
    gmm = GMMDetector(
        n_components=gmm_cfg.get("n_components", 12),
        covariance_type=gmm_cfg.get("covariance_type", "full"),
        threshold_percentile=gmm_cfg.get("threshold_percentile", 11),
        n_init=gmm_cfg.get("n_init", 3),
        max_iter=gmm_cfg.get("max_iter", 200),
        tune_subsample=gmm_cfg.get("tune_subsample", 80_000),
    )
    gmm.fit(X_train)
    m = gmm.evaluate(X_test, y_test)
    logger.info(f"  GMM  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  PR={m['pr_auc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "GMM (Model A)", **m})

    gmm.save(models_dir / "model_a_gmm.pkl")
    np.save(models_dir / "model_a_threshold.npy", np.array(gmm.threshold))

    results_csv = results_dir / "model_a_metrics.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    logger.info(f"Phase 1 results → {results_csv}")

    return gmm


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Sequence detector (configurable)
# ──────────────────────────────────────────────────────────────────────────────

def _make_seq_detector(cfg: dict):
    """Instantiate the configured sequence detector."""
    model_name = cfg.get("phase2", {}).get("model", "lstm_ae")
    if model_name not in SEQ_MODEL_CONFIG_KEYS:
        raise KeyError(
            f"phase2.model must be one of {sorted(SEQ_MODEL_CONFIG_KEYS)}, got '{model_name}'"
        )
    section, keys = SEQ_MODEL_CONFIG_KEYS[model_name]
    sec = cfg.get(section, {})
    kwargs = {k: sec[k] for k in keys if k in sec}
    return create_detector(model_name, **kwargs), model_name


def _seq_windows(cfg: dict, X_train: np.ndarray):
    """Build benign-only training windows + temporal validation tail."""
    model_name = cfg.get("phase2", {}).get("model", "lstm_ae")
    section = SEQ_MODEL_CONFIG_KEYS[model_name][0]
    sec = cfg.get(section, {})
    W = int(sec.get("window_size", 50))
    stride = int(cfg.get("phase2", {}).get("train_stride", 5))
    val_split = float(sec.get("val_split", 0.1))
    max_windows = cfg.get("phase2", {}).get("max_windows")

    windows = build_train_windows(X_train, window_size=W, stride=stride,
                                  max_windows=max_windows, seed=cfg.get("seed", 42))
    n_val = max(1, int(len(windows) * val_split)) if len(windows) > 10 else max(1, len(windows) // 5)
    return windows[:-n_val], windows[-n_val:], W


def train_phase2(cfg: dict, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 models_dir: pathlib.Path, results_dir: pathlib.Path):
    """
    Train the configured sequence detector on benign-only windows.

    Evaluation is FLOW-LEVEL (primary): per-timestep reconstruction errors are
    mapped back to flows; the flow threshold is calibrated on a benign slice
    of the training data. Window-level metrics are reported for reference.
    """

    logger.info("=" * 60)
    logger.info("PHASE 2 — Sequence anomaly detection")
    logger.info("=" * 60)

    det, model_name = _make_seq_detector(cfg)
    section = SEQ_MODEL_CONFIG_KEYS[model_name][0]
    sec = cfg.get(section, {})
    p2 = cfg.get("phase2", {})

    X_tr_win, X_val_win, W = _seq_windows(cfg, X_train)
    logger.info(f"  Windows: train={len(X_tr_win):,}  val={len(X_val_win):,}  (W={W})")

    t0 = time.time()
    det.fit(
        X_tr_win, X_val_seq=X_val_win,
        epochs=int(sec.get("epochs", 100)),
        batch_size=int(sec.get("batch_size", 256)),
        patience=int(sec.get("patience", 10)),
    )
    logger.info(f"  Training done [{time.time()-t0:.1f}s]")

    det.set_threshold(X_val_win, percentile=float(sec.get("threshold_pct", 95)))

    # ── Flow-level evaluation (primary) ───────────────────────────────────────
    n_cal = min(50_000, int(len(X_train) * 0.2))
    eval_stride = int(p2.get("eval_stride", 1))
    det.set_flow_threshold(X_train[:n_cal],
                           percentile=float(sec.get("threshold_pct", 95)),
                           stride=eval_stride)
    flow_metrics = det.evaluate_flows(X_test, y_test, stride=eval_stride)
    logger.info(
        f"  [{model_name}] FLOW-LEVEL  F1={flow_metrics['f1']:.4f}  "
        f"AUC={flow_metrics['auc']:.4f}  PR-AUC={flow_metrics['pr_auc']:.4f}  "
        f"FPR={flow_metrics['fpr']:.4f}"
    )

    # ── Window-level metrics (reference) ──────────────────────────────────────
    try:
        windows, coverage = build_eval_windows(X_test, W, stride=max(W // 2, 1))
        w_labels = window_labels_from_flows(y_test, coverage)
        win_metrics = det.evaluate(windows, w_labels)
        logger.info(
            f"  [{model_name}] WINDOW-LEVEL (ref)  F1={win_metrics['f1']:.4f}  "
            f"AUC={win_metrics['auc']:.4f}"
        )
    except Exception as e:  # noqa: BLE001 — reporting must never kill training
        win_metrics = {}
        logger.warning(f"  Window-level evaluation skipped: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    det.save(models_dir / f"{model_name}_best")
    pd.DataFrame([{**flow_metrics, "level": "flow"},
                  {**win_metrics, "level": "window"}]).to_csv(
        results_dir / f"{model_name}_metrics.csv", index=False
    )

    return det, model_name


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Leak-free calibrated fusion
# ──────────────────────────────────────────────────────────────────────────────

def train_phase3(cfg: dict, gmm: GMMDetector, seq_det, seq_model_name: str,
                 X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 models_dir: pathlib.Path, results_dir: pathlib.Path) -> FusionModel:
    """
    Calibrated, CV-selected meta-fusion without test leakage.

    Protocol: labeled pool → disjoint meta-fit / eval-only splits. Isotonic
    calibrators and the meta-learner see ONLY the meta-fit part; all reported
    metrics come from eval-only.
    """

    from sklearn.isotonic import IsotonicRegression

    logger.info("=" * 60)
    logger.info("PHASE 3 — Hybrid fusion (leak-free protocol)")
    logger.info("=" * 60)

    hy = cfg.get("hybrid", {})
    seed = cfg.get("seed", 42)
    seq_stride = int(cfg.get("phase2", {}).get("eval_stride", 1))

    # ── Raw base scores over the labeled pool ─────────────────────────────────
    logger.info("Scoring labeled pool …")
    t0 = time.time()
    gmm_raw = -gmm._clf.score_samples(X_test)
    seq_raw = seq_det.score_flows(X_test, stride=seq_stride)
    raw_pool = np.column_stack([gmm_raw, seq_raw])
    logger.info(f"  Base scores ready [{time.time()-t0:.1f}s]")

    # ── Disjoint splits ───────────────────────────────────────────────────────
    meta_idx, eval_idx = labeled_holdout_split(
        y_test, meta_frac=float(hy.get("meta_train_frac", 0.4)), seed=seed
    )
    y_fit = y_test[meta_idx]
    logger.info(
        f"  Meta-fit: {len(meta_idx):,} ({y_fit.mean():.2%} attacks)  |  "
        f"Eval-only: {len(eval_idx):,} ({y_test[eval_idx].mean():.2%} attacks)"
    )

    # ── Isotonic calibration fitted on meta-fit ONLY ──────────────────────────
    def apply_cals(raw: np.ndarray) -> np.ndarray:
        return np.column_stack([raw] + [iso.predict(raw[:, j])
                                        for j, iso in enumerate(calibrators)])

    calibrators = []
    for j in range(raw_pool.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_pool[meta_idx][:, j], y_fit)
        calibrators.append(iso)

    X_meta_fit  = apply_cals(raw_pool[meta_idx])
    X_meta_eval = apply_cals(raw_pool[eval_idx])

    # ── Candidate selection by CV inside meta-fit ─────────────────────────────
    candidates = tuple(hy.get("candidates", ["rf", "lr", "weighted"]))
    best_name, meta, cv_rows = select_meta_learner(
        X_meta_fit, y_fit, candidates=candidates,
        cv_folds=int(hy.get("cv_folds", 3)), seed=seed,
    )
    pd.DataFrame(cv_rows).to_csv(results_dir / "fusion_cv_selection.csv", index=False)
    logger.info(f"  Meta-learner selected: {best_name} "
                f"(CV PR-AUC {max(r['cv_pr_auc_mean'] for r in cv_rows):.4f})")

    # ── Final evaluation on eval-only ─────────────────────────────────────────
    threshold = float(hy.get("threshold", 0.5))
    fusion = FusionModel(
        gmm_detector=gmm, seq_detector=seq_det,
        calibrators=calibrators, meta_learner=meta,
        alpha=float(hy.get("alpha", 0.7)), threshold=threshold,
        seq_stride=seq_stride,
    )
    proba = meta.predict_proba(X_meta_eval)[:, 1]
    metrics = compute_metrics(y_test[eval_idx], (proba >= threshold).astype(int), proba)
    logger.info(
        f"  Fusion[{best_name}] EVAL-ONLY  F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc_roc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
        f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
    )

    # Honest fallbacks on the same eval-only part
    wac = WeightedAverageClassifier().fit(X_meta_fit, y_fit)
    w_proba = wac.predict_proba(X_meta_eval)[:, 1]
    w_metrics = compute_metrics(y_test[eval_idx], (w_proba >= threshold).astype(int), w_proba)
    logger.info(
        f"  Weighted(α={wac.alpha:.2f}) EVAL-ONLY  F1={w_metrics['f1']:.4f}  "
        f"AUC={w_metrics['auc_roc']:.4f}  PR-AUC={w_metrics['pr_auc']:.4f}"
    )

    # Single-base references
    for j, base_name in enumerate(["gmm", seq_model_name]):
        b_proba = X_meta_eval[:, j]
        b_metrics = compute_metrics(y_test[eval_idx], (b_proba >= threshold).astype(int), b_proba)
        logger.info(f"  Base[{base_name}] EVAL-ONLY  F1={b_metrics['f1']:.4f}  "
                    f"AUC={b_metrics['auc_roc']:.4f}  PR-AUC={b_metrics['pr_auc']:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    fusion.save(models_dir / "model_c_fusion")
    joblib.dump({"sequence_model": seq_model_name}, models_dir / "model_c_seq_meta.pkl")
    np.save(models_dir / "model_c_eval_mask.npy", eval_idx)
    pd.DataFrame([
        {"model": f"Fusion[{best_name}]", **metrics},
        {"model": f"Weighted(α={wac.alpha:.2f})", **w_metrics},
    ]).to_csv(results_dir / "model_c_metrics.csv", index=False)
    logger.info(f"Phase 3 results → {results_dir / 'model_c_metrics.csv'}")

    return fusion


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/default.yaml", phases: list[int] | None = None) -> None:
    if phases is None:
        phases = [1, 2, 3]

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    root        = pathlib.Path(__file__).parent.parent
    prep_dir    = root / cfg["paths"]["prep_dir"]
    models_dir  = root / cfg["paths"]["models_dir"]
    results_dir = root / cfg["paths"]["results_dir"]
    ensure_dirs(models_dir, results_dir)

    X_train, X_test, y_test = _load_arrays(prep_dir)

    gmm = None
    seq_det = None
    seq_model_name = cfg.get("phase2", {}).get("model", "lstm_ae")

    if 1 in phases:
        gmm = train_phase1(cfg, X_train, X_test, y_test, models_dir, results_dir)

    if 2 in phases:
        seq_det, seq_model_name = train_phase2(
            cfg, X_train, X_test, y_test, models_dir, results_dir
        )

    if 3 in phases:
        if gmm is None:
            logger.info("Loading saved GMM for Phase 3 …")
            gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        if seq_det is None:
            logger.info(f"Loading saved sequence detector ({seq_model_name}) for Phase 3 …")
            seq_det = load_detector(seq_model_name, models_dir / f"{seq_model_name}_best")
        train_phase3(cfg, gmm, seq_det, seq_model_name,
                     X_train, X_test, y_test, models_dir, results_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly-based IDS models")
    parser.add_argument("--config",  default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--phases",  nargs="+", type=int, default=[1, 2, 3],
                        help="Phases to run, e.g. --phases 1 3")
    args = parser.parse_args()
    run(args.config, args.phases)
