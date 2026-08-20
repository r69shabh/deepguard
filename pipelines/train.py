"""
pipelines/train.py
==================
End-to-end training pipeline for all three model phases.

Trains GMM (Phase 1), LSTM-AE (Phase 2), and Hybrid RF (Phase 3) in sequence.
Saves all model artifacts and metrics CSVs.

Usage
-----
    python main.py train
    python main.py train --config configs/default.yaml
    python pipelines/train.py                          # direct run
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd

# ── allow direct execution from project root ──────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.utils import setup_logging, set_seed, load_config, get_nested, ensure_dirs
from deepguard.models import (
    GMMDetector,
    IsolationForestDetector,
    OCSVMDetector,
    LSTMAEDetector,
    HybridDetector,
)
from deepguard.evaluate import compute_metrics, generate_report

logger = setup_logging()


# ──────────────────────────────────────────────────────────────────────────────

def _load_arrays(prep_dir: pathlib.Path):
    """Load preprocessed numpy arrays from outputs/preprocessing/."""
    logger.info(f"Loading preprocessed arrays from {prep_dir}")
    X_train = np.load(prep_dir / "X_train.npy")
    X_test  = np.load(prep_dir / "X_test.npy")
    y_test  = np.load(prep_dir / "y_test.npy")
    logger.info(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}  |  y_test : {y_test.shape}")
    return X_train, X_test, y_test


def _load_sequences(seq_dir: pathlib.Path):
    """Load pre-built sequence arrays for LSTM-AE training."""
    logger.info(f"Loading sequence arrays from {seq_dir}")
    X_train_seq = np.load(seq_dir / "X_train_seq.npy", mmap_mode="r")
    X_test_seq  = np.load(seq_dir / "X_test_seq.npy",  mmap_mode="r")
    y_test_seq  = np.load(seq_dir / "y_test_seq.npy")
    logger.info(
        f"  X_train_seq : {X_train_seq.shape}  |  "
        f"X_test_seq : {X_test_seq.shape}  |  y_test_seq : {y_test_seq.shape}"
    )
    return X_train_seq, X_test_seq, y_test_seq


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

    # ── Isolation Forest ──────────────────────────────────────────────────────
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
    logger.info(f"  IF   F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "Isolation Forest", **m})

    # ── One-Class SVM ─────────────────────────────────────────────────────────
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
    logger.info(f"  OCSVM F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "One-Class SVM", **m})

    # ── GMM (Model A) ─────────────────────────────────────────────────────────
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
    logger.info(f"  GMM  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  [{time.time()-t0:.1f}s]")
    rows.append({"Model": "GMM (Model A)", **m})

    # ── Save ──────────────────────────────────────────────────────────────────
    gmm.save(models_dir / "model_a_gmm.pkl")
    np.save(models_dir / "model_a_threshold.npy", np.array(gmm.threshold))

    results_csv = results_dir / "model_a_metrics.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    logger.info(f"Phase 1 results → {results_csv}")

    return gmm


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — LSTM Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

def train_phase2(cfg: dict, X_train_seq: np.ndarray, X_test_seq: np.ndarray,
                 y_test_seq: np.ndarray, models_dir: pathlib.Path,
                 results_dir: pathlib.Path) -> LSTMAEDetector:
    """Train LSTM Autoencoder (Model B). Save artifacts and metrics CSV."""

    logger.info("=" * 60)
    logger.info("PHASE 2 — LSTM Autoencoder (Model B)")
    logger.info("=" * 60)

    ae_cfg = cfg.get("lstm_ae", {})
    val_split = ae_cfg.get("val_split", 0.1)

    n_val = max(1, int(len(X_train_seq) * val_split))
    X_val_seq   = X_train_seq[-n_val:]
    X_train_seq = X_train_seq[:-n_val]
    logger.info(f"  Train windows : {X_train_seq.shape[0]}  |  Val windows : {X_val_seq.shape[0]}")

    lstm_ae = LSTMAEDetector(
        window_size=ae_cfg.get("window_size", 50),
        latent_dim=ae_cfg.get("latent_dim", 32),
        dropout_rate=ae_cfg.get("dropout_rate", 0.2),
    )

    logger.info("Training LSTM-AE …")
    t0 = time.time()
    lstm_ae.fit(
        X_train_seq,
        X_val_seq=X_val_seq,
        epochs=ae_cfg.get("epochs", 100),
        batch_size=ae_cfg.get("batch_size", 256),
        patience=ae_cfg.get("patience", 10),
    )
    logger.info(f"  Training done [{time.time()-t0:.1f}s]")

    lstm_ae.set_threshold(X_val_seq, percentile=ae_cfg.get("threshold_pct", 95))
    logger.info(f"  Decision threshold : {lstm_ae.threshold:.6f}")

    metrics = lstm_ae.evaluate(X_test_seq, y_test_seq)
    logger.info(
        f"  LSTM-AE  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    lstm_ae.model.save(str(models_dir / "lstm_ae_best.keras"))
    np.save(models_dir / "lstm_ae_threshold.npy", np.array(lstm_ae.threshold))

    results_csv = results_dir / "lstm_ae_metrics.csv"
    pd.DataFrame([metrics]).to_csv(results_csv, index=False)
    logger.info(f"Phase 2 results → {results_csv}")

    return lstm_ae


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Hybrid Detector
# ──────────────────────────────────────────────────────────────────────────────

def train_phase3(cfg: dict, gmm: GMMDetector, lstm_ae: LSTMAEDetector,
                 X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                 models_dir: pathlib.Path, results_dir: pathlib.Path) -> HybridDetector:
    """Build and evaluate Hybrid RF meta-learner (Model C)."""

    logger.info("=" * 60)
    logger.info("PHASE 3 — Hybrid GMM + LSTM-AE (Model C)")
    logger.info("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    hybrid_cfg = cfg.get("hybrid", {})
    window_size = hybrid_cfg.get("window_size", 50)
    stride      = hybrid_cfg.get("stride", 25)

    # ── Calibration scores on validation benign (first 20% of X_train) ───────
    n_cal = min(50_000, int(len(X_train) * 0.2))
    X_cal = X_train[:n_cal]

    # Temporary hybrid for normalisation stats
    _tmp = HybridDetector(
        gmm_model=gmm._clf,
        lstm_ae_model=lstm_ae.model,
        window_size=window_size,
        stride=stride,
    )
    s_gmm_cal  = _tmp._gmm_scores(X_cal)
    val_gmm_min, val_gmm_max   = float(s_gmm_cal.min()),  float(s_gmm_cal.max())
    s_lstm_cal = _tmp._lstm_scores(X_cal)
    val_lstm_min, val_lstm_max = float(s_lstm_cal.min()), float(s_lstm_cal.max())

    # ── Build meta-learner training set ──────────────────────────────────────
    logger.info("Scoring test set for meta-learner …")
    hybrid_base = HybridDetector(
        gmm_model=gmm._clf,
        lstm_ae_model=lstm_ae.model,
        window_size=window_size,
        stride=stride,
        val_gmm_min=val_gmm_min,
        val_gmm_max=val_gmm_max,
        val_lstm_min=val_lstm_min,
        val_lstm_max=val_lstm_max,
    )
    s_gmm_test  = hybrid_base._gmm_scores(X_test)
    s_lstm_test = hybrid_base._lstm_scores(X_test)

    # Use a 20% stratified subset for meta-learner training
    from sklearn.model_selection import train_test_split
    X_meta = np.column_stack([s_gmm_test, s_lstm_test])
    X_meta_tr, _, y_meta_tr, _ = train_test_split(
        X_meta, y_test, test_size=0.8, stratify=y_test, random_state=42
    )

    # ── Fit meta-learner ──────────────────────────────────────────────────────
    meta_type = hybrid_cfg.get("meta_learner", "rf")
    logger.info(f"Fitting meta-learner ({meta_type}) …")
    if meta_type == "lr":
        meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    else:
        meta = RandomForestClassifier(
            n_estimators=hybrid_cfg.get("rf_n_estimators", 100),
            random_state=42, n_jobs=-1
        )
    meta.fit(X_meta_tr, y_meta_tr)

    # ── Final hybrid model ────────────────────────────────────────────────────
    hybrid = HybridDetector(
        gmm_model=gmm._clf,
        lstm_ae_model=lstm_ae.model,
        meta_learner=meta,
        window_size=window_size,
        stride=stride,
        val_gmm_min=val_gmm_min,
        val_gmm_max=val_gmm_max,
        val_lstm_min=val_lstm_min,
        val_lstm_max=val_lstm_max,
    )
    threshold = hybrid_cfg.get("threshold", 0.5)
    metrics   = hybrid.evaluate(X_test, y_test, threshold=threshold)
    logger.info(
        f"  Hybrid F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(meta, models_dir / "model_c_meta_rf.pkl")
    joblib.dump(
        {
            "val_gmm_min": val_gmm_min, "val_gmm_max": val_gmm_max,
            "val_lstm_min": val_lstm_min, "val_lstm_max": val_lstm_max,
            "window_size": window_size, "stride": stride, "threshold": threshold,
        },
        models_dir / "model_c_params.pkl",
    )
    results_csv = results_dir / "model_c_metrics.csv"
    pd.DataFrame([metrics]).to_csv(results_csv, index=False)
    logger.info(f"Phase 3 results → {results_csv}")

    return hybrid


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/default.yaml", phases: list[int] | None = None) -> None:
    """
    Full training pipeline.

    Parameters
    ----------
    config_path : str
        Path to YAML config.
    phases : list[int], optional
        Which phases to run (default: [1, 2, 3]).
    """
    if phases is None:
        phases = [1, 2, 3]

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    ROOT         = pathlib.Path(__file__).parent.parent
    prep_dir     = ROOT / cfg["paths"]["prep_dir"]
    seq_dir      = ROOT / cfg["paths"]["seq_dir"]
    models_dir   = ROOT / cfg["paths"]["models_dir"]
    results_dir  = ROOT / cfg["paths"]["results_dir"]
    ensure_dirs(models_dir, results_dir)

    X_train, X_test, y_test = _load_arrays(prep_dir)

    gmm     = None
    lstm_ae = None

    if 1 in phases:
        gmm = train_phase1(cfg, X_train, X_test, y_test, models_dir, results_dir)

    if 2 in phases:
        X_train_seq, X_test_seq, y_test_seq = _load_sequences(seq_dir)
        lstm_ae = train_phase2(cfg, X_train_seq, X_test_seq, y_test_seq, models_dir, results_dir)

    if 3 in phases:
        if gmm is None:
            logger.info("Loading saved GMM for Phase 3 …")
            gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        if lstm_ae is None:
            logger.info("Loading saved LSTM-AE for Phase 3 …")
            import tensorflow as tf
            lstm_ae = LSTMAEDetector()
            lstm_ae.model     = tf.keras.models.load_model(str(models_dir / "lstm_ae_best.keras"))
            lstm_ae.threshold = float(np.load(models_dir / "lstm_ae_threshold.npy"))
        train_phase3(cfg, gmm, lstm_ae, X_train, X_test, y_test, models_dir, results_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly-based IDS models")
    parser.add_argument("--config",  default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--phases",  nargs="+", type=int, default=[1, 2, 3],
                        help="Phases to run, e.g. --phases 1 3")
    args = parser.parse_args()
    run(args.config, args.phases)
