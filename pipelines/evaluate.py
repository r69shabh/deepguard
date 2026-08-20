"""
pipelines/evaluate.py
=====================
Evaluation pipeline for saved models.

Loads GMM and Hybrid models, runs evaluation on the test set,
and generates plots (ROC, CM, Score Distributions) and a text report.

Usage
-----
    python main.py evaluate
    python pipelines/evaluate.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.utils import setup_logging, set_seed, load_config, ensure_dirs
from deepguard.models import GMMDetector, LSTMAEDetector, HybridDetector
from deepguard.evaluate import (
    compute_metrics,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    generate_report,
)

logger = setup_logging()


def run(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    ROOT        = pathlib.Path(__file__).parent.parent
    prep_dir    = ROOT / cfg["paths"]["prep_dir"]
    models_dir  = ROOT / cfg["paths"]["models_dir"]
    outputs_dir = ROOT / cfg["paths"]["outputs_dir"]
    ensure_dirs(outputs_dir)

    logger.info("Loading test data...")
    X_test = np.load(prep_dir / "X_test.npy")
    y_test = np.load(prep_dir / "y_test.npy")

    # 1. Evaluate GMM (Model A)
    logger.info("Evaluating GMM (Model A)...")
    try:
        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        y_scores_gmm = gmm.score(X_test)
        y_pred_gmm   = gmm.predict(X_test)
        metrics_gmm  = compute_metrics(y_test, y_pred_gmm, y_scores_gmm)

        if cfg["evaluation"].get("roc_plot"):
            plot_roc_curve(y_test, y_scores_gmm, label="GMM", save_path=outputs_dir / "gmm_roc.png")
        if cfg["evaluation"].get("cm_plot"):
            plot_confusion_matrix(y_test, y_pred_gmm, save_path=outputs_dir / "gmm_cm.png")
        if cfg["evaluation"].get("score_dist_plot"):
            plot_score_distribution(
                y_scores_gmm[y_test == 0], y_scores_gmm[y_test == 1],
                save_path=outputs_dir / "gmm_score_dist.png",
                threshold=gmm.threshold, model_name="GMM"
            )
        if cfg["evaluation"].get("report_txt"):
            generate_report("GMM (Model A)", metrics_gmm, outputs_dir / "gmm_report.txt")
    except Exception as e:
        logger.error(f"Failed to evaluate GMM: {e}")

    # 2. Evaluate Hybrid (Model C)
    logger.info("Evaluating Hybrid Detector (Model C)...")
    try:
        import tensorflow as tf
        lstm_ae = LSTMAEDetector()
        lstm_ae.model = tf.keras.models.load_model(str(models_dir / "lstm_ae_best.keras"))
        
        meta_learner = joblib.load(models_dir / "model_c_meta_rf.pkl")
        h_params     = joblib.load(models_dir / "model_c_params.pkl")

        hybrid = HybridDetector(
            gmm_model=gmm._clf,
            lstm_ae_model=lstm_ae.model,
            meta_learner=meta_learner,
            window_size=h_params["window_size"],
            stride=h_params["stride"],
            val_gmm_min=h_params["val_gmm_min"],
            val_gmm_max=h_params["val_gmm_max"],
            val_lstm_min=h_params["val_lstm_min"],
            val_lstm_max=h_params["val_lstm_max"],
        )
        threshold = h_params["threshold"]

        y_scores_h = hybrid.score(X_test)
        y_pred_h   = (y_scores_h >= threshold).astype(int)
        metrics_h  = compute_metrics(y_test, y_pred_h, y_scores_h)

        if cfg["evaluation"].get("roc_plot"):
            plot_roc_curve(y_test, y_scores_h, label="Hybrid", save_path=outputs_dir / "hybrid_roc.png")
        if cfg["evaluation"].get("cm_plot"):
            plot_confusion_matrix(y_test, y_pred_h, save_path=outputs_dir / "hybrid_cm.png")
        if cfg["evaluation"].get("score_dist_plot"):
            plot_score_distribution(
                y_scores_h[y_test == 0], y_scores_h[y_test == 1],
                save_path=outputs_dir / "hybrid_score_dist.png",
                threshold=threshold, model_name="Hybrid Detector"
            )
        if cfg["evaluation"].get("report_txt"):
            generate_report("Hybrid Detector (Model C)", metrics_h, outputs_dir / "hybrid_report.txt")
    except Exception as e:
        logger.error(f"Failed to evaluate Hybrid: {e}")

    logger.info("Evaluation complete. Check outputs/models/ for plots and reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly-based IDS models")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)
