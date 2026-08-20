"""
pipelines/detect.py
===================
Inference pipeline for running anomaly detection on new network flow CSV data.

Usage
-----
    python main.py detect --input new_flows.csv --output predictions.csv
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

from deepguard.utils import setup_logging, load_config, ensure_dirs
from deepguard.features import FeatureEngineer
from deepguard.models import GMMDetector, LSTMAEDetector, HybridDetector

logger = setup_logging()

def load_feature_engineer(prep_dir: pathlib.Path) -> FeatureEngineer:
    """Load the fitted FeatureEngineer from saved states."""
    fe = FeatureEngineer()
    # Mock loading from saved artifacts since we don't have a true save/load on FeatureEngineer yet.
    # We will just load the necessary parts. If it doesn't exist, we skip.
    fe.is_fitted_ = True
    # For a true project, FeatureEngineer should be pickled during training.
    # Since we are assuming it's already run in stage2, let's try to load if a pickle exists,
    # or fallback to something graceful.
    fe_path = prep_dir / "feature_engineer.pkl"
    if fe_path.exists():
        return joblib.load(fe_path)
    logger.warning("FeatureEngineer pickle not found. Preprocessing might fail if not fully fitted.")
    return fe

def run(input_csv: str, output_csv: str, config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    
    ROOT       = pathlib.Path(__file__).parent.parent
    prep_dir   = ROOT / cfg["paths"]["prep_dir"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    ensure_dirs(pathlib.Path(output_csv).parent)

    logger.info(f"Loading input data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        sys.exit(1)

    # Note: In a fully productized version, FeatureEngineer would be pickled during train.
    fe_path = prep_dir / "feature_engineer.pkl"
    if not fe_path.exists():
        logger.error("FeatureEngineer pickle not found. Run training/preprocessing first.")
        sys.exit(1)
    
    logger.info("Applying preprocessing...")
    fe = joblib.load(fe_path)
    try:
        X = fe.transform(df)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    model_type = cfg.get("detect", {}).get("model", "hybrid")
    logger.info(f"Running detection using {model_type.upper()}...")
    
    t0 = time.time()
    if model_type == "gmm":
        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        scores = gmm.score(X)
        preds = gmm.predict(X)
    elif model_type == "hybrid":
        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
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
        scores = hybrid.score(X)
        preds = (scores >= h_params["threshold"]).astype(int)
    else:
        logger.error(f"Unsupported model type for detection: {model_type}")
        sys.exit(1)
        
    logger.info(f"Detection finished in {time.time() - t0:.2f}s.")

    out_df = df.copy()
    if cfg.get("detect", {}).get("output_scores", True):
        out_df["anomaly_score"] = scores
    out_df["is_attack"] = preds

    out_df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")
    
    n_attacks = int(preds.sum())
    logger.info(f"Summary: {n_attacks} attacks detected out of {len(preds)} flows ({(n_attacks/len(preds))*100:.2f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on new data")
    parser.add_argument("--input", required=True, help="Input CSV file with network flows")
    parser.add_argument("--output", required=True, help="Output CSV file for predictions")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run(args.input, args.output, args.config)
