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
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.features import FeatureEngineer
from deepguard.models import GMMDetector
from deepguard.utils import ensure_dirs, load_config, setup_logging

logger = setup_logging()

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

    # The fitted FeatureEngineer is written by `main.py preprocess`.
    fe_path = prep_dir / "feature_engineer.pkl"
    if not fe_path.exists():
        logger.error(
            "FeatureEngineer pickle not found. Run `python main.py preprocess` first."
        )
        sys.exit(1)

    logger.info("Applying preprocessing...")
    fe = FeatureEngineer.load(fe_path)
    try:
        df.columns = df.columns.str.strip()
        # Keep exactly the raw columns seen at fit time, in order — drops
        # labels/IDs/extra fields and guarantees shape alignment downstream.
        missing = [c for c in fe._raw_feature_names if c not in df.columns]
        if missing:
            logger.error(f"Input is missing required columns: {missing}")
            sys.exit(1)
        X = fe.transform(df[fe._raw_feature_names])
    except SystemExit:
        raise
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
        from deepguard.fusion import FusionModel
        from deepguard.models_deep import load_detector

        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        seq_name = joblib.load(models_dir / "model_c_seq_meta.pkl")["sequence_model"]
        seq_det = load_detector(seq_name, models_dir / f"{seq_name}_best")
        fusion = FusionModel.load(models_dir / "model_c_fusion", gmm, seq_det)

        scores = fusion.score(X)
        preds = (scores >= fusion.threshold).astype(int)
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
