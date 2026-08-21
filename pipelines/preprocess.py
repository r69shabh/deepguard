"""
pipelines/preprocess.py
=======================
End-to-end preprocessing: raw CICIDS-2017 CSVs → model-ready arrays.

Replaces notebooks/stage2_preprocessing.py as the canonical, reproducible path.
Produces in outputs/preprocessing/:
    X_train.npy           processed benign-only training features
    X_test.npy            processed pooled test features
    y_test.npy            binary labels (1 = attack)
    attack_types.csv      multiclass labels for per-attack analysis
    feature_names.txt     ordered output feature names
    feature_engineer.pkl  fitted FeatureEngineer (required by detect pipeline)

Usage
-----
    python main.py preprocess
    python main.py preprocess --data-dir data/CICIDS2017
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.data_loaders import LABEL_COL, clean_raw, load_raw, make_splits
from deepguard.features import FeatureEngineer
from deepguard.utils import ensure_dirs, load_config, set_seed, setup_logging

logger = setup_logging()


def run(data_dir: str | None = None, config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    root = pathlib.Path(__file__).parent.parent
    data_dir = pathlib.Path(data_dir or root / cfg["paths"]["data_dir"])
    prep_dir = root / cfg["paths"]["prep_dir"]
    ensure_dirs(prep_dir)

    feat_cfg = cfg.get("features", {})

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading raw CSVs from {data_dir} …")
    t0 = time.time()
    raw = load_raw(data_dir)
    logger.info(f"  Raw shape: {raw.shape}  [{time.time() - t0:.1f}s]")

    # ── Clean (Steps 1–3) ─────────────────────────────────────────────────────
    logger.info("Cleaning …")
    t0 = time.time()
    clean = clean_raw(raw)
    n_benign = int((clean[LABEL_COL] == "BENIGN").sum())
    logger.info(
        f"  Clean shape: {clean.shape}  "
        f"(benign {n_benign:,}, attack {len(clean) - n_benign:,})  [{time.time() - t0:.1f}s]"
    )
    del raw

    # ── Split (benign-only train / mixed test) ────────────────────────────────
    splits = make_splits(clean, test_size=0.2, seed=cfg.get("seed", 42))
    X_train_benign: "np.ndarray" = splits["X_train_benign"]
    X_test_df = splits["X_test"]
    y_test: np.ndarray = splits["y_test"]  # type: ignore[assignment]
    attack_types = splits["attack_types"]
    logger.info(
        f"  Train(benign): {len(X_train_benign):,}  |  Test: {len(X_test_df):,}  "
        f"({y_test.mean():.2%} attacks)"
    )
    del clean

    # ── Feature engineering (fit on benign train only) ────────────────────────
    logger.info("Fitting FeatureEngineer on benign train …")
    fe = FeatureEngineer(
        iqr_lower=feat_cfg.get("iqr_lower", 0.01),
        iqr_upper=feat_cfg.get("iqr_upper", 0.99),
        skew_threshold=feat_cfg.get("skew_threshold", 1.0),
        corr_threshold=feat_cfg.get("corr_threshold", 0.95),
    )
    t0 = time.time()
    X_train = fe.fit_transform(X_train_benign)
    logger.info(f"  X_train: {X_train.shape}  [{time.time() - t0:.1f}s]")

    logger.info("Transforming test pool …")
    X_test = fe.transform(X_test_df)

    # ── Save artifacts ────────────────────────────────────────────────────────
    np.save(prep_dir / "X_train.npy", X_train)
    np.save(prep_dir / "X_test.npy", X_test)
    np.save(prep_dir / "y_test.npy", y_test)
    np.save(prep_dir / "attack_types.npy", np.asarray(attack_types, dtype=object))
    with open(prep_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(fe.get_feature_names()) + "\n")
    fe.save(prep_dir / "feature_engineer.pkl")

    logger.info(f"All artifacts written to {prep_dir}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CICIDS-2017 CSVs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory")
    args = parser.parse_args()
    run(args.data_dir, args.config)
