"""
pipelines/robustness.py
=======================
Adversarial robustness evaluation for saved models.

Runs mimicry-evasion and noise-FPR experiments against the GMM and the hybrid
fusion scorer on the preprocessed arrays, and writes CSV reports:

    results/robustness_{model}_mimicry.csv
    results/robustness_{model}_noise.csv

Usage
-----
    python main.py robustness
    python main.py robustness --models gmm hybrid
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import joblib
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.models import GMMDetector
from deepguard.robustness import robustness_report
from deepguard.utils import ensure_dirs, load_config, set_seed, setup_logging

logger = setup_logging()


def _fusion_scorer(models_dir: pathlib.Path):
    """Build the hybrid scorer callable from persisted artifacts."""
    from deepguard.fusion import FusionModel
    from deepguard.models_deep import load_detector

    gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
    seq_name = joblib.load(models_dir / "model_c_seq_meta.pkl")["sequence_model"]
    seq_det = load_detector(seq_name, models_dir / f"{seq_name}_best")
    fusion = FusionModel.load(models_dir / "model_c_fusion", gmm, seq_det)
    return (lambda X: np.asarray(fusion.score(X))), float(fusion.threshold)


def run(config_path: str = "configs/default.yaml",
        models: list[str] | None = None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    root = pathlib.Path(__file__).parent.parent
    prep_dir = root / cfg["paths"]["prep_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    results_dir = root / cfg["paths"]["results_dir"]
    ensure_dirs(results_dir)

    models = models or ["gmm", "hybrid"]

    logger.info("Loading preprocessed arrays …")
    y_test = np.load(prep_dir / "y_test.npy")
    X_test = np.load(prep_dir / "X_test.npy")
    X_attack = X_test[y_test == 1]
    # Subsample benign to keep experiments fast but representative
    rng = np.random.RandomState(cfg.get("seed", 42))
    benign_idx = np.where(y_test == 0)[0]
    n_ben = min(20_000, len(benign_idx))
    X_benign_eval = X_test[benign_idx[rng.choice(len(benign_idx), n_ben, replace=False)]]

    scorers: dict[str, tuple] = {}
    if "gmm" in models:
        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        # Stored τ is a log-likelihood cutoff (flag when ll < τ); our scorers
        # use higher = more anomalous, so the flag boundary is -τ on -log p.
        tau = float(np.load(models_dir / "model_a_threshold.npy"))
        scorers["gmm"] = ((lambda X: np.asarray(gmm.score(X))), -tau)
    if "hybrid" in models:
        try:
            scorers["hybrid"] = _fusion_scorer(models_dir)
        except Exception as e:  # noqa: BLE001 — report and continue with other models
            logger.error(f"Hybrid scorer unavailable ({e}); run train phase 2+3 first.")

    # Both scorers follow higher = more anomalous
    for name, (scorer_fn, threshold) in scorers.items():
        logger.info(f"Robustness experiment — {name.upper()} …")
        rep = robustness_report(
            X_benign_eval, X_attack,
            lambda X, _fn=scorer_fn: np.asarray(_fn(X)),
            threshold=float(threshold),
            out_prefix=results_dir / f"robustness_{name}",
            seed=cfg.get("seed", 42),
        )

        mim = rep["mimicry"]
        nz = rep["noise"]
        logger.info(f"  Mimicry: detection {mim.iloc[0]['detection_rate']:.2%} → "
                    f"{mim.iloc[-1]['detection_rate']:.2%} at α={mim.iloc[-1]['alpha']}")
        clean = nz["fpr_clean"].iloc[0]
        worst = nz[nz.sigma == nz.sigma.max()]["fpr_mean"].iloc[0]
        logger.info(f"  Noise  : FPR {clean:.2%} (clean) → {worst:.2%} "
                    f"(σ={nz.sigma.max():.2f})")

    logger.info(f"Reports written to {results_dir}/robustness_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=["gmm", "hybrid"],
                        choices=["gmm", "hybrid"])
    args = parser.parse_args()
    run(args.config, args.models)
