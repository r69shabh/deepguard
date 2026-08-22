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

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from deepguard.evaluate import (
    compute_metrics,
    generate_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
)
from deepguard.metrics_ext import (
    bootstrap_ci,
    fpr_at_tpr,
    per_attack_table,
    prevalence_sweep,
)
from deepguard.models import GMMDetector
from deepguard.utils import ensure_dirs, load_config, set_seed, setup_logging

logger = setup_logging()


def _rigor_block(tag: str, y: np.ndarray, scores: np.ndarray, y_pred: np.ndarray,
                 attack_types, cfg: dict, results_dir) -> dict:
    """Compute and persist the extended evaluation block for one scorer."""
    ev_cfg = cfg.get("evaluation", {})
    out: dict = {}

    out["fpr_at_95tpr"] = fpr_at_tpr(y, scores,
                                     target_tpr=float(ev_cfg.get("tpr_target", 0.95)))

    n_boot = int(ev_cfg.get("bootstrap_n", 500))
    for metric_name in ("auc_roc", "pr_auc"):
        try:
            if metric_name == "auc_roc":
                from sklearn.metrics import roc_auc_score as _fn
            else:
                from sklearn.metrics import average_precision_score as _fn
            ci = bootstrap_ci(y, scores, metric_fn=_fn, n_boot=n_boot,
                              seed=cfg.get("seed", 42))
            out[f"{metric_name}_ci95"] = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  Bootstrap CI failed for {metric_name}: {e}")

    if attack_types is not None:
        df_attacks = per_attack_table(y, attack_types, scores, y_pred=y_pred)
        df_attacks.to_csv(results_dir / f"per_attack_{tag}.csv", index=False)
        if "detection_rate" in df_attacks.columns:
            weak = df_attacks[df_attacks["detection_rate"] < 0.5]
            if len(weak):
                names = ", ".join(weak.attack_type.head(5))
                logger.info(f"  Weak spots (<50% detection): {names}")

    try:
        prevs = tuple(ev_cfg.get("prevalences", [0.20, 0.10, 0.05, 0.01]))
        sweep = prevalence_sweep(y, scores, prevalences=prevs, seed=cfg.get("seed", 42))
        sweep.to_csv(results_dir / f"prevalence_sweep_{tag}.csv", index=False)
        low = sweep[sweep.prevalence == sweep.prevalence.min()]
        if len(low):
            r = low.iloc[0]
            out[f"pr_auc@{r['prevalence']:.0%}"] = float(r["pr_auc"])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  Prevalence sweep failed: {e}")

    return out


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

    attack_types = None
    at_path = prep_dir / "attack_types.npy"
    if at_path.exists():
        attack_types = np.load(at_path, allow_pickle=True)
    else:
        logger.info("No attack_types.npy — per-attack tables will be skipped.")

    models_dir.mkdir(parents=True, exist_ok=True)
    ROOT_ = pathlib.Path(__file__).parent.parent
    results_dir = ROOT_ / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate GMM (Model A)
    logger.info("Evaluating GMM (Model A)...")
    metrics_gmm = None
    try:
        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        y_scores_gmm = gmm.score(X_test)
        y_pred_gmm   = gmm.predict(X_test)
        metrics_gmm  = compute_metrics(y_test, y_pred_gmm, y_scores_gmm)

        rigor = _rigor_block("gmm", y_test, y_scores_gmm, y_pred_gmm,
                             attack_types, cfg, results_dir)
        metrics_gmm.update(rigor)
        for k, v in rigor.items():
            logger.info(f"  GMM {k}: {v}")

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

    # 2. Evaluate Hybrid (Model C) — on the held-out eval split only
    logger.info("Evaluating Hybrid Detector (Model C)...")
    try:
        from deepguard.fusion import FusionModel
        from deepguard.models_deep import load_detector

        # The meta-learner was fitted on a disjoint slice; restrict evaluation
        # to the saved eval-only indices so reported metrics stay leak-free.
        mask_path = models_dir / "model_c_eval_mask.npy"
        if mask_path.exists():
            eval_idx = np.load(mask_path)
            X_test_h, y_test_h = X_test[eval_idx], y_test[eval_idx]
            logger.info(f"  Using held-out eval split ({len(eval_idx):,} rows).")
        else:
            X_test_h, y_test_h = X_test, y_test
            logger.warning(
                "  No eval mask found (model_c_eval_mask.npy) — retrain with the "
                "current pipeline for leak-free hybrid numbers."
            )

        gmm = GMMDetector.load(models_dir / "model_a_gmm.pkl")
        seq_name = joblib.load(models_dir / "model_c_seq_meta.pkl")["sequence_model"]
        seq_det = load_detector(seq_name, models_dir / f"{seq_name}_best")
        fusion = FusionModel.load(models_dir / "model_c_fusion", gmm, seq_det)
        threshold = fusion.threshold

        y_scores_h = fusion.score(X_test_h)
        y_pred_h   = (y_scores_h >= threshold).astype(int)
        metrics_h  = compute_metrics(y_test_h, y_pred_h, y_scores_h)

        at_h = attack_types[eval_idx] if (attack_types is not None and mask_path.exists()) else attack_types
        rigor_h = _rigor_block("hybrid", y_test_h, y_scores_h, y_pred_h,
                               at_h, cfg, results_dir)
        metrics_h.update(rigor_h)
        for k, v in rigor_h.items():
            logger.info(f"  Hybrid {k}: {v}")

        if cfg["evaluation"].get("roc_plot"):
            plot_roc_curve(y_test_h, y_scores_h, label="Hybrid", save_path=outputs_dir / "hybrid_roc.png")
        if cfg["evaluation"].get("cm_plot"):
            plot_confusion_matrix(y_test_h, y_pred_h, save_path=outputs_dir / "hybrid_cm.png")
        if cfg["evaluation"].get("score_dist_plot"):
            plot_score_distribution(
                y_scores_h[y_test_h == 0], y_scores_h[y_test_h == 1],
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
