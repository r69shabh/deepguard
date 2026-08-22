#!/usr/bin/env python3
"""
main.py
=======
Command-line entrypoint for the anomaly-based-ids project.

Provides a unified interface to run training, evaluation, and detection pipelines.

Usage
-----
    python main.py train --phases 1 2 3
    python main.py evaluate
    python main.py detect --input data/new_flows.csv --output preds.csv
"""

import argparse
import sys

from pipelines import detect, evaluate, monitor, preprocess, robustness, train


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly-Based Intrusion Detection System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # ── Preprocess Command ────────────────────────────────────────────────────
    prep_parser = subparsers.add_parser(
        "preprocess", help="Raw CICIDS-2017 CSVs → model-ready arrays"
    )
    prep_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )
    prep_parser.add_argument(
        "--data-dir", default=None, help="Directory containing raw dataset CSVs"
    )

    # ── Train Command ────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train the anomaly detection models")
    train_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )
    train_parser.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2, 3],
        help="Phases to train (1=GMM, 2=LSTM-AE, 3=Hybrid)"
    )

    # ── Evaluate Command ─────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate saved models on test data")
    eval_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )

    # ── Detect Command ───────────────────────────────────────────────────────
    detect_parser = subparsers.add_parser("detect", help="Run inference on new flow data")
    detect_parser.add_argument(
        "--input", required=True, help="Input CSV file containing network flows"
    )
    detect_parser.add_argument(
        "--output", required=True, help="Output CSV file to save predictions"
    )
    detect_parser.add_argument(
        "--explain", action="store_true",
        help="Add top contributing-feature columns for flagged flows",
    )
    detect_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )

    # ── Ablation Command (Optional hook to existing script) ──────────────────
    subparsers.add_parser("ablation", help="Run the full ablation study")

    # ── Robustness Command ───────────────────────────────────────────────────
    rob_parser = subparsers.add_parser(
        "robustness", help="Mimicry-evasion and noise-FPR robustness experiments"
    )
    rob_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )
    rob_parser.add_argument(
        "--models", nargs="+", default=["gmm", "hybrid"],
        choices=["gmm", "hybrid"], help="Which scorers to test",
    )

    # ── Monitor Command ──────────────────────────────────────────────────────
    mon_parser = subparsers.add_parser(
        "monitor", help="Streaming drift monitoring + adaptive retraining"
    )
    mon_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to configuration file"
    )
    mon_parser.add_argument(
        "--inject-drift-after", type=int, default=None,
        help="Step after which a synthetic covariate shift is applied",
    )
    mon_parser.add_argument(
        "--drift-shift", type=float, default=3.0,
        help="Shift magnitude in feature-std units",
    )
    mon_parser.add_argument(
        "--max-steps", type=int, default=None, help="Cap the stream length",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess.run(data_dir=args.data_dir, config_path=args.config)
    elif args.command == "train":
        train.run(config_path=args.config, phases=args.phases)
    elif args.command == "evaluate":
        evaluate.run(config_path=args.config)
    elif args.command == "detect":
        detect.run(input_csv=args.input, output_csv=args.output,
                   config_path=args.config, explain=args.explain)
    elif args.command == "robustness":
        robustness.run(config_path=args.config, models=args.models)
    elif args.command == "monitor":
        monitor.run(config_path=args.config,
                    inject_drift_after=args.inject_drift_after,
                    drift_shift=args.drift_shift,
                    max_steps=args.max_steps)
    elif args.command == "ablation":
        # Simply execute the run_ablation script
        import subprocess
        subprocess.run([sys.executable, "scripts/run_ablation.py"])


if __name__ == "__main__":
    main()
