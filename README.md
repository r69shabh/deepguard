# Adaptive Cyber-Physical Security: Anomaly-Based Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline for **Anomaly-Based Network Intrusion Detection**.

Traditional Intrusion Detection Systems (IDS) rely on signature matching, rendering them blind to zero-day vulnerabilities. This project implements a robust, multi-phase anomaly detection system that learns *exclusively* from benign network traffic. By establishing a comprehensive profile of "normal" behavior, the system can flag novel, unseen cyber-attacks simply as statistical or temporal deviations from the norm.

---

## 🧠 Architecture Overview

The system operates in a three-phase pipeline, culminating in a highly accurate hybrid detector:

1. **Phase 1: Statistical Baseline (GMM)**
   Establishes a classical Machine Learning baseline using a full-covariance Gaussian Mixture Model (GMM). It clusters normal traffic regimes to isolate malicious deviations at the individual flow level.

2. **Phase 2: Deep Sequential Learning (configurable)**
   A deep sequence model learns temporal representations over sliding windows of benign traffic. Pick the architecture in `configs/default.yaml` (`phase2.model`):
   - `lstm_ae` — LSTM Autoencoder
   - `transformer_ae` — attention encoder-decoder with positional encoding
   - `usad` — adversarial autoencoder (USAD, KDD'20)
   
   Anomaly scores are computed at **flow level**: per-timestep reconstruction errors are mapped back to individual flows, avoiding the whole-window dilution that hides isolated attacks. A flow-level **Deep SVDD** baseline is also available.

3. **Phase 3: Calibrated Fusion Ensemble (leak-free)**
   Combines GMM flow-density scores and sequence-model flow scores through per-score isotonic calibration and a meta-learner selected by cross-validation (RF / LR / weighted-average) on PR-AUC.
   
   **Evaluation protocol:** calibrators and the meta-learner are fitted exclusively on a disjoint *meta-fit* split; all reported metrics come from the held-out *eval-only* split.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/r69shabh/deepguard.git
   cd deepguard
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Get the dataset** (not included — see `data/README.md`):
   Download CICIDS-2017 from https://www.unb.ca/cic/datasets/ids-2017.html and place all 8 CSVs in `data/CICIDS2017/`.

---

## 💻 Usage & CLI

The project exposes a unified Command-Line Interface (CLI) via `main.py`.

### 0. Preprocessing (raw CSVs → model-ready arrays)
Cleans the raw CICIDS-2017 flows (metadata drops, inf→median imputation with benign-only statistics, duplicate removal), builds the one-class split (benign-only train / mixed test), fits the `FeatureEngineer`, and writes all arrays plus `feature_engineer.pkl` to `outputs/preprocessing/`.

```bash
python main.py preprocess
```
> **Note:** training and inference both require this step. The fitted
> `FeatureEngineer` is persisted so that `detect` applies *identical*
> preprocessing at serving time.

### 1. Training the Models
Train the entire pipeline (GMM, sequence model, and fusion) sequentially. Model artifacts will be saved to the `models/` directory.

```bash
python main.py train
```
*(Optional)* Train specific phases only (1=GMM, 2=sequence model, 3=fusion):
```bash
python main.py train --phases 1 3
```
*(Optional)* Use a different sequence architecture:
```bash
# edit configs/default.yaml → phase2.model: transformer_ae | usad
python main.py train --phases 2 3
```

### 2. Evaluating the Models
Evaluate the saved models against the test dataset. This command calculates precision, recall, F1-scores, and generates ROC curves, confusion matrices, and score distribution plots in the `outputs/models/` directory.

```bash
python main.py evaluate
```

> **Evaluation protocol (leak-free):** the hybrid meta-learner is fitted only on a
> disjoint meta-fit slice of the labeled pool (`hybrid.meta_train_frac`, default 0.4);
> all reported metrics come from the remaining held-out rows (`models/model_c_eval_mask.npy`).
> Earlier versions of this project trained the meta-learner on the test set itself,
> which inflated reported scores — treat pre-2026 numbers as invalid.

### 3. Inference / Detection
Run the anomaly detector on brand new, raw network flow data (CSV format). The pipeline will handle feature engineering automatically.

```bash
python main.py detect --input data/new_flows.csv --output predictions.csv
```

### 4. Ablation Study
Run the standalone script that evaluates the system's performance under various architectural ablations.
```bash
python main.py ablation
```

---

## ⚙️ Configuration

All hyperparameters, file paths, and model settings are centralized in `configs/default.yaml`. You do not need to modify the source code to tune the models.

To use a custom configuration file:
```bash
python main.py train --config configs/custom_experiment.yaml
```

---

## 📁 Project Structure

```text
deepguard/
├── deepguard/           # Core Python package
│   ├── features.py        # Feature engineering pipeline (fit/transform + persistence)
│   ├── data_loaders.py    # Raw CICIDS-2017 ingestion and cleaning
│   ├── models.py          # Model definitions (GMM, LSTM-AE, Hybrid)
│   ├── evaluate.py        # Evaluation metrics and plotting utils
│   └── utils.py           # Logging, config loading, seeding, split helpers
│
├── pipelines/             # Executable workflows
│   ├── preprocess.py      # Raw CSVs → model-ready arrays
│   ├── train.py           # End-to-end training pipeline
│   ├── evaluate.py        # Evaluation pipeline
│   └── detect.py          # Inference/detection pipeline
│
├── configs/               # YAML configuration files
│   └── default.yaml       
│
├── tests/                 # Unit tests (pytest)
│
├── data/                  # Raw dataset directory
├── models/                # Saved model artifacts (.pkl, .keras)
├── outputs/               # Generated plots, arrays, and intermediate data
├── results/               # Metrics and CSV reports
│
├── main.py                # Unified CLI entrypoint
└── setup.py               # Package installation script
```

---

## 🧪 Testing

The project uses `pytest` for unit testing. To run the test suite and ensure all core components (features, models, evaluation) are functioning correctly:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
