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

2. **Phase 2: Deep Sequential Learning (LSTM-AE)**
   Introduces a deep sequential representation learner—an LSTM Autoencoder. It analyzes sequences of traffic over time to detect complex, multi-stage attacks (e.g., botnets, slow infiltrations) that are invisible at the single-flow level.

3. **Phase 3: Hybrid Fusion Ensemble**
   Combines the strengths of the GMM (flow-level anomalies) and the LSTM-AE (temporal anomalies) into a robust Random Forest meta-learner. 
   
   **🏆 Benchmark Performance (CICIDS-2017):** 
   - **F1-Score:** 93.63%
   - **AUC-ROC:** 98.66%

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/r69shabh/anomaly-based-ids.git
   cd anomaly-based-ids
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install -e .
   ```

---

## 💻 Usage & CLI

The project exposes a unified Command-Line Interface (CLI) via `main.py`.

### 1. Training the Models
Train the entire pipeline (GMM, LSTM-AE, and Hybrid models) sequentially. Model artifacts will be saved to the `models/` directory.

```bash
python main.py train
```
*(Optional)* Train specific phases only (1=GMM, 2=LSTM-AE, 3=Hybrid):
```bash
python main.py train --phases 1 3
```

### 2. Evaluating the Models
Evaluate the saved models against the test dataset. This command calculates precision, recall, F1-scores, and generates ROC curves, confusion matrices, and score distribution plots in the `outputs/models/` directory.

```bash
python main.py evaluate
```

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
anomaly-based-ids/
├── anomaly_ids/           # Core Python package
│   ├── features.py        # Feature engineering pipeline
│   ├── models.py          # Model definitions (GMM, LSTM-AE, Hybrid)
│   ├── evaluate.py        # Evaluation metrics and plotting utils
│   └── utils.py           # Logging, config loading, and seeding
│
├── pipelines/             # Executable workflows
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
