# anomaly_ids — Anomaly-Based Intrusion Detection System
"""
Top-level package for the anomaly-based-ids project.

Exposes the core modules:
  - features  : FeatureEngineer preprocessing pipeline
  - models    : GMMDetector, LSTMAEDetector, HybridDetector, and friends
  - evaluate  : metrics, plots, and report utilities
  - utils     : logging, seeding, config loading
"""

__version__ = "1.0.0"
__author__  = "Rishabh Gusain"

from anomaly_ids.features import FeatureEngineer
from anomaly_ids.models   import (
    GMMDetector,
    IsolationForestDetector,
    OCSVMDetector,
    LSTMAEDetector,
    HybridDetector,
)

__all__ = [
    "FeatureEngineer",
    "GMMDetector",
    "IsolationForestDetector",
    "OCSVMDetector",
    "LSTMAEDetector",
    "HybridDetector",
]
