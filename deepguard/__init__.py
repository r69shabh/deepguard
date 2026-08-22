# deepguard — Anomaly-Based Intrusion Detection System
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

from deepguard.features import FeatureEngineer
from deepguard.models import (
    GMMDetector,
    HybridDetector,
    IsolationForestDetector,
    LSTMAEDetector,
    OCSVMDetector,
)

__all__ = [
    "FeatureEngineer",
    "GMMDetector",
    "IsolationForestDetector",
    "OCSVMDetector",
    "LSTMAEDetector",
    "HybridDetector",
]
