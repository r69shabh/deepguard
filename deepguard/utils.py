"""
deepguard/utils.py
====================
Shared utilities: logging setup, random seeding, config loading, path helpers.
"""

from __future__ import annotations

import logging
import pathlib
import random
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, pathlib.Path]] = None,
) -> logging.Logger:
    """
    Configure and return the root logger for the project.

    Outputs structured log lines to stdout. Optionally also writes to a file.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    log_file : str or Path, optional
        If provided, also write logs to this file path.

    Returns
    -------
    logging.Logger
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logger = logging.getLogger("deepguard")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'deepguard' namespace."""
    return logging.getLogger(f"deepguard.{name}")


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds globally for reproducibility.

    Covers Python's random, NumPy, and TensorFlow (if installed).

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Union[str, pathlib.Path] = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_nested(cfg: Dict, *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a deeply nested value from a config dict.

    Parameters
    ----------
    cfg : dict
    *keys : str
        Sequence of keys to traverse.
    default
        Value to return if any key is missing.

    Returns
    -------
    Any
    """
    for key in keys:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(key, default)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: Union[str, pathlib.Path]) -> None:
    """Create all given directories (and their parents) if they don't exist."""
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def project_root() -> pathlib.Path:
    """Return the project root directory (two levels above this file)."""
    return pathlib.Path(__file__).parent.parent
