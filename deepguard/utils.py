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


# ──────────────────────────────────────────────────────────────────────────────
# Validation splits
# ──────────────────────────────────────────────────────────────────────────────

def labeled_holdout_split(
    y: np.ndarray,
    meta_frac: float = 0.4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified index split of a labeled pool into disjoint meta-fit / eval parts.

    Used by the hybrid fusion stage: any supervised component (the meta-learner)
    must be fitted exclusively on the meta-fit part; final metrics are computed
    only on the eval part. This removes the train-on-test leakage that inflated
    earlier reported scores.

    Parameters
    ----------
    y : np.ndarray of int — binary labels of the labeled pool.
    meta_frac : float — fraction of the pool assigned to meta-fit.
    seed : int — RNG seed.

    Returns
    -------
    (meta_idx, eval_idx) : np.ndarray of int — disjoint index arrays whose
    union covers range(len(y)).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    meta_idx, eval_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_meta = max(1, int(round(len(cls_idx) * meta_frac))) if len(cls_idx) > 1 else len(cls_idx)
        meta_idx.append(cls_idx[:n_meta])
        eval_idx.append(cls_idx[n_meta:])
    meta = np.sort(np.concatenate(meta_idx))
    ev = np.sort(np.concatenate(eval_idx))
    if len(ev) == 0:
        raise ValueError(
            "Eval split is empty: reduce hybrid.meta_train_frac or provide more data."
        )
    return meta, ev
