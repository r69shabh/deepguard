"""
deepguard/data_loaders.py
=========================
Raw-data ingestion for CICIDS-2017.

Replicates the cleaning pipeline documented in notebooks/stage2_preprocessing.py
as importable, testable functions so the full pipeline is reproducible from the
raw CSVs without running any notebook:

  Step 1  Drop metadata/identifier columns and near-constant columns.
  Step 2  Replace ±inf → NaN; impute columns whose NaN fraction < 1% with the
          benign median; drop columns whose NaN fraction ≥ 1%.
  Step 3  Drop exact duplicate rows.
  Split   Benign rows: 80% train / 20% holdout (shuffled, seeded).
          Test pool   : benign holdout + ALL attack rows, shuffled.

All imputation statistics come from BENIGN rows only — no attack information
influences preprocessing.

Usage
-----
>>> from deepguard.data_loaders import load_raw, clean_raw, make_splits
>>> raw = load_raw("data/CICIDS2017")
>>> clean = clean_raw(raw)
>>> splits = make_splits(clean)
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

CICIDS_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"

ID_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Timestamp", "source_file",
]

NAN_DROP_THRESHOLD = 0.01     # drop columns with ≥1% NaN after inf removal
CONST_STD_THRESHOLD = 1e-6    # drop numeric columns with std below this


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw(
    data_dir: Union[str, pathlib.Path],
    files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate the raw CICIDS-2017 CSV files.

    Parameters
    ----------
    data_dir : str or Path — directory containing the dataset CSVs.
    files : list[str], optional — override the default file list.

    Returns
    -------
    pd.DataFrame with all rows, original column names, plus 'source_file'.
    """
    data_dir = pathlib.Path(data_dir)
    files = files or CICIDS_FILES

    frames = []
    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {fpath}\n"
                "Download CICIDS-2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
                "and place the CSVs in this directory."
            )
        df = pd.read_csv(fpath, low_memory=False, encoding="latin-1")
        # Thursday-Morning file has ~288k all-NaN rows appended at EOF (known artefact)
        df = df.dropna(how="all")
        df["source_file"] = fname
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the EDA-justified cleaning pipeline (Steps 1–3).

    Parameters
    ----------
    df : pd.DataFrame — raw frame as produced by load_raw().

    Returns
    -------
    pd.DataFrame — cleaned feature frame including LABEL_COL.
    """
    out = df.copy()
    out.columns = out.columns.str.strip()

    if LABEL_COL not in out.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not found after stripping whitespace.")

    # ── Step 1: metadata / identifier columns + near-constant columns ─────────
    drop_ids = [c for c in ID_COLS if c in out.columns]
    out = out.drop(columns=drop_ids)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    const_cols = [
        c for c in numeric_cols
        if c != LABEL_COL
        and out[c].replace([np.inf, -np.inf], np.nan).dropna().std() < CONST_STD_THRESHOLD
    ]
    out = out.drop(columns=const_cols)

    # ── Step 2: inf → NaN, benign-median imputation, high-NaN column drops ────
    out = out.replace([np.inf, -np.inf], np.nan)
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    benign_medians = out.loc[out[LABEL_COL] == BENIGN_LABEL, numeric_cols].median()

    to_drop = []
    for c in numeric_cols:
        nan_frac = out[c].isna().mean()
        if nan_frac == 0:
            continue
        if nan_frac < NAN_DROP_THRESHOLD:
            out[c] = out[c].fillna(benign_medians[c])
        else:
            to_drop.append(c)
    if to_drop:
        out = out.drop(columns=to_drop)

    # ── Step 3: exact duplicate rows ───────────────────────────────────────────
    out = out.drop_duplicates().reset_index(drop=True)
    return out


# ─── Splits ───────────────────────────────────────────────────────────────────

def make_splits(
    df_clean: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Build the one-class train/test split from a cleaned frame.

    Train = benign only (one-class paradigm). Test = benign holdout plus ALL
    attack flows, shuffled with the same seed.

    Parameters
    ----------
    df_clean : pd.DataFrame — output of clean_raw() (includes LABEL_COL).
    test_size : float — benign holdout fraction (default 0.2).
    seed : int — RNG seed for the shuffle/split.

    Returns
    -------
    dict with keys:
        X_train_benign : pd.DataFrame — benign-only feature rows.
        X_test         : pd.DataFrame — pooled test features.
        y_test         : np.ndarray int — 1 = attack, 0 = benign.
        attack_types   : np.ndarray str — original multiclass labels.
    """
    if LABEL_COL not in df_clean.columns:
        raise KeyError(f"Expected '{LABEL_COL}' in cleaned frame.")

    feat_cols = [c for c in df_clean.columns if c != LABEL_COL]

    benign = df_clean.loc[df_clean[LABEL_COL] == BENIGN_LABEL, feat_cols + [LABEL_COL]]
    attacks = df_clean.loc[df_clean[LABEL_COL] != BENIGN_LABEL, feat_cols + [LABEL_COL]]

    # Seeded shuffle of each part before splitting
    benign_shuffled = benign.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test_benign = max(1, int(round(len(benign_shuffled) * test_size)))

    test_pool = pd.concat(
        [
            benign_shuffled.iloc[:n_test_benign],
            attacks,
        ],
        ignore_index=True,
    )
    train_benign = benign_shuffled.iloc[n_test_benign:]

    # Shuffle the pooled test set (benign holdout + all attacks)
    test_pool = test_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    y_test = (test_pool[LABEL_COL].values != BENIGN_LABEL).astype(int)
    attack_types = test_pool[LABEL_COL].to_numpy()

    return {
        "X_train_benign": train_benign[feat_cols].reset_index(drop=True),
        "X_test": test_pool[feat_cols],
        "y_test": y_test,
        "attack_types": attack_types,
    }
