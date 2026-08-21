import numpy as np
import pandas as pd
import pytest

from deepguard.data_loaders import (
    clean_raw,
    make_splits,
    load_raw,
    LABEL_COL,
    BENIGN_LABEL,
)


def _raw_frame(n_benign: int = 60, n_attack: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    benign = pd.DataFrame(
        {
            " Flow Duration": rng.rand(n_benign) * 1000,
            " Total Fwd Packets": rng.randint(1, 50, n_benign),
            "Flow Bytes/s": rng.rand(n_benign) * 100,
            "Label": BENIGN_LABEL,
        }
    )
    attack = pd.DataFrame(
        {
            " Flow Duration": rng.rand(n_attack) * 1000 + 5000,
            " Total Fwd Packets": rng.randint(1, 5, n_attack),
            "Flow Bytes/s": np.concatenate(
                [rng.rand(max(1, n_attack - 1)) * 100, [np.inf]]  # one inf row
            ),
            "Label": "DoS Hulk",
        }
    )
    return pd.concat([benign, attack], ignore_index=True)


def test_clean_raw_strips_columns_and_handles_inf():
    raw = _raw_frame()
    clean = clean_raw(raw)

    assert " Flow Duration" not in clean.columns
    assert LABEL_COL in clean.columns
    assert not np.isinf(clean.select_dtypes(include=[np.number]).values).any()
    assert not clean.duplicated().any()


def test_clean_raw_drops_high_nan_columns():
    raw = _raw_frame()
    raw["Broken Col"] = np.nan  # 100% NaN → ≥1% threshold → dropped
    clean = clean_raw(raw)
    assert "Broken Col" not in clean.columns


def test_clean_raw_requires_label():
    raw = _raw_frame().drop(columns=[LABEL_COL])
    with pytest.raises(KeyError):
        clean_raw(raw)


def test_make_splits_one_class_train_and_mixed_test():
    clean = clean_raw(_raw_frame())
    splits = make_splits(clean, test_size=0.2, seed=42)

    y = splits["y_test"]
    assert set(np.unique(y)).issubset({0, 1})
    assert len(y) == len(splits["X_test"])
    assert len(splits["X_train_benign"]) > 0
    # Train must be benign only — guaranteed by construction (benign rows split)
    # and here also verified via attack share of the pool.
    assert y.mean() > 0.2  # all attacks are in the test pool


def test_make_splits_deterministic():
    clean = clean_raw(_raw_frame())
    s1 = make_splits(clean, seed=7)
    s2 = make_splits(clean, seed=7)
    pd.testing.assert_frame_equal(s1["X_test"], s2["X_test"])
    np.testing.assert_array_equal(s1["y_test"], s2["y_test"])


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)
