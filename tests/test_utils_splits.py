import numpy as np
import pytest

from deepguard.utils import labeled_holdout_split


def test_disjoint_and_complete():
    y = np.array([0] * 80 + [1] * 20)
    meta_idx, eval_idx = labeled_holdout_split(y, meta_frac=0.4, seed=42)

    assert len(np.intersect1d(meta_idx, eval_idx)) == 0
    assert np.array_equal(np.sort(np.concatenate([meta_idx, eval_idx])), np.arange(len(y)))


def test_stratified():
    y = np.array([0] * 100 + [1] * 40)
    meta_idx, eval_idx = labeled_holdout_split(y, meta_frac=0.5, seed=0)

    assert (y[meta_idx] == 1).sum() == 20
    assert (y[eval_idx] == 1).sum() == 20
    assert (y[meta_idx] == 0).sum() == 50
    assert (y[eval_idx] == 0).sum() == 50


def test_deterministic():
    y = np.random.randint(0, 2, 500)
    a = labeled_holdout_split(y, meta_frac=0.3, seed=7)
    b = labeled_holdout_split(y, meta_frac=0.3, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_empty_eval_raises():
    with pytest.raises(ValueError):
        labeled_holdout_split(np.array([1, 1]), meta_frac=1.0)
