import numpy as np
import pytest

from deepguard.sequences import (
    build_train_windows,
    build_eval_windows,
    scatter_window_errors_to_flows,
    window_labels_from_flows,
)


def test_build_train_windows_shapes_and_order():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    W = build_train_windows(X, window_size=5, stride=1)
    assert W.shape == (16, 5, 2)
    # windows follow row order
    np.testing.assert_array_equal(W[0, :, 0], np.arange(0, 5) * 2)
    np.testing.assert_array_equal(W[1, :, 0], np.arange(1, 6) * 2)


def test_build_train_windows_stride_and_cap():
    X = np.random.rand(50, 3).astype(np.float32)
    W = build_train_windows(X, window_size=10, stride=5)
    assert W.shape[0] == ((50 - 10) // 5) + 1
    Wc = build_train_windows(X, window_size=10, stride=1, max_windows=7, seed=1)
    assert Wc.shape[0] == 7


def test_build_train_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_train_windows(np.zeros((3, 2), dtype=np.float32), window_size=5)


def test_build_eval_windows_coverage():
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    windows, coverage = build_eval_windows(X, window_size=4, stride=2)
    assert len(windows) == len(coverage) == 4
    assert coverage[0] == [0, 1, 2, 3]
    assert coverage[-1] == [6, 7, 8, 9]
    # every flow covered at least once
    covered = {f for ids in coverage for f in ids}
    assert covered == set(range(10))


def test_scatter_errors_mean_over_covering_windows():
    # 2 windows of size 2 over 3 flows: w0=[f0,f1], w1=[f1,f2]
    err = np.array([[1.0, 2.0],
                    [10.0, 20.0]])
    coverage = [[0, 1], [1, 2]]
    s = scatter_window_errors_to_flows(err, coverage, n_flows=3)
    np.testing.assert_allclose(s, [1.0, 6.0, 20.0])


def test_window_labels_any_attack():
    y = np.array([0, 0, 1, 0])
    _, coverage = build_eval_windows(np.zeros((4, 1)), window_size=2, stride=1)
    labels = window_labels_from_flows(y, coverage)
    np.testing.assert_array_equal(labels, [0, 1, 1])
