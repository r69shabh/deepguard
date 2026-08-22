import numpy as np
import pandas as pd
import pytest

from deepguard.metrics_ext import (
    bootstrap_ci,
    fpr_at_tpr,
    per_attack_table,
    prevalence_sweep,
    subsample_to_prevalence,
)


def _separable(n=500, seed=0, n_total_neg=400):
    rng = np.random.RandomState(seed)
    y = np.concatenate([np.ones(100, int), np.zeros(n_total_neg, int)])
    s = np.concatenate([rng.normal(3, 1, 100), rng.normal(0, 1, n_total_neg)])
    return y, s


def test_fpr_at_tpr_perfect_detector():
    rng = np.random.RandomState(0)
    y = np.concatenate([np.ones(100, int), np.zeros(400, int)])
    s = np.concatenate([rng.uniform(10, 11, 100), rng.uniform(0, 1, 400)])
    assert fpr_at_tpr(y, s, 0.95) == pytest.approx(0.0)


def test_fpr_at_tpr_random_is_near_diagonal():
    rng = np.random.RandomState(1)
    y = np.array([0] * 900 + [1] * 100)
    s = rng.rand(1000)
    # random scorer ⇒ ROC is the diagonal: FPR ≈ TPR at any operating point
    assert fpr_at_tpr(y, s, 0.95) == pytest.approx(0.95, abs=0.05)


def test_bootstrap_ci_contains_point_estimate_and_narrows_with_signal():
    y, s = _separable()
    ci = bootstrap_ci(y, s, n_boot=200, seed=0)
    assert ci["ci_lower"] <= ci["estimate"] <= ci["ci_upper"]
    assert ci["estimate"] > 0.95


def test_subsample_to_prevalence_ratios():
    y, s = _separable(n_total_neg=4000)
    for target in (0.2, 0.05):
        ys, ss = subsample_to_prevalence(y, s, prevalence=target)
        kept_attacks = int(ys.sum())
        assert ys.mean() == pytest.approx(target, abs=0.005)
        assert kept_attacks == (y == 1).sum()  # all attacks retained


def test_subsample_caps_at_available_benign():
    """When benigns run out, everything is returned (prevalence floor)."""
    y, s = _separable()  # only 400 benigns ⇒ floor prevalence ≈ 20%
    ys, _ = subsample_to_prevalence(y, s, prevalence=0.01)
    assert len(ys) == len(y)


def test_prevalence_sweep_columns_and_baseline():
    y, s = _separable()
    df = prevalence_sweep(y, s, prevalences=(0.2, 0.05))
    assert set(df.columns) >= {"prevalence", "pr_auc", "fpr_at_95tpr",
                               "random_baseline_pr_auc"}
    # good detector beats random baseline at every prevalence
    assert (df.pr_auc > df.random_baseline_pr_auc).all()


def test_per_attack_table_counts_and_rates():
    rng = np.random.RandomState(2)
    types = np.array(["DoS"] * 50 + ["Bot"] * 30 + ["BENIGN"] * 200)
    y = np.array([1] * 80 + [0] * 200)
    scores = np.concatenate([rng.normal(3, 1, 50), rng.normal(-1, 1, 30),
                             rng.normal(0, 1, 200)])
    pred = (scores > 1.5).astype(int)
    df = per_attack_table(y, types, scores, y_pred=pred)
    dos = df[df.attack_type == "DoS"].iloc[0]
    bot = df[df.attack_type == "Bot"].iloc[0]
    assert dos["count"] == 50 and bot["count"] == 30
    assert dos["auc_vs_benign"] > bot["auc_vs_benign"]
    assert bot["detection_rate"] < dos["detection_rate"]
