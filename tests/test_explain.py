import numpy as np
import pytest

from deepguard.explain import (
    benign_reference_point,
    counterfactual_attribution,
    explain_flow,
    permutation_importance_scores,
)


def test_counterfactual_blames_the_deviant_feature():
    rng = np.random.RandomState(0)
    X_benign = rng.normal(0, 1, (500, 3))
    ref = benign_reference_point(X_benign)

    # Scorer: purely driven by deviation of feature 1
    scorer = lambda Z: np.abs(Z[:, 1] - ref[1]).reshape(-1)  # noqa: E731

    x = ref.copy()
    x[1] += 5.0
    attribs = counterfactual_attribution(x, scorer, ref, ["a", "b", "c"])
    assert attribs[0][0] == "b"
    assert attribs[0][1] == pytest.approx(5.0, abs=1e-6)
    # untouched features contribute ~nothing
    for name, contrib in attribs[1:]:
        assert abs(contrib) < 1e-6


def test_explain_flow_output_contract():
    rng = np.random.RandomState(1)
    ref = rng.rand(4)
    scorer = lambda Z: ((Z - ref) ** 2).sum(axis=1)  # noqa: E731
    out = explain_flow(ref + 2, scorer, ref, list("wxyz"), top_k=3)
    assert len(out["top_features"]) == 3
    assert len(out["contributions"]) == 3


def test_permutation_importance_ranks_informative_feature_first():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(3)
    n = 400
    y = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    X = rng.normal(0, 1, (n, 2))
    X[: n // 2, 0] += 3.0          # feature_0 informative; feature_1 noise
    est = LogisticRegression().fit(X, y)
    drops = permutation_importance_scores(est, X, y, ["feature_0", "feature_1"],
                                          n_repeats=5, seed=0)
    assert drops[0][0] == "feature_0"
    assert drops[0][1] > drops[1][1]
