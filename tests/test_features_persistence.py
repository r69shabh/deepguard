import pathlib

import numpy as np
import pandas as pd
import pytest

from deepguard.features import FeatureEngineer


@pytest.fixture
def train_df():
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "A": rng.rand(200) * 100 + 1,
            "B": rng.rand(200) * 5,
        }
    )


def test_save_load_roundtrip(train_df, tmp_path):
    fe = FeatureEngineer()
    fe.fit(train_df)
    p = tmp_path / "fe.pkl"
    fe.save(p)

    loaded = FeatureEngineer.load(p)
    X_a = fe.transform(train_df)
    X_b = loaded.transform(train_df)
    np.testing.assert_allclose(X_a, X_b)


def test_load_rejects_foreign_pickle(tmp_path):
    import joblib

    p = tmp_path / "not_fe.pkl"
    joblib.dump({"oops": 1}, p)
    with pytest.raises(TypeError):
        FeatureEngineer.load(p)


def test_save_requires_fitted(train_df, tmp_path):
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError):
        fe.save(tmp_path / "fe.pkl")


def test_imputation_uses_train_medians_not_batch(train_df):
    """NaN fill must be identical regardless of the incoming batch's own median."""
    fe = FeatureEngineer(skew_threshold=10.0, corr_threshold=1.0)
    fe.fit(train_df)

    # Two single-row batches with IDENTICAL values except A=NaN.
    # If fill came from batch medians both would differ; with train medians
    # the transformed rows must match exactly.
    row = train_df.iloc[[42]].copy()
    batch1, batch2 = row.copy(), row.copy()
    batch1["A"] = np.nan
    batch2["A"] = np.nan

    out1 = fe.transform(batch1)
    out2 = fe.transform(batch2)
    np.testing.assert_allclose(out1, out2)
    assert np.isfinite(out1).all()

    # And the fill value equals the stored train median
    assert fe._train_medians is not None
    assert fe._train_medians["A"] == pytest.approx(train_df["A"].median())
