"""Classification smoke tests on synthetic data (no real dataset needed)."""
from __future__ import annotations

import joblib
import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src import classification as cls


def test_train_classical_returns_six_models(supervised_frame):
    outcomes, table, splits, meta = cls.train_classical(supervised_frame, n_splits=3)
    assert len(outcomes) == 6
    assert len(table) == 6
    assert set(meta).issuperset(
        {"feature_columns", "seed", "test_size", "n_splits", "best_model_name"})


def test_metrics_in_unit_range(supervised_frame):
    _, table, _, _ = cls.train_classical(supervised_frame, n_splits=3)
    for col in ("AUC", "F1", "Precision", "Recall", "Accuracy"):
        assert table[col].between(0.0, 1.0).all()


def test_pipeline_has_preprocessor_before_smote(supervised_frame):
    outcomes, _, _, _ = cls.train_classical(supervised_frame, n_splits=3)
    pipe = outcomes[0].estimator
    assert isinstance(pipe, ImbPipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names.index("pre") < step_names.index("smote") < step_names.index("clf")


def test_saved_model_roundtrips(tmp_path, supervised_frame):
    outcomes, _, splits, _ = cls.train_classical(supervised_frame, n_splits=3)
    cls.save_models(outcomes, folder=tmp_path)
    loaded = joblib.load(tmp_path / f"{outcomes[0].name}.joblib")
    preds = loaded.predict(splits["X_test_raw"])
    assert len(preds) == len(splits["X_test_raw"])


def test_splits_contain_raw_and_transformed(supervised_frame):
    _, _, splits, _ = cls.train_classical(supervised_frame, n_splits=3)
    assert {"X_train_raw", "X_test_raw", "X_train_t", "X_test_t",
            "preprocessor", "transformed_feature_names"} <= set(splits)
    assert np.isfinite(splits["X_train_t"]).all()


def test_mlp_proba_fn_returns_probabilities():
    """SHAP-ready MLP wrapper returns calibrated-shape probabilities in [0, 1]."""
    pytest.importorskip("torch")
    from sklearn.preprocessing import StandardScaler

    X = np.random.RandomState(0).randn(20, 6).astype("float32")
    model = cls.build_mlp(6)
    scaler = StandardScaler().fit(X)
    proba = cls.mlp_proba_fn(model, scaler)(X)
    assert proba.shape == (20,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
