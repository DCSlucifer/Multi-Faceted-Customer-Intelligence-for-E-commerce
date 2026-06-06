"""Artifact validation tests.

These assert on the *real* pipeline outputs, so they ``skip`` (rather than fail)
on a fresh clone that hasn't run the pipeline yet. Run the pipeline first to
exercise them:  ``python scripts/run_pipeline.py --stage all``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import artifacts as A


def _require(key: str):
    path = A.ARTIFACTS[key]
    if not path.exists():
        pytest.skip(f"{key} not generated yet ({path.name}); run the pipeline first")
    return path


def test_core_artifacts_exist_when_pipeline_ran():
    # If preprocessing ran, all of these should be present together.
    if not A.ARTIFACTS["classification_features"].exists():
        pytest.skip("pipeline not run yet")
    missing = A.missing_artifacts(
        ["rfm_features", "churn_labels", "classification_features", "classical_results"])
    assert not missing, f"missing artifacts: {missing}"


@pytest.mark.parametrize("key", [
    "rfm_features", "churn_labels", "classification_features",
    "classical_results", "baseline_results", "ablation_results",
    "calibration_results", "association_rules",
])
def test_csv_schema(key):
    _require(key)
    missing = A.check_csv_schema(key)
    assert not missing, f"{key} missing columns {missing}"


def test_feature_label_consistency():
    _require("classification_features")
    _require("churn_labels")
    feats = pd.read_csv(A.ARTIFACTS["classification_features"])
    churn = pd.read_csv(A.ARTIFACTS["churn_labels"])
    assert set(feats["CustomerID"]) == set(churn["CustomerID"])


def test_no_missing_values_in_critical_artifacts():
    for key in ("classification_features", "churn_labels", "classical_results"):
        _require(key)
        df = pd.read_csv(A.ARTIFACTS[key])
        assert not df.isnull().any().any(), f"{key} has NaNs"


def test_best_auc_is_realistic():
    _require("classical_results")
    res = pd.read_csv(A.ARTIFACTS["classical_results"])
    best = float(res["AUC"].max())
    assert 0.6 <= best <= 0.95, f"AUC={best} suggests leakage or a bug"


def test_deep_learning_artifacts_exist_when_pipeline_ran():
    if not A.ARTIFACTS["mlp_vs_classical"].exists():
        pytest.skip("pipeline not run yet")
    for key in ("mlp_model", "autoencoder_model", "dl_scalers", "mlp_metadata"):
        assert A.ARTIFACTS[key].exists(), f"{key} missing"
        assert A.ARTIFACTS[key].stat().st_size > 0, f"{key} empty"
