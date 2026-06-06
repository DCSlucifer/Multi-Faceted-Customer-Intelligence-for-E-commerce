"""Tests for the single-source-of-truth submission metrics snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.submission_snapshot import load_submission_snapshot


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_artifacts(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    processed = tmp_path / "data" / "processed"
    reports.mkdir(parents=True)
    processed.mkdir(parents=True)

    (reports / "manifest.json").write_text(json.dumps({
        "seed": 42,
        "churn_cutoff": "2011-09-01",
        "churn_window_days": 90,
        "n_clean_rows": 805549,
        "n_customers_total": 5878,
        "n_customers_labelled": 5249,
        "churn_rate": 0.5731,
        "classification_best_model": "LogisticRegression",
        "classification_best_auc": 0.785,
        "classification_mlp_auc": 0.7832,
        "clustering_silhouette": 0.4193,
        "association_n_rules": 17,
        "churn_gap_between_segments": 0.673,
        "decision_table_customers": 5249,
        "decision_retention_targets": 1893,
        "decision_cross_sell_targets": 620,
        "decision_manual_review_targets": 240,
    }), encoding="utf-8")
    write_csv(reports / "classical_results.csv", [
        {"Model": "LogisticRegression", "CV_AUC_mean": 0.7987, "CV_AUC_std": 0.0161, "AUC": 0.785, "F1": 0.7392, "Precision": 0.7698, "Recall": 0.7110, "Accuracy": 0.7124, "best_params": "{}"},
        {"Model": "RandomForest", "CV_AUC_mean": 0.7999, "CV_AUC_std": 0.0177, "AUC": 0.7816, "F1": 0.7454, "Precision": 0.7467, "Recall": 0.7442, "Accuracy": 0.7086, "best_params": "{}"},
    ])
    write_csv(reports / "mlp_vs_classical.csv", [
        {"Model": "LogisticRegression", "AUC": 0.785, "F1": 0.7392, "Precision": 0.7698, "Recall": 0.7110, "Accuracy": 0.7124},
        {"Model": "MLP", "AUC": 0.7832, "F1": 0.7381, "Precision": 0.7561, "Recall": 0.7209, "Accuracy": 0.7067},
    ])
    write_csv(reports / "baseline_results.csv", [
        {"Baseline": "MajorityClass", "n_features": 1, "AUC": 0.5, "F1": 0.7288, "Precision": 0.5733, "Recall": 1.0, "Accuracy": 0.5733},
        {"Baseline": "RecencyOnly_LR", "n_features": 1, "AUC": 0.7513, "F1": 0.6883, "Precision": 0.7766, "Recall": 0.6179, "Accuracy": 0.679},
        {"Baseline": "LR_without_Recency", "n_features": 6, "AUC": 0.7546, "F1": 0.7303, "Precision": 0.7365, "Recall": 0.7243, "Accuracy": 0.6933},
    ])
    write_csv(reports / "calibration_results.csv", [
        {"model": "LogisticRegression", "bin": 1, "mean_predicted_prob": 0.1, "fraction_positive": 0.1048, "brier_score": 0.1892},
    ])
    write_csv(reports / "cluster_stability.csv", [
        {"seed": "0", "silhouette": 0.4193, "size_cv": 0.235, "cluster_sizes": "1882,1643,1405,948"},
        {"seed": "SUMMARY", "silhouette": 0.4193, "size_cv": 0.235, "cluster_sizes": "silhouette_std=0.0000"},
    ])
    write_csv(reports / "association_rules.csv", [
        {"antecedents": "frozenset({'22697'})", "consequents": "frozenset({'22699'})", "support": 0.0209, "confidence": 0.797, "lift": 26.758},
    ])
    write_csv(processed / "classification_features.csv", [
        {"CustomerID": 1, "Recency": 0, "Frequency": 1, "Monetary": 10, "avg_basket_value": 10, "avg_basket_size": 1, "unique_products": 1, "DominantCountry": "United Kingdom"},
    ])


def test_snapshot_loads_current_classification_metrics(tmp_path):
    _build_artifacts(tmp_path)
    snapshot = load_submission_snapshot(root=tmp_path)

    assert snapshot.best_model == "LogisticRegression"
    assert snapshot.best_auc == 0.785
    assert snapshot.best_auc_3dp == "0.785"
    assert snapshot.mlp_auc_3dp == "0.783"
    assert snapshot.n_clean_rows_text == "805,549"
    assert snapshot.churn_cutoff == "2011-09-01"
    assert snapshot.churn_rate_text == "57.3%"
    assert snapshot.silhouette_std_3dp == "0.000"
    assert snapshot.without_recency_auc_2dp == "0.75"
    assert snapshot.leaky_metric_values == {"0.998", "0.990", "0.9981", "0.9900"}


def test_snapshot_tokens_have_no_unfilled_values(tmp_path):
    _build_artifacts(tmp_path)
    tokens = load_submission_snapshot(root=tmp_path).tokens()
    assert tokens["classification_best_model"] == "LogisticRegression"
    assert tokens["classification_best_auc"] == "0.785"
    assert tokens["churn_gap_points"] == "67.3"
    assert all(v != "" and v is not None for v in tokens.values())


def test_snapshot_exposes_decision_tokens(tmp_path):
    _build_artifacts(tmp_path)
    snapshot = load_submission_snapshot(root=tmp_path)
    tokens = snapshot.tokens()
    assert snapshot.decision_table_customers == 5249
    assert tokens["decision_table_customers"] == "5,249"
    assert tokens["decision_retention_targets"] == "1,893"
    assert tokens["decision_cross_sell_targets"] == "620"
    assert tokens["decision_manual_review_targets"] == "240"
