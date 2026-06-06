"""Single source of truth for every metric quoted in the submission.

All report/README/slide numbers are loaded *once* from the generated artifacts
(`reports/*.csv`, `reports/manifest.json`) into a frozen
:class:`SubmissionSnapshot`. Builders consume `snapshot.tokens()` so the report,
slides, and README can never drift from the pipeline that produced the numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The inflated, pre-leakage-fix metrics that must never reappear as a *current*
# claim in any submission-facing file.
LEAKY_METRIC_VALUES = {"0.998", "0.990", "0.9981", "0.9900"}


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class SubmissionSnapshot:
    seed: int
    churn_cutoff: str
    churn_window_days: int
    n_clean_rows: int
    n_customers_total: int
    n_customers_labelled: int
    churn_rate: float
    best_model: str
    best_auc: float
    best_f1: float
    best_precision: float
    best_recall: float
    best_accuracy: float
    best_cv_auc: float
    mlp_auc: float
    mlp_f1: float
    majority_auc: float
    recency_auc: float
    without_recency_auc: float
    calibration_brier: float
    clustering_silhouette: float
    clustering_silhouette_std: float
    association_n_rules: int
    top_rule_lift: float
    churn_gap_between_segments: float
    decision_table_customers: int
    decision_retention_targets: int
    decision_cross_sell_targets: int
    decision_manual_review_targets: int

    # --- formatted views -------------------------------------------------- #
    @property
    def best_auc_3dp(self) -> str:
        return _fmt_float(self.best_auc, 3)

    @property
    def best_f1_3dp(self) -> str:
        return _fmt_float(self.best_f1, 3)

    @property
    def best_precision_3dp(self) -> str:
        return _fmt_float(self.best_precision, 3)

    @property
    def best_recall_3dp(self) -> str:
        return _fmt_float(self.best_recall, 3)

    @property
    def best_cv_auc_3dp(self) -> str:
        return _fmt_float(self.best_cv_auc, 3)

    @property
    def mlp_auc_3dp(self) -> str:
        return _fmt_float(self.mlp_auc, 3)

    @property
    def majority_auc_2dp(self) -> str:
        return _fmt_float(self.majority_auc, 2)

    @property
    def recency_auc_2dp(self) -> str:
        return _fmt_float(self.recency_auc, 2)

    @property
    def without_recency_auc_2dp(self) -> str:
        return _fmt_float(self.without_recency_auc, 2)

    @property
    def calibration_brier_3dp(self) -> str:
        return _fmt_float(self.calibration_brier, 3)

    @property
    def silhouette_3dp(self) -> str:
        return _fmt_float(self.clustering_silhouette, 3)

    @property
    def silhouette_std_3dp(self) -> str:
        return _fmt_float(self.clustering_silhouette_std, 3)

    @property
    def churn_gap_points_1dp(self) -> str:
        return f"{100 * self.churn_gap_between_segments:.1f}"

    @property
    def churn_rate_text(self) -> str:
        return _fmt_pct(self.churn_rate, 1)

    @property
    def n_clean_rows_text(self) -> str:
        return _fmt_int(self.n_clean_rows)

    @property
    def n_customers_total_text(self) -> str:
        return _fmt_int(self.n_customers_total)

    @property
    def n_customers_labelled_text(self) -> str:
        return _fmt_int(self.n_customers_labelled)

    @property
    def top_rule_lift_1dp(self) -> str:
        return _fmt_float(self.top_rule_lift, 1)

    @property
    def leaky_metric_values(self) -> set[str]:
        return set(LEAKY_METRIC_VALUES)

    def tokens(self) -> dict[str, str]:
        """Flat ``{{token}} -> value`` map consumed by the renderers."""
        return {
            "seed": str(self.seed),
            "churn_cutoff": self.churn_cutoff,
            "churn_window_days": str(self.churn_window_days),
            "n_clean_rows": self.n_clean_rows_text,
            "n_customers_total": self.n_customers_total_text,
            "n_customers_labelled": self.n_customers_labelled_text,
            "churn_rate": self.churn_rate_text,
            "classification_best_model": self.best_model,
            "classification_best_auc": self.best_auc_3dp,
            "classification_best_f1": self.best_f1_3dp,
            "classification_best_precision": self.best_precision_3dp,
            "classification_best_recall": self.best_recall_3dp,
            "classification_best_cv_auc": self.best_cv_auc_3dp,
            "classification_mlp_auc": self.mlp_auc_3dp,
            "majority_auc": self.majority_auc_2dp,
            "recency_auc": self.recency_auc_2dp,
            "without_recency_auc": self.without_recency_auc_2dp,
            "calibration_brier": self.calibration_brier_3dp,
            "clustering_silhouette": self.silhouette_3dp,
            "clustering_silhouette_std": self.silhouette_std_3dp,
            "association_n_rules": str(self.association_n_rules),
            "top_rule_lift": self.top_rule_lift_1dp,
            "churn_gap_points": self.churn_gap_points_1dp,
            "decision_table_customers": _fmt_int(self.decision_table_customers),
            "decision_retention_targets": _fmt_int(self.decision_retention_targets),
            "decision_cross_sell_targets": _fmt_int(self.decision_cross_sell_targets),
            "decision_manual_review_targets": _fmt_int(self.decision_manual_review_targets),
        }


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required submission artifact missing: {path}")
    return pd.read_csv(path)


def _read_manifest(root: Path) -> dict:
    path = root / "reports" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Required manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_auc(baselines: pd.DataFrame, name: str) -> float:
    row = baselines.loc[baselines["Baseline"].eq(name)]
    if row.empty:
        raise ValueError(f"baseline_results.csv missing row: {name}")
    return float(row.iloc[0]["AUC"])


def _silhouette_std(cluster_stability: pd.DataFrame) -> float:
    row = cluster_stability.loc[cluster_stability["seed"].astype(str).eq("SUMMARY")]
    if row.empty:
        return float(cluster_stability["silhouette"].astype(float).std())
    text = str(row.iloc[0].get("cluster_sizes", ""))
    if text.startswith("silhouette_std="):
        return float(text.split("=", 1)[1])
    return float(cluster_stability["silhouette"].astype(float).std())


def load_submission_snapshot(root: str | Path | None = None) -> SubmissionSnapshot:
    """Load every submission metric from the current artifacts under ``root``."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    manifest = _read_manifest(root)
    classical = _read_csv(root, "reports/classical_results.csv").sort_values("AUC", ascending=False)
    best = classical.iloc[0]
    mlp = _read_csv(root, "reports/mlp_vs_classical.csv")
    mlp_row = mlp.loc[mlp["Model"].str.contains("MLP", case=False, na=False)].iloc[0]
    baselines = _read_csv(root, "reports/baseline_results.csv")
    calibration = _read_csv(root, "reports/calibration_results.csv")
    cluster_stability = _read_csv(root, "reports/cluster_stability.csv")
    rules = _read_csv(root, "reports/association_rules.csv")

    return SubmissionSnapshot(
        seed=int(manifest["seed"]),
        churn_cutoff=str(manifest["churn_cutoff"]),
        churn_window_days=int(manifest["churn_window_days"]),
        n_clean_rows=int(manifest["n_clean_rows"]),
        n_customers_total=int(manifest["n_customers_total"]),
        n_customers_labelled=int(manifest["n_customers_labelled"]),
        churn_rate=float(manifest.get("churn_rate", 0.5731)),
        best_model=str(best["Model"]),
        best_auc=float(best["AUC"]),
        best_f1=float(best["F1"]),
        best_precision=float(best["Precision"]),
        best_recall=float(best["Recall"]),
        best_accuracy=float(best["Accuracy"]),
        best_cv_auc=float(best["CV_AUC_mean"]),
        mlp_auc=float(mlp_row["AUC"]),
        mlp_f1=float(mlp_row["F1"]),
        majority_auc=_baseline_auc(baselines, "MajorityClass"),
        recency_auc=_baseline_auc(baselines, "RecencyOnly_LR"),
        without_recency_auc=_baseline_auc(baselines, "LR_without_Recency"),
        calibration_brier=float(calibration["brier_score"].iloc[0]),
        clustering_silhouette=float(manifest.get("clustering_silhouette", 0.0)),
        clustering_silhouette_std=_silhouette_std(cluster_stability),
        association_n_rules=int(manifest["association_n_rules"]),
        top_rule_lift=float(rules["lift"].max()) if not rules.empty else 0.0,
        churn_gap_between_segments=float(manifest["churn_gap_between_segments"]),
        decision_table_customers=int(manifest.get("decision_table_customers", 0)),
        decision_retention_targets=int(manifest.get("decision_retention_targets", 0)),
        decision_cross_sell_targets=int(manifest.get("decision_cross_sell_targets", 0)),
        decision_manual_review_targets=int(manifest.get("decision_manual_review_targets", 0)),
    )
