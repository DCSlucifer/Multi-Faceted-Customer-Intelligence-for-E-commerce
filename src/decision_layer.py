"""Business Decision Layer: turn churn scores, segments, anomaly flags and
association rules into a per-customer decision table.

Pure decision functions live at the top (independently testable); I/O and
artifact assembly live in :func:`build_decision_artifacts` and :func:`run`.

See docs/superpowers/specs/2026-06-02-business-decision-layer-design.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts as A
from .utils import MODELS_DIR, REPORTS_DIR, get_logger

log = get_logger("decision-layer")

# Risk-tier cutoffs (assume a calibrated [0, 1] probability).
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40
# Segment-churn context bands.
HIGH_SEGMENT_CHURN = 0.70
MEDIUM_SEGMENT_CHURN = 0.40

ACTION_VIP = "VIP nurture / Cross-sell"

DECISION_TABLE_COLUMNS = [
    "CustomerID", "source_run_generated_at",
    "Recency", "Frequency", "Monetary", "avg_basket_value",
    "unique_products", "DominantCountry",
    "churn_probability", "churn_prediction", "churn_risk_tier", "churn_label",
    "segment", "segment_churn_rate", "anomaly_flag", "anomaly_score",
    "customer_value_tier",
    "priority_rank", "priority_score", "recommended_action", "action_reason",
    "retention_offer_type", "cross_sell_signal", "next_best_offer",
]


def churn_risk_tier(prob: float) -> str:
    """Map a calibrated churn probability to a risk tier.

    ``"High"`` if prob >= HIGH_RISK, ``"Medium"`` if prob >= MEDIUM_RISK,
    else ``"Low"``.
    """
    if prob >= HIGH_RISK:
        return "High"
    if prob >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def assign_value_tiers(monetary: pd.Series) -> pd.Series:
    """High = top quartile by Monetary, Low = bottom quartile, else Medium.

    Thresholds are computed at runtime from the scored set (no hard-coded cuts).
    """
    q25 = monetary.quantile(0.25)
    q75 = monetary.quantile(0.75)

    def tier(v: float) -> str:
        if v >= q75:
            return "High"
        if v < q25:
            return "Low"
        return "Medium"

    return monetary.apply(tier)


def priority_score(*, risk_tier: str, value_tier: str, segment_churn_rate: float,
                   anomaly_flag: bool, recency_top_quartile: bool) -> int:
    """Interpretable additive score in [0, 120]. See spec section 6.2."""
    score = 0
    if risk_tier == "High":
        score += 50
    elif risk_tier == "Medium":
        score += 30
    if value_tier == "High":
        score += 25
    elif value_tier == "Medium":
        score += 10
    rate = segment_churn_rate if pd.notna(segment_churn_rate) else 0.0
    if rate >= HIGH_SEGMENT_CHURN:
        score += 20
    elif rate >= MEDIUM_SEGMENT_CHURN:
        score += 10
    if anomaly_flag:
        score += 15
    if recency_top_quartile:
        score += 10
    return score


def recommended_action(*, risk_tier: str, value_tier: str, anomaly_flag: bool,
                       segment_churn_rate: float,
                       recency_top_quartile: bool) -> tuple[str, str]:
    """Return (action, action_reason) following the spec section 6.3 precedence."""
    if risk_tier == "High" and value_tier == "High":
        action, reason = "Retain immediately", "High churn risk and high value"
    elif risk_tier == "High":
        action, reason = "Win-back campaign", "High churn risk, lower value"
    elif risk_tier == "Low" and value_tier == "High":
        action, reason = ACTION_VIP, "Low churn risk and high value"
    elif anomaly_flag:
        action, reason = "Manual review", "Anomalous behaviour, no stronger action"
    elif recency_top_quartile:
        action, reason = "Reactivation campaign", "Dormant (top-quartile recency)"
        rate = segment_churn_rate if pd.notna(segment_churn_rate) else 0.0
        if rate >= HIGH_SEGMENT_CHURN:
            reason += "; in high-churn segment"
    else:
        action, reason = "Monitor / Standard nurture", "No urgent signal"

    if anomaly_flag and action != "Manual review":
        reason += "; anomaly flagged"
    return action, reason


def retention_offer_type(*, risk_tier: str, value_tier: str) -> str:
    if risk_tier == "High" and value_tier == "High":
        return "High-touch voucher"
    if risk_tier == "High":
        return "Light discount"
    if risk_tier == "Low" and value_tier == "High":
        return "Loyalty perk"
    return "None"


def score_churn(features: pd.DataFrame, model,
                feature_columns: list[str]) -> pd.DataFrame:
    """Score the saved best classical pipeline on the raw feature columns.

    The saved estimator is a fitted imblearn pipeline (preprocessor -> SMOTE ->
    classifier); SMOTE is inert at predict time, so raw columns go in directly.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Scored model {type(model).__name__} has no predict_proba; risk "
            "tiers require a calibrated [0, 1] probability. Re-run "
            "`python scripts/run_pipeline.py --stage classification`."
        )
    X = features[feature_columns]
    proba = model.predict_proba(X)
    if proba.shape[1] != 2:
        raise ValueError(
            f"Expected a binary classifier (2 predict_proba columns), "
            f"got {proba.shape[1]}.")
    prob = proba[:, 1]
    pred = model.predict(X)
    # Build from plain ndarrays so an arbitrary pandas index on ``features``
    # cannot misalign the output rows.
    out = pd.DataFrame({
        "CustomerID": features["CustomerID"].to_numpy(),
        "churn_probability": prob,
        "churn_prediction": np.asarray(pred, dtype=int),
    })
    out["churn_risk_tier"] = out["churn_probability"].apply(churn_risk_tier)
    return out


def _top_offer(rules: pd.DataFrame) -> str:
    """Human-readable consequent of the highest-lift rule, or '' if none."""
    if rules.empty or "consequents_desc" not in rules.columns:
        return ""
    top = rules.sort_values("lift", ascending=False).iloc[0]
    return str(top["consequents_desc"])


def build_decision_artifacts(*, features: pd.DataFrame, churn_labels: pd.DataFrame,
                             segments: pd.DataFrame, churn_by_cluster: pd.DataFrame,
                             anomalies: pd.DataFrame, rules: pd.DataFrame, model,
                             feature_columns: list[str], scored_model_name: str,
                             source_run_generated_at: str,
                             decision_generated_at: str) -> dict:
    """Assemble the four decision outputs + manifest metadata (pure, no I/O)."""
    scores = score_churn(features, model, feature_columns)

    t = features[[
        "CustomerID", "Recency", "Frequency", "Monetary",
        "avg_basket_value", "unique_products", "DominantCountry",
    ]].copy()
    t = t.merge(scores, on="CustomerID", how="left")
    t["scored_model"] = scored_model_name

    # churn label
    t = t.merge(churn_labels.rename(columns={"churn": "churn_label"}),
                on="CustomerID", how="left")

    # segment + segment churn rate
    seg = segments[["CustomerID", "cluster"]].merge(
        churn_by_cluster[["cluster", "churn_rate"]], on="cluster", how="left")
    seg = seg.rename(columns={"cluster": "segment",
                              "churn_rate": "segment_churn_rate"})
    t = t.merge(seg, on="CustomerID", how="left")
    t["segment_churn_rate"] = t["segment_churn_rate"].fillna(0.0)

    # anomaly flag + nullable score
    anomaly_map = (anomalies.set_index("CustomerID")["reconstruction_error"]
                   if "reconstruction_error" in anomalies.columns
                   else pd.Series(dtype=float))
    t["anomaly_flag"] = t["CustomerID"].isin(set(anomalies["CustomerID"]))
    t["anomaly_score"] = t["CustomerID"].map(anomaly_map)

    # value tiers + dormancy
    t["customer_value_tier"] = assign_value_tiers(t["Monetary"])
    recency_cut = t["Recency"].quantile(0.75)
    t["recency_top_quartile_flag"] = t["Recency"] >= recency_cut

    # priority score
    t["priority_score"] = [
        priority_score(risk_tier=r.churn_risk_tier, value_tier=r.customer_value_tier,
                       segment_churn_rate=r.segment_churn_rate,
                       anomaly_flag=bool(r.anomaly_flag),
                       recency_top_quartile=bool(r.recency_top_quartile_flag))
        for r in t.itertuples(index=False)
    ]

    # action + reason
    actions = [
        recommended_action(risk_tier=r.churn_risk_tier, value_tier=r.customer_value_tier,
                           anomaly_flag=bool(r.anomaly_flag),
                           segment_churn_rate=r.segment_churn_rate,
                           recency_top_quartile=bool(r.recency_top_quartile_flag))
        for r in t.itertuples(index=False)
    ]
    t["recommended_action"] = [a for a, _ in actions]
    t["action_reason"] = [reason for _, reason in actions]

    # retention offer
    t["retention_offer_type"] = [
        retention_offer_type(risk_tier=r.churn_risk_tier,
                             value_tier=r.customer_value_tier)
        for r in t.itertuples(index=False)
    ]

    # cross-sell (minimum-viable: VIP rows get the generic top-lift offer)
    top_offer = _top_offer(rules)
    is_vip = t["recommended_action"] == ACTION_VIP
    t["cross_sell_signal"] = is_vip.map({True: "Yes", False: "No"})
    t["next_best_offer"] = ""
    t.loc[is_vip, "next_best_offer"] = top_offer

    # provenance + deterministic rank
    t["source_run_generated_at"] = source_run_generated_at
    t = t.sort_values(["priority_score", "Monetary", "CustomerID"],
                      ascending=[False, False, True]).reset_index(drop=True)
    t["priority_rank"] = range(1, len(t) + 1)

    decision_table = t[DECISION_TABLE_COLUMNS].copy()
    if len(decision_table) != len(features):
        raise ValueError(
            f"Decision table has {len(decision_table)} rows but {len(features)} "
            "customers were scored; an input join fanned out (duplicate keys in "
            "segments, churn_by_cluster, or churn_labels).")

    churn_scores = decision_table[[
        "CustomerID", "churn_probability", "churn_prediction", "churn_risk_tier",
    ]].copy()
    churn_scores["scored_model"] = scored_model_name

    summary = _build_summary(decision_table)
    top = decision_table.sort_values("priority_rank").head(25).copy()

    metadata = {
        "decision_table_customers": int(len(decision_table)),
        "decision_high_priority_customers": int((decision_table["churn_risk_tier"] == "High").sum()),
        "decision_retention_targets": int(decision_table["recommended_action"].isin(
            ["Retain immediately", "Win-back campaign"]).sum()),
        "decision_cross_sell_targets": int((decision_table["cross_sell_signal"] == "Yes").sum()),
        "decision_manual_review_targets": int((decision_table["recommended_action"] == "Manual review").sum()),
        "decision_scored_model": scored_model_name,
        "decision_source_run_generated_at": source_run_generated_at,
        "decision_generated_at": decision_generated_at,
    }
    return {"churn_scores": churn_scores, "decision_table": decision_table,
            "summary": summary, "top": top, "metadata": metadata}


def _build_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Action-bucket summary for the report table."""
    total = len(table)
    grouped = table.groupby(
        ["recommended_action", "churn_risk_tier", "customer_value_tier", "segment"],
        dropna=False)
    summary = grouped.agg(
        n_customers=("CustomerID", "count"),
        avg_monetary=("Monetary", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        anomaly_count=("anomaly_flag", "sum"),
    ).reset_index()
    summary["share_of_customers"] = (summary["n_customers"] / total).round(4)
    summary["avg_monetary"] = summary["avg_monetary"].round(2)
    summary["avg_churn_probability"] = summary["avg_churn_probability"].round(4)
    summary["anomaly_count"] = summary["anomaly_count"].astype(int)
    return summary[[
        "recommended_action", "churn_risk_tier", "customer_value_tier", "segment",
        "n_customers", "share_of_customers", "avg_monetary",
        "avg_churn_probability", "anomaly_count",
    ]].sort_values("n_customers", ascending=False).reset_index(drop=True)


def run() -> dict:
    """Load current artifacts, build the decision outputs, write CSVs, and
    return the manifest metadata dict (caller persists it via update_manifest)."""
    import joblib

    manifest = A.read_manifest()
    # Provenance: capture the upstream run timestamp BEFORE we write anything.
    source_run_generated_at = str(manifest.get("generated_at", ""))
    scored_model_name = manifest["classification_best_model"]
    feature_columns = list(manifest["feature_columns"])

    model_path = MODELS_DIR / f"{scored_model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model '{scored_model_name}' not found at {model_path}. "
            "Re-run `python scripts/run_pipeline.py --stage classification`.")
    model = joblib.load(model_path)

    features = pd.read_csv(A.ARTIFACTS["classification_features"])
    churn_labels = pd.read_csv(A.ARTIFACTS["churn_labels"])
    segments = pd.read_csv(A.ARTIFACTS["segments_kmeans"])
    churn_by_cluster = pd.read_csv(REPORTS_DIR / "churn_by_cluster.csv")
    anomalies = pd.read_csv(REPORTS_DIR / "anomaly_customers.csv")
    rules = pd.read_csv(A.ARTIFACTS["association_rules"])

    decision_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    built = build_decision_artifacts(
        features=features, churn_labels=churn_labels, segments=segments,
        churn_by_cluster=churn_by_cluster, anomalies=anomalies, rules=rules,
        model=model, feature_columns=feature_columns,
        scored_model_name=scored_model_name,
        source_run_generated_at=source_run_generated_at,
        decision_generated_at=decision_generated_at,
    )

    built["churn_scores"].to_csv(A.ARTIFACTS["customer_churn_scores"], index=False)
    built["decision_table"].to_csv(A.ARTIFACTS["customer_decision_table"], index=False)
    built["summary"].to_csv(A.ARTIFACTS["customer_decision_summary"], index=False)
    built["top"].to_csv(A.ARTIFACTS["top_priority_customers"], index=False)
    log.info("[decision-layer] wrote %d-customer decision table",
             built["metadata"]["decision_table_customers"])
    return built["metadata"]
