"""Decision-layer tests: pure rules (always run) + real-artifact schema checks
(skip until the pipeline has produced the outputs)."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import decision_layer as dl
from src.features import make_preprocessor


# --------------------------------------------------------------------------- #
# Pure rule functions                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prob,expected", [
    (0.95, "High"), (0.70, "High"),
    (0.69, "Medium"), (0.40, "Medium"),
    (0.39, "Low"), (0.0, "Low"),
])
def test_churn_risk_tier(prob, expected):
    assert dl.churn_risk_tier(prob) == expected


def test_assign_value_tiers_quartiles():
    monetary = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    tiers = dl.assign_value_tiers(monetary)
    # Q25 == 27.5, Q75 == 62.5 -> >=62.5 High, <27.5 Low, else Medium
    assert tiers.iloc[-1] == "High"   # 80
    assert tiers.iloc[0] == "Low"     # 10
    assert tiers.iloc[3] == "Medium"  # 40
    # boundary checks around Q25 (27.5) and Q75 (62.5)
    assert tiers.iloc[1] == "Low"     # 20 < 27.5
    assert tiers.iloc[2] == "Medium"  # 30 >= 27.5, < 62.5
    assert tiers.iloc[5] == "Medium"  # 60 >= 27.5, < 62.5
    assert tiers.iloc[6] == "High"    # 70 >= 62.5


def test_priority_score_caps_and_components():
    # High risk + High value + high-churn segment + anomaly + dormant
    assert dl.priority_score(
        risk_tier="High", value_tier="High", segment_churn_rate=0.90,
        anomaly_flag=True, recency_top_quartile=True) == 50 + 25 + 20 + 15 + 10
    # No signal at all
    assert dl.priority_score(
        risk_tier="Low", value_tier="Low", segment_churn_rate=0.10,
        anomaly_flag=False, recency_top_quartile=False) == 0
    # Medium bands
    assert dl.priority_score(
        risk_tier="Medium", value_tier="Medium", segment_churn_rate=0.50,
        anomaly_flag=False, recency_top_quartile=False) == 30 + 10 + 10


def test_recommended_action_retain_immediately():
    action, reason = dl.recommended_action(
        risk_tier="High", value_tier="High", anomaly_flag=False,
        segment_churn_rate=0.5, recency_top_quartile=False)
    assert action == "Retain immediately"


@pytest.mark.parametrize("value_tier", ["Medium", "Low"])
def test_recommended_action_winback(value_tier):
    action, _ = dl.recommended_action(
        risk_tier="High", value_tier=value_tier, anomaly_flag=False,
        segment_churn_rate=0.5, recency_top_quartile=False)
    assert action == "Win-back campaign"


def test_recommended_action_vip():
    action, _ = dl.recommended_action(
        risk_tier="Low", value_tier="High", anomaly_flag=False,
        segment_churn_rate=0.1, recency_top_quartile=False)
    assert action == "VIP nurture / Cross-sell"


def test_recommended_action_manual_review_when_no_stronger():
    action, reason = dl.recommended_action(
        risk_tier="Medium", value_tier="Medium", anomaly_flag=True,
        segment_churn_rate=0.1, recency_top_quartile=False)
    assert action == "Manual review"


def test_recommended_action_anomaly_kept_in_reason_when_stronger_wins():
    action, reason = dl.recommended_action(
        risk_tier="High", value_tier="High", anomaly_flag=True,
        segment_churn_rate=0.5, recency_top_quartile=False)
    assert action == "Retain immediately"
    assert "anomaly" in reason.lower()


def test_recommended_action_reactivation_for_dormant():
    action, _ = dl.recommended_action(
        risk_tier="Low", value_tier="Low", anomaly_flag=False,
        segment_churn_rate=0.9, recency_top_quartile=True)
    assert action == "Reactivation campaign"


def test_recommended_action_default_monitor():
    action, _ = dl.recommended_action(
        risk_tier="Low", value_tier="Low", anomaly_flag=False,
        segment_churn_rate=0.1, recency_top_quartile=False)
    assert action == "Monitor / Standard nurture"


def test_recommended_action_medium_risk_low_value_monitors():
    # Medium risk alone (no anomaly, not dormant) must not escalate to Win-back.
    action, _ = dl.recommended_action(
        risk_tier="Medium", value_tier="Low", anomaly_flag=False,
        segment_churn_rate=0.1, recency_top_quartile=False)
    assert action == "Monitor / Standard nurture"


@pytest.mark.parametrize("risk,value,expected", [
    ("High", "High", "High-touch voucher"),
    ("High", "Medium", "Light discount"),
    ("High", "Low", "Light discount"),
    ("Low", "High", "Loyalty perk"),
    ("Medium", "Medium", "None"),
])
def test_retention_offer_type(risk, value, expected):
    assert dl.retention_offer_type(risk_tier=risk, value_tier=value) == expected


# --------------------------------------------------------------------------- #
# score_churn integration tests                                               #
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary", "avg_basket_value",
    "avg_basket_size", "unique_products", "DominantCountry",
]


def _fit_dummy_pipeline(frame):
    # Plain sklearn Pipeline is fine here: SMOTE (used by the production
    # imblearn pipeline) is inert at predict time, which is all we score.
    pipe = Pipeline([("pre", make_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(frame[FEATURE_COLUMNS], frame["churn"])
    return pipe


def test_score_churn_returns_valid_probabilities(supervised_frame):
    model = _fit_dummy_pipeline(supervised_frame)
    scores = dl.score_churn(supervised_frame, model, FEATURE_COLUMNS)

    assert list(scores.columns) == [
        "CustomerID", "churn_probability", "churn_prediction", "churn_risk_tier"]
    assert len(scores) == len(supervised_frame)
    assert scores["churn_probability"].between(0.0, 1.0).all()
    assert set(scores["churn_prediction"].unique()) <= {0, 1}
    assert set(scores["churn_risk_tier"].unique()) <= {"High", "Medium", "Low"}


def test_score_churn_rejects_model_without_predict_proba(supervised_frame):
    class NoProba:
        def predict(self, X):
            return [0] * len(X)
    with pytest.raises(TypeError, match="predict_proba"):
        dl.score_churn(supervised_frame, NoProba(), FEATURE_COLUMNS)


def _synthetic_inputs():
    """Sixty synthetic customers spanning the action branches + a fitted model."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Recency": rng.integers(0, 400, n),
        "Frequency": rng.integers(1, 30, n),
        "Monetary": rng.gamma(2.0, 500, n),
        "avg_basket_value": rng.gamma(2.0, 80, n),
        "avg_basket_size": rng.gamma(2.0, 20, n),
        "unique_products": rng.integers(1, 120, n),
        "DominantCountry": rng.choice(["United Kingdom", "France", "Germany"], n),
        "churn": rng.integers(0, 2, n),
    })
    model = _fit_dummy_pipeline(frame)
    features = frame.drop(columns=["churn"])
    churn_labels = frame[["CustomerID", "churn"]].copy()
    segments = pd.DataFrame({
        "CustomerID": frame["CustomerID"],
        "cluster": (frame["CustomerID"] % 2),
    })
    churn_by_cluster = pd.DataFrame({
        "cluster": [0, 1], "n_customers": [30, 30], "churn_rate": [0.90, 0.23]})
    anomalies = features.head(5).assign(reconstruction_error=[0.9, 0.8, 0.7, 0.6, 0.5])
    rules = pd.DataFrame({
        "antecedents": ["frozenset({'A'})"], "consequents": ["frozenset({'B'})"],
        "lift": [26.7], "consequents_desc": ["GREEN REGENCY TEACUP AND SAUCER"]})
    return features, churn_labels, segments, churn_by_cluster, anomalies, rules, model


def test_build_decision_artifacts_shapes_and_invariants():
    (features, churn_labels, segments, churn_by_cluster,
     anomalies, rules, model) = _synthetic_inputs()

    built = dl.build_decision_artifacts(
        features=features, churn_labels=churn_labels, segments=segments,
        churn_by_cluster=churn_by_cluster, anomalies=anomalies, rules=rules,
        model=model, feature_columns=FEATURE_COLUMNS,
        scored_model_name="LogisticRegression",
        source_run_generated_at="2026-05-31T07:29:07+00:00",
        decision_generated_at="2026-06-02T10:00:00+00:00",
    )

    table = built["decision_table"]
    # One row per scored customer, exact column order.
    assert len(table) == len(features)
    assert list(table.columns) == dl.DECISION_TABLE_COLUMNS
    # No-null guarantees.
    assert table["recommended_action"].notna().all()
    assert table["priority_score"].notna().all()
    # anomaly_score nullable: exactly the 5 flagged customers have a value.
    assert table["anomaly_flag"].sum() == 5
    assert table["anomaly_score"].notna().sum() == 5
    # priority_rank is a contiguous 1..N permutation.
    assert sorted(table["priority_rank"]) == list(range(1, len(table) + 1))
    # rank ordering matches the tiebreaker.
    ordered = table.sort_values(
        ["priority_score", "Monetary", "CustomerID"],
        ascending=[False, False, True]).reset_index(drop=True)
    assert (ordered["priority_rank"].to_numpy() == range(1, len(table) + 1)).all()
    # source provenance stamped identically on every row.
    assert table["source_run_generated_at"].nunique() == 1

    # Cross-sell rows carry the top-lift offer; others are blank.
    vip = table["recommended_action"] == dl.ACTION_VIP
    assert (table.loc[vip, "cross_sell_signal"] == "Yes").all()
    assert (table.loc[vip, "next_best_offer"]
            == "GREEN REGENCY TEACUP AND SAUCER").all()
    assert (table.loc[~vip, "cross_sell_signal"] == "No").all()

    # churn_scores subset of columns.
    assert list(built["churn_scores"].columns) == [
        "CustomerID", "churn_probability", "churn_prediction",
        "churn_risk_tier", "scored_model"]

    # top_priority_customers is the top 25 by rank and a subset of the table.
    top = built["top"]
    assert len(top) == min(25, len(table))
    assert set(top["CustomerID"]).issubset(set(table["CustomerID"]))

    # metadata counts are internally consistent.
    meta = built["metadata"]
    assert meta["decision_table_customers"] == len(table)
    assert meta["decision_scored_model"] == "LogisticRegression"
    assert meta["decision_source_run_generated_at"] == "2026-05-31T07:29:07+00:00"
    assert meta["decision_cross_sell_targets"] == int((table["cross_sell_signal"] == "Yes").sum())


def test_cross_sell_offer_assigned_to_vip_rows():
    """Deterministic VIP scenario: a low-risk high-value customer gets the
    top-lift offer; everyone else's next_best_offer is blank."""
    import numpy as np

    class _FixedProbaModel:
        def __init__(self, probs):
            self._p = np.asarray(probs, dtype=float)

        def predict_proba(self, X):
            p = self._p[:len(X)]
            return np.column_stack([1.0 - p, p])

        def predict(self, X):
            return (self._p[:len(X)] >= 0.5).astype(int)

    n = 5
    features = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Recency": [10, 20, 30, 40, 50],
        "Frequency": [5, 5, 5, 5, 5],
        "Monetary": [10.0, 20.0, 30.0, 40.0, 1000.0],  # customer 5 = top value
        "avg_basket_value": [1.0] * n,
        "avg_basket_size": [1.0] * n,
        "unique_products": [1] * n,
        "DominantCountry": ["United Kingdom"] * n,
    })
    # customer 5: lowest churn prob (Low risk) + highest Monetary (High value) -> VIP
    probs = [0.9, 0.9, 0.9, 0.9, 0.05]
    model = _FixedProbaModel(probs)
    churn_labels = pd.DataFrame(
        {"CustomerID": np.arange(1, n + 1), "churn": [1, 1, 1, 1, 0]})
    segments = pd.DataFrame(
        {"CustomerID": np.arange(1, n + 1), "cluster": [0] * n})
    churn_by_cluster = pd.DataFrame(
        {"cluster": [0], "n_customers": [n], "churn_rate": [0.8]})
    anomalies = pd.DataFrame({"CustomerID": [], "reconstruction_error": []})
    rules = pd.DataFrame({"lift": [10.0], "consequents_desc": ["RED TEACUP"]})

    built = dl.build_decision_artifacts(
        features=features, churn_labels=churn_labels, segments=segments,
        churn_by_cluster=churn_by_cluster, anomalies=anomalies, rules=rules,
        model=model, feature_columns=FEATURE_COLUMNS,
        scored_model_name="Fixed", source_run_generated_at="x",
        decision_generated_at="y")

    table = built["decision_table"]
    vip = table["recommended_action"] == dl.ACTION_VIP
    # The VIP branch is genuinely exercised (not vacuous).
    assert vip.sum() >= 1
    # Cross-sell signal matches VIP membership for EVERY row (full contract).
    assert (((table["recommended_action"] == dl.ACTION_VIP))
            == (table["cross_sell_signal"] == "Yes")).all()
    # VIP rows carry the top-lift offer; non-VIP rows are blank.
    assert (table.loc[vip, "next_best_offer"] == "RED TEACUP").all()
    assert (table.loc[~vip, "next_best_offer"] == "").all()


# --------------------------------------------------------------------------- #
# Real-artifact schema checks (skip until the pipeline has produced outputs)   #
# --------------------------------------------------------------------------- #
from src import artifacts as A


def _require(key: str):
    path = A.ARTIFACTS[key]
    if not path.exists():
        pytest.skip(f"{key} not generated yet ({path.name}); run the pipeline first")
    return path


@pytest.mark.parametrize("key", [
    "customer_churn_scores", "customer_decision_table",
    "customer_decision_summary", "top_priority_customers",
])
def test_decision_csv_schema(key):
    _require(key)
    missing = A.check_csv_schema(key)
    assert not missing, f"{key} missing columns {missing}"


def test_decision_table_matches_scored_set():
    _require("customer_decision_table")
    _require("classification_features")
    table = pd.read_csv(A.ARTIFACTS["customer_decision_table"])
    feats = pd.read_csv(A.ARTIFACTS["classification_features"])
    assert len(table) == len(feats)
    assert table["recommended_action"].notna().all()
    assert table["priority_score"].notna().all()
    assert sorted(table["priority_rank"]) == list(range(1, len(table) + 1))


def test_top_priority_is_subset_of_table():
    _require("top_priority_customers")
    _require("customer_decision_table")
    top = pd.read_csv(A.ARTIFACTS["top_priority_customers"])
    table = pd.read_csv(A.ARTIFACTS["customer_decision_table"])
    assert set(top["CustomerID"]).issubset(set(table["CustomerID"]))
