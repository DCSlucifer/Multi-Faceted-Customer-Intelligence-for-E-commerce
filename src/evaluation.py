"""Honest baselines, ablation study, and probability calibration.

These three analyses are what move a churn model from "nice AUC" to a
defensible, research-grade result:

* **Baselines** prove the model beats trivial rules (majority class, recency
  only) — and explain why AUC is ~0.78 rather than ~0.99.
* **Ablation** shows *which* feature groups carry the signal.
* **Calibration** checks the probabilities are usable for retention decisions,
  not just rank-ordering.

Every function returns a tidy DataFrame and optionally writes a CSV (+ figure)
to ``reports/`` so the report/slides trace straight back to the artifacts.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .classification import _metrics, _proba, make_pipeline
from .features import (
    BEHAV_COLS,
    NUMERIC_FEATURES,
    RFM_COLS,
    TARGET,
    make_preprocessor,
)
from .utils import REPORTS_DIR, SEED


def _split(frame: pd.DataFrame, feature_columns: List[str], target: str,
           test_size: float, seed: int):
    """Deterministic train/test split — matches ``train_classical`` when called
    with the same ``test_size``/``seed`` so baselines are comparable."""
    X = frame[feature_columns].reset_index(drop=True)
    y = frame[target].astype(int).reset_index(drop=True)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# 1. Honest baselines                                                         #
# --------------------------------------------------------------------------- #
def run_baselines(
    frame: pd.DataFrame,
    *,
    target: str = TARGET,
    test_size: float = 0.2,
    seed: int = SEED,
    save: bool = True,
) -> pd.DataFrame:
    """Evaluate trivial / partial baselines on the held-out test split.

    Baselines: majority class, recency-only logistic regression, logistic
    regression *without* recency, and a full-feature random forest.
    """
    full_features = NUMERIC_FEATURES + ["DominantCountry"]
    rows: List[dict] = []

    def _evaluate(name: str, estimator, features: List[str], use_pipeline: bool = True):
        X_tr, X_te, y_tr, y_te = _split(frame, features, target, test_size, seed)
        if use_pipeline:
            model = make_pipeline(estimator, make_preprocessor(features), seed=seed)
        else:
            model = estimator
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = _proba(model, X_te)
        rows.append({"Baseline": name, "n_features": len(features),
                     **{k: round(v, 4) for k, v in _metrics(y_te, y_pred, y_proba).items()}})

    # Majority class — no preprocessing needed.
    _evaluate("MajorityClass", DummyClassifier(strategy="most_frequent"),
              ["Recency"], use_pipeline=False)
    _evaluate("RecencyOnly_LR", LogisticRegression(max_iter=2000, random_state=seed),
              ["Recency"])
    _evaluate("LR_without_Recency", LogisticRegression(max_iter=2000, random_state=seed),
              [f for f in full_features if f != "Recency"])
    _evaluate("RandomForest_all", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
              full_features)

    table = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "baseline_results.csv", index=False)
    return table


# --------------------------------------------------------------------------- #
# 2. Ablation study                                                           #
# --------------------------------------------------------------------------- #
FEATURE_GROUPS: Dict[str, List[str]] = {
    "RFM_only": RFM_COLS,
    "Behavioral_only": BEHAV_COLS,
    "RFM+Behavioral": RFM_COLS + BEHAV_COLS,
    "RFM+Behavioral+Country": RFM_COLS + BEHAV_COLS + ["DominantCountry"],
    "Without_Recency": [c for c in (RFM_COLS + BEHAV_COLS + ["DominantCountry"]) if c != "Recency"],
}


def run_ablation(
    frame: pd.DataFrame,
    *,
    target: str = TARGET,
    test_size: float = 0.2,
    n_splits: int = 5,
    seed: int = SEED,
    save: bool = True,
    figure: bool = True,
) -> pd.DataFrame:
    """Train one fixed model (LogisticRegression) on each feature group and
    report CV-AUC (mean±std) + held-out test AUC. Isolates where the signal is.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: List[dict] = []

    for group_name, features in FEATURE_GROUPS.items():
        X_tr, X_te, y_tr, y_te = _split(frame, features, target, test_size, seed)
        pipe = make_pipeline(
            LogisticRegression(max_iter=2000, random_state=seed),
            make_preprocessor(features), seed=seed,
        )
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        test_auc = roc_auc_score(y_te, _proba(pipe, X_te))
        rows.append({
            "FeatureGroup": group_name,
            "n_features": len(features),
            "CV_AUC_mean": round(float(cv_auc.mean()), 4),
            "CV_AUC_std": round(float(cv_auc.std()), 4),
            "Test_AUC": round(float(test_auc), 4),
        })

    table = pd.DataFrame(rows)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "ablation_results.csv", index=False)
    if figure:
        _plot_ablation(table)
    return table


def _plot_ablation(table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from .plot_style import PALETTE, apply_style
    from .utils import savefig

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    order = table.sort_values("CV_AUC_mean")
    ax.barh(order["FeatureGroup"], order["CV_AUC_mean"],
            xerr=order["CV_AUC_std"], color=PALETTE[1], alpha=0.9, capsize=4)
    ax.set_xlim(0.5, max(0.85, order["CV_AUC_mean"].max() + 0.05))
    ax.axvline(0.5, ls="--", color="gray", alpha=0.6, label="random (0.5)")
    ax.set_xlabel("5-fold CV AUC (mean ± std)")
    ax.set_title("Ablation: which feature groups carry the churn signal")
    ax.legend(loc="lower right")
    savefig(fig, "03_ablation")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Probability calibration                                                  #
# --------------------------------------------------------------------------- #
def run_calibration(
    estimator,
    X_test,
    y_test,
    *,
    n_bins: int = 10,
    model_name: str = "best",
    save: bool = True,
    figure: bool = True,
) -> pd.DataFrame:
    """Brier score + reliability curve for a fitted probabilistic classifier.

    ``estimator`` must expose ``predict_proba`` and accept the same raw frame
    used during training (i.e. a full pipeline).
    """
    from sklearn.calibration import calibration_curve

    y_test = np.asarray(y_test).ravel()
    proba = estimator.predict_proba(X_test)[:, 1]
    brier = float(brier_score_loss(y_test, proba))
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")

    table = pd.DataFrame({
        "model": model_name,
        "bin": np.arange(1, len(frac_pos) + 1),
        "mean_predicted_prob": np.round(mean_pred, 4),
        "fraction_positive": np.round(frac_pos, 4),
    })
    table["brier_score"] = round(brier, 4)

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORTS_DIR / "calibration_results.csv", index=False)
    if figure:
        _plot_calibration(mean_pred, frac_pos, brier, model_name)
    return table


def _plot_calibration(mean_pred, frac_pos, brier: float, model_name: str) -> None:
    import matplotlib.pyplot as plt
    from .plot_style import PALETTE, apply_style
    from .utils import savefig

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[4], lw=2,
            label=f"{model_name} (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn fraction")
    ax.set_title("Reliability curve — churn probability calibration")
    ax.legend(loc="upper left")
    savefig(fig, "03_calibration")
    plt.close(fig)
