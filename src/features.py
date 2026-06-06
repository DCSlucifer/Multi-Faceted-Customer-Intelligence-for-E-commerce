"""Feature matrix construction for clustering and classification.

The classification side is **leakage-safe by construction**:

* Features come from ``classification_features.csv``, which is built only from
  transactions strictly before ``CHURN_CUTOFF`` (see ``preprocessing.py``).
* Categorical encoding (``DominantCountry``) is performed by a
  :class:`~sklearn.compose.ColumnTransformer` that is fit *inside* each
  cross-validation fold / on the training split only. Rare countries are folded
  into an ``infrequent`` bucket and unseen countries at test time are ignored
  (``handle_unknown="ignore"``) — so neither the country vocabulary nor the
  top-k selection ever sees the held-out rows.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Column groups (single source of truth for the supervised task)              #
# --------------------------------------------------------------------------- #
RFM_COLS = ["Recency", "Frequency", "Monetary"]
BEHAV_COLS = ["avg_basket_value", "avg_basket_size", "unique_products"]

NUMERIC_FEATURES: List[str] = RFM_COLS + BEHAV_COLS
CATEGORICAL_FEATURES: List[str] = ["DominantCountry"]

# Right-skewed, strictly-non-negative columns get a log1p before scaling.
# log1p is a fixed, parameter-free transform, so applying it before the split
# introduces no leakage (nothing is *fit* on the data).
LOG_FEATURES: List[str] = ["Frequency", "Monetary", "avg_basket_value",
                           "avg_basket_size", "unique_products"]
LINEAR_FEATURES: List[str] = ["Recency"]

TARGET = "churn"


def log1p_rfm(rfm: pd.DataFrame, cols=("Frequency", "Monetary")) -> pd.DataFrame:
    """Log-transform right-skewed numerical features (returns a copy)."""
    out = rfm.copy()
    for c in cols:
        out[c] = np.log1p(out[c])
    return out


# --------------------------------------------------------------------------- #
# Clustering matrix (unsupervised — uses full-history RFM)                    #
# --------------------------------------------------------------------------- #
def build_clustering_matrix(rfm: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Return scaled, log-transformed RFM matrix for clustering algorithms."""
    transformed = log1p_rfm(rfm)
    X = transformed[RFM_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler, transformed


# --------------------------------------------------------------------------- #
# Supervised (classification) frame + leakage-safe preprocessor               #
# --------------------------------------------------------------------------- #
def build_supervised_frame(features: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-cutoff features with churn labels into one tidy frame.

    Returns a DataFrame with ``CustomerID`` + :data:`NUMERIC_FEATURES` +
    :data:`CATEGORICAL_FEATURES` + :data:`TARGET`. No transformation or encoding
    is applied here — that is deferred to :func:`make_preprocessor` so it can be
    fit on the training split only.
    """
    frame = features.merge(churn, on="CustomerID", how="inner")

    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
               if c not in frame.columns]
    if missing:
        raise KeyError(f"build_supervised_frame: missing required columns {missing}")

    frame = frame.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()
    frame[CATEGORICAL_FEATURES[0]] = frame[CATEGORICAL_FEATURES[0]].astype(str)
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def make_preprocessor(
    features: List[str] | None = None,
    min_country_frequency: float = 0.01,
) -> ColumnTransformer:
    """Leakage-safe feature transformer for the churn classifier.

    Pipeline per column group:

    * log-then-scale for skewed numerics (:data:`LOG_FEATURES`),
    * scale-only for the rest (:data:`LINEAR_FEATURES`),
    * :class:`OneHotEncoder` with rare-country folding (``min_frequency``) and
      ``handle_unknown="infrequent_if_exist"`` for :data:`CATEGORICAL_FEATURES`.

    Pass ``features`` to restrict to a subset (used by the ablation study);
    defaults to the full feature set. Designed to live *inside* a
    scikit-learn / imblearn ``Pipeline`` so that
    :class:`~sklearn.model_selection.GridSearchCV` re-fits it on each training
    fold — the country vocabulary never sees the validation/test rows.
    """
    features = features or (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    log_cols = [f for f in features if f in LOG_FEATURES]
    lin_cols = [f for f in features if f in NUMERIC_FEATURES and f not in LOG_FEATURES]
    cat_cols = [f for f in features if f in CATEGORICAL_FEATURES]

    log_pipe = Pipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    lin_pipe = Pipeline(steps=[("scale", StandardScaler())])
    cat_pipe = Pipeline(steps=[(
        "ohe",
        OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_country_frequency,
            sparse_output=False,
        ),
    )])

    transformers = []
    if log_cols:
        transformers.append(("log", log_pipe, log_cols))
    if lin_cols:
        transformers.append(("lin", lin_pipe, lin_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Backward-compatible matrix builder (deprecated)                             #
# --------------------------------------------------------------------------- #
def build_classification_matrix(
    rfm: pd.DataFrame,
    churn: pd.DataFrame,
    top_k_countries: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """**Deprecated.** Eagerly one-hot encodes top-k countries on the *full*
    labelled frame (a mild pre-split leak) and log-transforms RFM.

    Kept only so older callers keep working. New code should use
    :func:`build_supervised_frame` + :func:`make_preprocessor` inside the
    training pipeline (see :func:`src.classification.train_classical`).
    """
    df = rfm.merge(churn, on="CustomerID", how="inner")

    top_countries = df["DominantCountry"].value_counts().head(top_k_countries).index
    df["DominantCountry"] = df["DominantCountry"].where(
        df["DominantCountry"].isin(top_countries), other="Other"
    )
    country_dummies = pd.get_dummies(df["DominantCountry"], prefix="country", dtype=int)

    transformed = log1p_rfm(df)
    feature_cols = RFM_COLS + BEHAV_COLS
    X = pd.concat([transformed[feature_cols], country_dummies], axis=1)
    y = df["churn"].astype(int)
    return X, y, list(X.columns)
