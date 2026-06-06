"""Supervised-frame builder + leakage-safe preprocessor tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_supervised_frame,
    make_preprocessor,
)


def test_build_supervised_frame_schema(supervised_frame):
    feats = supervised_frame.drop(columns=[TARGET])
    churn = supervised_frame[["CustomerID", TARGET]]
    frame = build_supervised_frame(feats, churn)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in frame.columns
    assert frame[TARGET].dtype.kind == "i"
    assert set(frame[TARGET].unique()) <= {0, 1}
    assert not frame.isnull().any().any()


def test_build_supervised_frame_missing_column_raises():
    feats = pd.DataFrame({"CustomerID": [1, 2], "Recency": [1, 2]})
    churn = pd.DataFrame({"CustomerID": [1, 2], "churn": [0, 1]})
    with pytest.raises(KeyError):
        build_supervised_frame(feats, churn)


def test_preprocessor_handles_unseen_country(supervised_frame):
    """A country present only in test must not break transform (no leakage of
    the country vocabulary into the fit)."""
    train = supervised_frame[supervised_frame["DominantCountry"] != "Rareland"]
    test = supervised_frame[supervised_frame["DominantCountry"] == "Rareland"]
    assert len(test) >= 1  # fixture guarantees at least one rare row

    pre = make_preprocessor().fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    Xt = pre.transform(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    assert Xt.shape[0] == len(test)
    assert np.isfinite(Xt).all()
    # Train and test transform to the same number of columns.
    assert Xt.shape[1] == pre.transform(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]).shape[1]


def test_preprocessor_subset_features(supervised_frame):
    subset = ["Recency", "Frequency", "Monetary"]
    pre = make_preprocessor(subset).fit(supervised_frame[subset])
    out = pre.transform(supervised_frame[subset])
    assert out.shape == (len(supervised_frame), len(subset))
    assert np.isfinite(out).all()
