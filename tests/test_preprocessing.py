"""Cleaning + leakage-safe labelling/feature unit tests."""
from __future__ import annotations

import pandas as pd

from src import preprocessing as pp


def test_clean_drops_null_customer_id(raw_transactions):
    out = pp.clean(raw_transactions)
    assert out["CustomerID"].notna().all()
    assert 17850 in out["CustomerID"].values


def test_clean_removes_cancellations(raw_transactions):
    out = pp.clean(raw_transactions)
    assert not out["Invoice"].str.startswith("C").any()


def test_clean_removes_non_positive_quantity_and_price(raw_transactions):
    out = pp.clean(raw_transactions)
    assert (out["Quantity"] > 0).all()
    assert (out["Price"] > 0).all()


def test_clean_adds_total_amount(raw_transactions):
    out = pp.clean(raw_transactions)
    assert "TotalAmount" in out.columns
    assert (out["TotalAmount"] == out["Quantity"] * out["Price"]).all()


def test_features_use_only_pre_cutoff_rows(raw_transactions):
    clean = pp.clean(raw_transactions)
    feats = pp.make_classification_features(clean)
    # Recency under a pre-cutoff snapshot is never negative.
    assert (feats["Recency"] >= 0).all()
    # Customer 3 (post-cutoff only) must not appear in the pre-cutoff features.
    assert 17852 not in feats["CustomerID"].values


def test_feature_and_label_customer_sets_match(raw_transactions):
    clean = pp.clean(raw_transactions)
    feats = pp.make_classification_features(clean)
    labels = pp.label_churn(clean)
    assert set(feats["CustomerID"]) == set(labels["CustomerID"])


def test_churn_label_logic(raw_transactions):
    clean = pp.clean(raw_transactions)
    labels = pp.label_churn(clean).set_index("CustomerID")["churn"].to_dict()
    assert labels[17850] == 0   # active before and after cutoff
    assert labels[17851] == 1   # active before only
