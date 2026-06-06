"""Association-rule mining tests."""
from __future__ import annotations

import pandas as pd
import pytest

from src import association as asso


def test_mine_rules_finds_expected_pattern(transaction_lists):
    enc = asso.encode_transactions(transaction_lists)
    freq, rules, runtime = asso.mine_rules(enc, "fpgrowth", min_support=0.2,
                                           min_confidence=0.5, min_lift=1.0)
    assert not freq.empty
    assert runtime >= 0
    # A->B should be discoverable (A and B co-occur in 6/8 baskets).
    assert not rules.empty


def test_mine_rules_empty_when_support_too_high(transaction_lists):
    enc = asso.encode_transactions(transaction_lists)
    freq, rules, _ = asso.mine_rules(enc, "fpgrowth", min_support=0.99)
    assert freq.empty
    assert rules.empty  # returns safely, no crash


def test_invalid_algorithm_raises(transaction_lists):
    enc = asso.encode_transactions(transaction_lists)
    with pytest.raises(ValueError):
        asso.mine_rules(enc, "not_an_algo")


def test_filter_popular_items():
    df = pd.DataFrame({
        "Invoice": ["1", "1", "2", "3", "4"],
        "StockCode": ["A", "B", "A", "A", "B"],
    })
    # A is in 3 invoices, B in 2. Threshold 3 -> only A is "popular".
    filtered, popular = asso.filter_popular_items(df, min_invoice_frac=0.0, min_invoices=3)
    assert popular == {"A"}
    assert set(filtered["StockCode"]) == {"A"}


def test_cluster_specific_rules_reports_skips():
    df = pd.DataFrame({
        "Invoice": ["1", "1", "2", "2"],
        "StockCode": ["A", "B", "A", "B"],
        "CustomerID": [1, 1, 2, 2],
    })
    # both customers in cluster 0; far fewer than min_invoices -> skipped, not dropped
    out = asso.cluster_specific_rules(df, {1: 0, 2: 0}, min_invoices=50)
    assert 0 in out                         # cluster key preserved
    assert out[0].empty
    assert "skip_reason" in out[0].attrs
