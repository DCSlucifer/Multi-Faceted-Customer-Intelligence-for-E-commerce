"""Shared fixtures: small synthetic datasets so the suite runs in seconds
without the 45 MB Excel file."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Raw-shaped frame (pre-``clean``) exercising every cleaning rule and
    straddling the churn cutoff 2011-09-01."""
    rows = [
        # customer 1: active before AND after cutoff  -> churn 0
        ("100", "85123A", "WHITE MUG", 6, "2011-08-01", 2.5, 17850, "United Kingdom"),
        ("101", "85123A", "WHITE MUG", 3, "2011-09-15", 2.5, 17850, "United Kingdom"),
        # customer 2: active before only             -> churn 1
        ("102", "71053", "RED LAMP", 4, "2011-07-01", 3.0, 17851, "France"),
        ("103", "22423", "PLATE", 2, "2011-08-20", 5.0, 17851, "France"),
        # customer 3: post-cutoff only -> excluded from labels/features
        ("104", "21730", "GLASS", 1, "2011-10-01", 4.0, 17852, "Germany"),
        # --- rows that must be dropped by clean() ---
        ("105", "21730", "GLASS", 1, "2011-08-01", 4.0, np.nan, "United Kingdom"),  # null id
        ("C900", "85123A", "WHITE MUG", -6, "2011-08-01", 2.5, 17850, "United Kingdom"),  # cancel
        ("106", "21730", "GLASS", 0, "2011-08-01", 4.0, 17851, "France"),  # qty 0
        ("107", "21730", "GLASS", 5, "2011-08-01", 0.0, 17851, "France"),  # price 0
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def supervised_frame() -> pd.DataFrame:
    """Synthetic, learnable churn frame matching the real feature schema."""
    rng = np.random.default_rng(42)
    n = 240
    recency = rng.integers(0, 400, n)
    noise = rng.normal(0, 60, n)
    churn = ((recency + noise) > 200).astype(int)
    countries = rng.choice(
        ["United Kingdom", "France", "Germany", "Spain", "Rareland"],
        size=n, p=[0.7, 0.12, 0.1, 0.07, 0.01],
    )
    return pd.DataFrame({
        "CustomerID": np.arange(n),
        "Recency": recency,
        "Frequency": rng.integers(1, 30, n),
        "Monetary": rng.gamma(2.0, 500, n),
        "avg_basket_value": rng.gamma(2.0, 80, n),
        "avg_basket_size": rng.gamma(2.0, 20, n),
        "unique_products": rng.integers(1, 120, n),
        "DominantCountry": countries,
        "churn": churn,
    })


@pytest.fixture
def transaction_lists() -> list[list[str]]:
    """A handful of baskets with a clear A<->B co-occurrence pattern."""
    return [
        ["A", "B", "C"], ["A", "B"], ["A", "B", "D"], ["A", "B"],
        ["A", "B", "C"], ["C", "D"], ["A", "B", "E"], ["A", "B"],
    ]
