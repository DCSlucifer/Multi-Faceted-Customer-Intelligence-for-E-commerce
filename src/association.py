"""Market basket analysis: Apriori vs FP-Growth, per-cluster rule mining."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Rule-mining thresholds (single source of truth for notebooks + pipeline).
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.5
# Main basket analysis restricts to popular items, then mines at this support.
POPULAR_MIN_INVOICE_FRAC = 0.005   # keep items in >= 0.5% of invoices
POPULAR_MIN_INVOICES = 50
MAIN_MIN_SUPPORT = 0.02            # support used on the popularity-filtered matrix
# Per-segment mining needs a denser support and a minimum invoice count to be meaningful.
CLUSTER_MIN_SUPPORT = 0.02
CLUSTER_MIN_INVOICES = 50


def filter_popular_items(
    df: pd.DataFrame,
    item_col: str = "StockCode",
    min_invoice_frac: float = POPULAR_MIN_INVOICE_FRAC,
    min_invoices: int = POPULAR_MIN_INVOICES,
) -> Tuple[pd.DataFrame, set]:
    """Restrict to items appearing in at least ``min_invoice_frac`` of invoices.

    Keeps basket mining tractable and the resulting rules meaningful (rare items
    produce spurious high-lift rules). Returns ``(filtered_df, popular_codes)``.
    """
    n_invoices = df["Invoice"].nunique()
    threshold = max(min_invoices, int(min_invoice_frac * n_invoices))
    counts = df.groupby(item_col)["Invoice"].nunique()
    popular = set(counts[counts >= threshold].index)
    return df[df[item_col].isin(popular)], popular


def build_transactions(df: pd.DataFrame, item_col: str = "StockCode") -> List[List[str]]:
    """Group cleaned dataframe by Invoice -> list of items."""
    return (
        df.groupby("Invoice")[item_col]
          .apply(lambda s: list(s.unique()))
          .tolist()
    )


def encode_transactions(transactions: List[List[str]]) -> pd.DataFrame:
    """One-hot encode a list of transactions into a boolean DataFrame."""
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_, dtype=bool)


def mine_rules(
    df_encoded: pd.DataFrame,
    algorithm: str = "fpgrowth",
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Mine frequent itemsets and association rules.
    Returns (frequent_itemsets, rules, runtime_seconds).
    """
    if algorithm == "apriori":
        miner = apriori
    elif algorithm == "fpgrowth":
        miner = fpgrowth
    else:
        raise ValueError("algorithm must be 'apriori' or 'fpgrowth'")

    t0 = time.perf_counter()
    freq = miner(df_encoded, min_support=min_support, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame(), time.perf_counter() - t0

    rules = association_rules(
        freq,
        num_itemsets=len(df_encoded),
        metric="confidence",
        min_threshold=min_confidence,
    )
    rules = rules[rules["lift"] >= min_lift].reset_index(drop=True)
    rt = time.perf_counter() - t0
    return freq, rules, rt


def add_descriptions(
    rules: pd.DataFrame, code_to_desc: Dict[str, str]
) -> pd.DataFrame:
    """Attach human-readable descriptions to the antecedent/consequent codes."""
    def _names(items):
        return ", ".join(
            sorted({code_to_desc.get(c, str(c))[:40] for c in items})
        )

    out = rules.copy()
    out["antecedents_desc"] = out["antecedents"].apply(_names)
    out["consequents_desc"] = out["consequents"].apply(_names)
    return out


def prune_redundant(rules: pd.DataFrame) -> pd.DataFrame:
    """For rules with identical consequents, keep the one with highest lift.

    A coarse but practical pruning step for presentation tables.
    """
    if rules.empty:
        return rules
    out = rules.copy()
    out["_csig"] = out["consequents"].apply(lambda s: tuple(sorted(s)))
    out = (
        out.sort_values("lift", ascending=False)
           .drop_duplicates(subset="_csig", keep="first")
           .drop(columns="_csig")
           .reset_index(drop=True)
    )
    return out


def cluster_specific_rules(
    df: pd.DataFrame,
    customer_to_cluster: Dict[int, int],
    code_to_desc: Dict[str, str] | None = None,
    min_support: float = CLUSTER_MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    min_invoices: int = CLUSTER_MIN_INVOICES,
    top_n: int = 10,
) -> Dict[int, pd.DataFrame]:
    """For each customer cluster, mine rules over only that segment's invoices.

    Every cluster present in ``customer_to_cluster`` appears as a key — clusters
    that are too sparse to mine are returned as an empty DataFrame whose
    ``.attrs['skip_reason']`` explains why (rather than silently disappearing).
    """
    df = df.copy()
    df["cluster"] = df["CustomerID"].map(customer_to_cluster)
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)

    def _empty(reason: str) -> pd.DataFrame:
        empty = pd.DataFrame()
        empty.attrs["skip_reason"] = reason
        return empty

    out: Dict[int, pd.DataFrame] = {}
    for c, sub in df.groupby("cluster"):
        n_inv = sub["Invoice"].nunique()
        if n_inv < min_invoices:
            out[c] = _empty(f"only {n_inv} invoices (< {min_invoices})")
            continue
        txs = build_transactions(sub)
        enc = encode_transactions(txs)
        _, rules, _ = mine_rules(enc, "fpgrowth", min_support, min_confidence, min_lift)
        if rules.empty:
            out[c] = _empty(f"no rules at support>={min_support}, lift>={min_lift}")
            continue
        rules = prune_redundant(rules).sort_values("lift", ascending=False).head(top_n)
        if code_to_desc is not None:
            rules = add_descriptions(rules, code_to_desc)
        out[c] = rules.reset_index(drop=True)
    return out
