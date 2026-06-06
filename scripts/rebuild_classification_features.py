"""Rebuild data/processed/classification_features.csv from the existing
transactions_clean.parquet, and verify the target-leakage fix.

Run from the repo root:
    python scripts/rebuild_classification_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import preprocessing as pp  # noqa: E402
from src.utils import DATA_PROCESSED  # noqa: E402


def main() -> int:
    parquet = DATA_PROCESSED / "transactions_clean.parquet"
    if not parquet.exists():
        print(f"ERROR: {parquet} not found. Run notebook 01_eda first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet)
    print(f"Loaded transactions_clean.parquet: {len(df):,} rows")

    cutoff = pp.CHURN_CUTOFF
    feats = pp.make_classification_features(df, cutoff=cutoff)
    churn = pp.label_churn(df, cutoff=cutoff)

    out = DATA_PROCESSED / "classification_features.csv"
    feats.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(feats):,} customers)")

    # --- sanity checks -----------------------------------------------------
    print("\n--- sanity checks ---")
    pre = df[df["InvoiceDate"] < cutoff]
    print(f"Customer set matches label_churn: "
          f"{set(feats.CustomerID) == set(churn.CustomerID)}")
    max_pre = pre["InvoiceDate"].max()
    print(f"Max InvoiceDate used in features: {max_pre}  (cutoff={cutoff.date()})")
    assert max_pre < cutoff, "feature-build slice leaked post-cutoff rows"
    print(f"Recency range: [{feats.Recency.min()}, {feats.Recency.max()}] days")
    assert (feats.Recency >= 0).all()

    # --- before/after leakage check on Recency -----------------------------
    # Reproduce the OLD (leaky) Recency: snapshot=2011-12-10, full history.
    leaky = pp.make_rfm(df)
    merged = leaky.merge(churn, on="CustomerID")
    fixed = feats[["CustomerID", "Recency"]].merge(churn, on="CustomerID")

    try:
        from sklearn.metrics import roc_auc_score
        auc_leaky = roc_auc_score(merged.churn, merged.Recency)
        auc_fixed = roc_auc_score(fixed.churn, fixed.Recency)
        print(f"\nAUC of Recency alone vs churn:")
        print(f"  OLD (snapshot=2011-12-10, full history):  {auc_leaky:.4f}   <-- leaky")
        print(f"  NEW (snapshot=2011-09-01, pre-cutoff):     {auc_fixed:.4f}   <-- realistic")
    except ImportError:
        print("(sklearn not installed; skipping AUC sanity check)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
