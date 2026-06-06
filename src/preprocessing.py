"""Data loading, cleaning, RFM feature engineering, and churn labeling
for the Online Retail II dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .utils import DATA_PROCESSED, DATA_RAW, get_logger

logger = get_logger(__name__)

RAW_FILENAME = "online_retail_II.xlsx"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]
SNAPSHOT_DATE = pd.Timestamp("2011-12-10")           # one day after last invoice
CHURN_CUTOFF = pd.Timestamp("2011-09-01")            # train/score split for churn
CHURN_WINDOW_DAYS = 90                                # not seen in 90 days post cutoff


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load both sheets of online_retail_II.xlsx and concatenate."""
    path = path or (DATA_RAW / RAW_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from "
            "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
        )
    frames = [pd.read_excel(path, sheet_name=s) for s in SHEETS]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame, winsorize_q: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Apply standard cleaning steps:
       - drop null CustomerID
       - drop cancellation invoices (start with 'C')
       - drop non-positive Quantity / Price
       - winsorize Quantity & Price at the given quantiles
       - normalize types, add TotalAmount
    """
    out = df.copy()

    # CustomerID may be float; coerce to nullable Int64
    out["Customer ID"] = pd.to_numeric(out["Customer ID"], errors="coerce")
    out = out.dropna(subset=["Customer ID"])
    out["CustomerID"] = out["Customer ID"].astype("Int64")
    out = out.drop(columns=["Customer ID"])

    # Cancellation invoices
    out["Invoice"] = out["Invoice"].astype(str)
    out = out[~out["Invoice"].str.startswith("C")]

    # Non-positive
    out = out[(out["Quantity"] > 0) & (out["Price"] > 0)]

    # Winsorize
    lo_q, hi_q = winsorize_q
    for col in ["Quantity", "Price"]:
        lo, hi = out[col].quantile([lo_q, hi_q])
        out[col] = out[col].clip(lower=lo, upper=hi)

    # Types
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out = out.dropna(subset=["InvoiceDate"])
    out["StockCode"] = out["StockCode"].astype(str).str.strip()
    out["Description"] = out["Description"].astype(str).str.strip()
    out["Country"] = out["Country"].astype(str).str.strip()

    # Derived
    out["TotalAmount"] = out["Quantity"] * out["Price"]

    out = out.reset_index(drop=True)
    return out


def make_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per CustomerID."""
    grouped = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()
    return grouped


def make_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra behavioral features per CustomerID for the classifier."""
    by_inv = df.groupby(["CustomerID", "Invoice"]).agg(
        invoice_total=("TotalAmount", "sum"),
        items_in_invoice=("Quantity", "sum"),
    ).reset_index()

    behav = by_inv.groupby("CustomerID").agg(
        avg_basket_value=("invoice_total", "mean"),
        avg_basket_size=("items_in_invoice", "mean"),
    ).reset_index()

    variety = df.groupby("CustomerID").agg(
        unique_products=("StockCode", "nunique"),
    ).reset_index()

    # dominant country per customer
    dom_country = (
        df.groupby(["CustomerID", "Country"]).size()
          .reset_index(name="n")
          .sort_values(["CustomerID", "n"], ascending=[True, False])
          .drop_duplicates("CustomerID")
          .drop(columns="n")
          .rename(columns={"Country": "DominantCountry"})
    )

    feat = behav.merge(variety, on="CustomerID").merge(dom_country, on="CustomerID")
    return feat


def label_churn(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = CHURN_CUTOFF,
    window_days: int = CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Label each customer as churned (1) if they were active before `cutoff`
    AND have no purchases in the `window_days` after `cutoff`.

    Returns a DataFrame with columns [CustomerID, churn].
    """
    pre = df[df["InvoiceDate"] < cutoff]
    post = df[(df["InvoiceDate"] >= cutoff)
              & (df["InvoiceDate"] < cutoff + pd.Timedelta(days=window_days))]

    customers = pre["CustomerID"].unique()
    active_post = set(post["CustomerID"].unique())

    labels = pd.DataFrame({"CustomerID": customers})
    labels["churn"] = (~labels["CustomerID"].isin(active_post)).astype(int)
    return labels


def make_classification_features(
    df: pd.DataFrame,
    cutoff: pd.Timestamp = CHURN_CUTOFF,
) -> pd.DataFrame:
    """RFM + behavioral features computed using ONLY transactions strictly
    before `cutoff`, with snapshot_date = cutoff.

    Pairs with `label_churn(df, cutoff)`: both consume the same `pre` slice,
    so the customer sets match and no future information leaks into features.
    """
    pre = df[df["InvoiceDate"] < cutoff]
    if pre.empty:
        raise ValueError(f"No transactions strictly before cutoff={cutoff!r}")

    rfm = make_rfm(pre, snapshot_date=cutoff)
    behav = make_behavioral_features(pre)
    feats = rfm.merge(behav, on="CustomerID")

    assert feats["Recency"].min() >= 0, "Recency must be non-negative under pre-cutoff snapshot"
    return feats


def build_processed_artifacts(
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline. Saves parquet/csv outputs and returns
    ``(clean_df, rfm, churn, cls_feats)``.

    Writes two distinct feature files:
      - rfm_features.csv: full-history RFM (snapshot=SNAPSHOT_DATE) for clustering / EDA.
      - classification_features.csv: pre-cutoff RFM+behavioral for the churn classifier.
        Use this with churn_labels.csv to avoid target leakage.
    """
    raw = load_raw()
    logger.info("Raw rows: %s", f"{len(raw):,}")
    clean_df = clean(raw)
    logger.info("After cleaning: %s", f"{len(clean_df):,}")

    rfm = make_rfm(clean_df)
    behav = make_behavioral_features(clean_df)
    rfm = rfm.merge(behav, on="CustomerID")

    churn = label_churn(clean_df)
    cls_feats = make_classification_features(clean_df)
    logger.info("Customers: %s  |  Churn rate: %.2f%%", f"{len(rfm):,}", churn["churn"].mean() * 100)
    logger.info("Classification feature set (pre-cutoff): %s customers", f"{len(cls_feats):,}")

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        clean_df.to_parquet(DATA_PROCESSED / "transactions_clean.parquet", index=False)
        rfm.to_csv(DATA_PROCESSED / "rfm_features.csv", index=False)
        churn.to_csv(DATA_PROCESSED / "churn_labels.csv", index=False)
        cls_feats.to_csv(DATA_PROCESSED / "classification_features.csv", index=False)
        logger.info("Saved processed artifacts to %s", DATA_PROCESSED)

    return clean_df, rfm, churn, cls_feats


if __name__ == "__main__":
    build_processed_artifacts()
