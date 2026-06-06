"""Clustering algorithms and validation indices for customer segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .utils import SEED


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    extra: Dict | None = None


def _safe_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Compute the three internal validation indices, ignoring noise points (-1)."""
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan"), float("nan"), float("nan")
    Xm, lm = X[mask], labels[mask]
    return (
        float(silhouette_score(Xm, lm)),
        float(davies_bouldin_score(Xm, lm)),
        float(calinski_harabasz_score(Xm, lm)),
    )


def elbow_silhouette_search(
    X: np.ndarray, k_range: range = range(2, 11), random_state: int = SEED
) -> pd.DataFrame:
    """Sweep K-means k values; return inertia and silhouette per k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def run_kmeans(X: np.ndarray, k: int, random_state: int = SEED) -> ClusterResult:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    sil, db, ch = _safe_metrics(X, labels)
    return ClusterResult("KMeans", labels, k, sil, db, ch,
                         extra={"inertia": float(km.inertia_), "centers": km.cluster_centers_})


def k_distance_plot_values(X: np.ndarray, k: int = 4) -> np.ndarray:
    """Return sorted k-th nearest-neighbor distances for DBSCAN eps tuning."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nbrs.kneighbors(X)
    return np.sort(dist[:, k - 1])


def run_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> ClusterResult:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    sil, dbi, ch = _safe_metrics(X, labels)
    n_noise = int((labels == -1).sum())
    return ClusterResult("DBSCAN", labels, n, sil, dbi, ch,
                         extra={"eps": eps, "min_samples": min_samples, "n_noise": n_noise})


def run_agnes(X: np.ndarray, k: int, linkage: str = "ward") -> ClusterResult:
    ag = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = ag.fit_predict(X)
    sil, dbi, ch = _safe_metrics(X, labels)
    return ClusterResult("AGNES", labels, k, sil, dbi, ch, extra={"linkage": linkage})


def results_table(results: List[ClusterResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r.name,
            "n_clusters": r.n_clusters,
            "Silhouette": r.silhouette,
            "Davies-Bouldin": r.davies_bouldin,
            "Calinski-Harabasz": r.calinski_harabasz,
            "Notes": str(r.extra) if r.extra else "",
        })
    return pd.DataFrame(rows)


def cluster_stability(
    X: np.ndarray,
    k: int,
    seeds: range | List[int] = range(10),
) -> pd.DataFrame:
    """Re-run K-means with many seeds and report how stable the solution is.

    Returns one row per seed (silhouette + sorted cluster sizes) plus a final
    ``SUMMARY`` row carrying the mean/std silhouette and the coefficient of
    variation of cluster sizes. Low silhouette std and low size CV => a stable,
    trustworthy segmentation rather than a seed-specific artefact.
    """
    rows = []
    sils = []
    size_cvs = []
    for s in seeds:
        km = KMeans(n_clusters=k, n_init=10, random_state=s)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        sizes = np.sort(np.bincount(labels, minlength=k))[::-1]
        size_cv = float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else float("nan")
        sils.append(sil)
        size_cvs.append(size_cv)
        rows.append({
            "seed": int(s),
            "silhouette": round(sil, 4),
            "size_cv": round(size_cv, 4),
            "cluster_sizes": ",".join(map(str, sizes.tolist())),
        })

    rows.append({
        "seed": "SUMMARY",
        "silhouette": round(float(np.mean(sils)), 4),
        "size_cv": round(float(np.mean(size_cvs)), 4),
        "cluster_sizes": f"silhouette_std={np.std(sils):.4f}",
    })
    return pd.DataFrame(rows)


def profile_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean RFM + size — for interpretation."""
    df = rfm.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").agg(
        n_customers=("CustomerID", "count"),
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
    ).round(2)
    profile["share_%"] = (profile["n_customers"] / profile["n_customers"].sum() * 100).round(1)
    return profile.reset_index()
