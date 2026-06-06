"""Shared matplotlib style for publication-quality figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
           "#8AB17D", "#6A4C93", "#1982C4"]


def apply_style() -> None:
    """Apply a consistent style across all notebooks/figures."""
    sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
    mpl.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    })
