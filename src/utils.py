"""Shared utilities: seeding, paths, logging, IO helpers."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

SEED = 42


def get_logger(name: str = "cintel", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single stream handler.

    Used instead of bare ``print`` in library code so output is timestamped,
    levelled, and silenceable (``get_logger().setLevel(logging.WARNING)``)
    without touching call sites. Idempotent — safe to call from every module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def seed_all(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str, dpi: int = 300) -> Path:
    """Save a matplotlib figure to reports/figures/<name>.png."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
