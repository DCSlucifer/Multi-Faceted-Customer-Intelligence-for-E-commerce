"""Tests guarding the submission package against stale/leaky metrics."""
from __future__ import annotations

from pathlib import Path

STALE = ("0.998", "0.990", "0.9981", "0.9900")


def test_build_slides_source_has_no_stale_leaky_metrics():
    text = Path("src/build_slides.py").read_text(encoding="utf-8")
    for stale in STALE:
        assert stale not in text, f"stale metric {stale} in build_slides.py"


def test_report_template_has_no_stale_leaky_metrics():
    text = Path("overleaf/main.template.tex").read_text(encoding="utf-8")
    for stale in STALE:
        assert stale not in text, f"stale metric {stale} in main.template.tex"


def test_validator_source_lists_report_and_overleaf_scan_scope():
    text = Path("scripts/validate_artifacts.py").read_text(encoding="utf-8")
    assert "overleaf/*.tex" in text
    assert "README.md" in text
    assert "STALE_METRICS" in text
