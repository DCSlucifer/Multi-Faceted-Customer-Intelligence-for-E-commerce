"""Tests for the LaTeX/Markdown template renderer."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.report_renderer import classification_table_body, render_text_template


def test_render_text_template_replaces_all_tokens():
    rendered = render_text_template(
        "AUC {{classification_best_auc}} model {{classification_best_model}}",
        {"classification_best_auc": "0.785", "classification_best_model": "LogisticRegression"},
    )
    assert rendered == "AUC 0.785 model LogisticRegression"


def test_render_text_template_rejects_unresolved_tokens():
    with pytest.raises(KeyError) as exc:
        render_text_template("AUC {{missing_token}}", {})
    assert "missing_token" in str(exc.value)


def test_classification_table_body_is_current_when_artifacts_exist():
    if not Path("reports/classical_results.csv").exists():
        pytest.skip("pipeline not run yet")
    body = classification_table_body()
    # 6 classical + MLP = 7 rows
    assert body.count(r"\\") == 7
    assert "MLP (PyTorch)" in body
    for stale in ("0.998", "0.9981", "0.990", "0.9900"):
        assert stale not in body
