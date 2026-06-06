"""Render `overleaf/main.tex` from a tokenised template + the metrics snapshot.

The template carries `{{token}}` placeholders for every metric that could go
stale (classification numbers). The 7-row classification table is generated
programmatically from `classical_results.csv` + `mlp_vs_classical.csv` so it can
never drift from the artifacts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .submission_snapshot import SubmissionSnapshot, load_submission_snapshot

TOKEN_RE = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")

DISPLAY_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest",
    "KNN": "KNN",
    "SVM_RBF": "SVM (RBF)",
    "GaussianNB": "Gaussian NB",
    "MLP": "MLP (PyTorch)",
}


def render_text_template(template: str, tokens: dict[str, str]) -> str:
    """Substitute every ``{{token}}`` and fail loudly on any unresolved token."""
    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in tokens:
            raise KeyError(f"Template token has no value: {key}")
        return tokens[key]

    rendered = TOKEN_RE.sub(replace, template)
    unresolved = TOKEN_RE.findall(rendered)
    if unresolved:
        raise KeyError(f"Unresolved template tokens: {sorted(set(unresolved))}")
    return rendered


def classification_table_body(root: str | Path | None = None) -> str:
    """Build the LaTeX body rows of the churn-classifier table from artifacts.

    Returns the six classical models (sorted by AUC) plus the MLP, with the best
    AUC and best F1 cells bolded. No leading/trailing newlines so it drops into
    a ``tabular`` between ``\\midrule`` and ``\\bottomrule``.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    classical = pd.read_csv(root / "reports" / "classical_results.csv").sort_values(
        "AUC", ascending=False)
    mlp = pd.read_csv(root / "reports" / "mlp_vs_classical.csv")
    mlp_row = mlp.loc[mlp["Model"].str.contains("MLP", case=False, na=False)].iloc[0]

    rows = []
    for _, r in classical.iterrows():
        rows.append({"Model": str(r["Model"]), "AUC": float(r["AUC"]), "F1": float(r["F1"]),
                     "Precision": float(r["Precision"]), "Recall": float(r["Recall"])})
    rows.append({"Model": "MLP", "AUC": float(mlp_row["AUC"]), "F1": float(mlp_row["F1"]),
                 "Precision": float(mlp_row["Precision"]), "Recall": float(mlp_row["Recall"])})

    best_auc = max(r["AUC"] for r in rows)
    best_f1 = max(r["F1"] for r in rows)

    def cell(value: float, is_best: bool) -> str:
        text = f"{value:.4f}"
        return f"\\textbf{{{text}}}" if is_best else text

    lines = []
    for r in rows:
        name = DISPLAY_NAMES.get(r["Model"], r["Model"])
        lines.append(
            f"{name} & {cell(r['AUC'], r['AUC'] == best_auc)} & "
            f"{cell(r['F1'], r['F1'] == best_f1)} & "
            f"{r['Precision']:.4f} & {r['Recall']:.4f} \\\\"
        )
    return "\n".join(lines)


def render_report(
    root: str | Path | None = None,
    snapshot: SubmissionSnapshot | None = None,
    template_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    snapshot = snapshot or load_submission_snapshot(root)
    template = Path(template_path) if template_path is not None else root / "overleaf" / "main.template.tex"
    output = Path(output_path) if output_path is not None else root / "overleaf" / "main.tex"

    tokens = snapshot.tokens()
    tokens["classification_table_body"] = classification_table_body(root)

    rendered = render_text_template(template.read_text(encoding="utf-8"), tokens)
    output.write_text(rendered, encoding="utf-8")
    return output


def main() -> int:
    out = render_report()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
