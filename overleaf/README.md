# Overleaf — IEEE Report

This folder is a **self-contained** Overleaf project: `main.tex`, `references.bib`,
and a `figures/` folder with every PNG referenced by the report.

## How to compile

**Option A — Overleaf Workshop (VS Code):** open this `overleaf/` folder, log in to
Overleaf via the extension, and push/sync. It compiles as-is.

**Option B — overleaf.com:** create a project from the IEEE Conference template
(https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn),
then upload `main.tex`, `references.bib`, and the whole `figures/` folder.

Compile with `pdflatex` → `bibtex` → `pdflatex` ×2 (so citations and cross-refs
resolve). Export the result to `reports/ieee_report.pdf`.

## What's inside

- **12 figures**: architecture (Fig. 1), monthly revenue, K-means sweep, PCA/t-SNE,
  ROC, calibration, association network, Apriori-vs-FP-Growth, SHAP, autoencoder,
  churn-by-segment, segment×product heatmap.
- **8 tables**: dataset stats, cleaning breakdown, clustering validation, classifier
  leaderboard, honest baselines, feature-group ablation, top association rules,
  churn/value per segment.
- **16 references**, all cited in the text (no undefined `\cite`s).

## Important notes

- **Do not hand-edit metric numbers.** `main.tex` is GENERATED from
  `main.template.tex` by `python -m src.report_renderer`. Edit prose in the
  template, then re-render. The numbers come from the pipeline snapshot, so they
  can never drift.
- **If you re-run the pipeline** (figures change), re-copy them:
  `cp reports/figures/*.png overleaf/figures/` (or just the referenced ones).
- **Page budget (8–10 pages).** With all 12 figures + 8 tables the draft runs
  long. To trim without losing the story, comment out the lowest-value floats
  first — `fig:sweep` (Fig. 3), `fig:assocrt` (Apriori-vs-FP-Growth), and
  `fig:ae` (autoencoder) — and **keep** the cross-task synthesis figures
  (`fig:synth`, `fig:heatmap`) and the honest-evaluation tables, which carry the
  paper's contribution.
