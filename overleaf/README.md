# Overleaf — IEEE Report

How to use this folder:

1. Create a new Overleaf project from the IEEE Conference template:
   https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn
2. Upload `main.tex` and `references.bib` from this folder.
3. Create a `figures/` folder in the Overleaf project and upload the PNGs from `reports/figures/` (only the ones referenced in `main.tex` are required; you can add more as you go).
4. Search the source for `TODO` and fill in numbers / paragraphs after each notebook runs.

**Page budget:** keep total at 8–10 pages. Drop low-value figures rather than the cross-task synthesis figures (notebook 06).

Compile with `pdflatex` (twice + bibtex). Final PDF should land in `reports/ieee_report.pdf`.
