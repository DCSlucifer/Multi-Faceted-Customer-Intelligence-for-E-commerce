# Q&A Defense Sheet — Multi-Faceted Customer Intelligence (Team 4)

Anticipated questions for the oral defense, with answers grounded in the
**current, leakage-safe** artifacts (`reports/manifest.json`). Numbers here match
`classical_results.csv`, `baseline_results.csv`, `ablation_results.csv`,
`calibration_results.csv`, `cluster_stability.csv`, and `association_rules.csv`.

---

### 1. Your best AUC is "only" ~0.78 — isn't that weak for a churn model?
That 0.785 is the **honest** number *after* we removed target leakage. It is not
weak — it is realistic, and it clearly beats the baselines: majority-class **0.50**,
recency-only logistic regression **0.75**. An earlier version with full-history
features hit ~0.99, but that was leakage (the feature secretly encoded the label),
not skill. We chose to report the defensible number.

### 2. Isn't churn just "predicted" by Recency — the label *is* future inactivity?
There is a genuine structural link, and we address it head-on:
- Features are computed **strictly before** the cutoff (snapshot = cutoff `2011-09-01`); the label is "no purchase in the next 90 days."
- **Ablation:** removing Recency drops CV-AUC only from **0.798 → 0.762** — the model still works on behavioural + country features alone, so Recency is dominant but not the whole story.
- Probabilities are **well-calibrated** (Brier **0.189**), so the score is usable, not just a Recency proxy.

### 3. How exactly did you prevent target leakage?
Four concrete safeguards:
1. Supervised features built only from transactions **before** the cutoff (`make_classification_features`).
2. Encoding (log-scaling + rare-folded one-hot of country) lives **inside** the `imblearn` pipeline, so `GridSearchCV` refits it **per training fold** — the test rows never influence scaling or the country vocabulary.
3. **SMOTE** is a pipeline step applied **only at fit time** (training folds), never on the held-out 20% test set.
4. A sanity script (`scripts/rebuild_classification_features.py`) shows recency-only AUC falling from **0.97 (leaky, full-history)** to **0.76 (pre-cutoff)** — proof the fix is real.

### 4. Why only k=2 clusters — isn't that trivial?
k was chosen by a joint elbow + silhouette sweep over k=2..10; **k=2** wins on
silhouette (**0.419**), is confirmed by AGNES (0.388), and is **stable across 10
seeds** (silhouette std **0.000**). DBSCAN fails (silhouette **−0.12**, 13 fragments
+ 403 noise) because RFM is heavy-tailed. Two segments are not trivial — they are
*interpretable* (Dormant vs Active-loyal) and they drive the headline 67-point churn
gap. Forcing more clusters reduces separability and actionability.

### 5. The MLP doesn't beat the classical models — so why include deep learning?
This is an honest, defensible finding: on RFM + behavioural tabular features there is
a **performance ceiling** — MLP **0.783** ≈ best classical **0.785**. The DL extension
still earns its place: (a) it demonstrates we can build, tune (early stopping, SMOTE),
and explain a neural net; (b) the **autoencoder** solves a *different* task —
unsupervised anomaly detection (top 5% / 263 customers by reconstruction error);
(c) **SHAP on both** RF (TreeExplainer) and MLP (KernelExplainer) agree on the drivers.
To beat the ceiling you would need richer signals (purchase **sequences**), which we
list as future work.

### 6. Did you apply SMOTE correctly?
Yes. SMOTE is a step in the `imblearn` Pipeline placed **after** the preprocessor and
**before** the classifier. Because `GridSearchCV` clones and refits the whole pipeline
on each fold, SMOTE only ever sees training data; the validation/test folds are
evaluated on their **original** distribution. A unit test asserts the step order
(`pre → smote → clf`).

### 7. Apriori vs FP-Growth gave the same rules — why run both?
Identical output (**242 frequent itemsets, 17 rules** at min_support=0.02, lift≥1.5) is
the point: it's a **correctness cross-check**. The contribution is the **runtime**
comparison — FP-Growth (~4.0 s) is ~**1.4–1.5× faster** than Apriori (~5.6–6.1 s),
reproducing the textbook efficiency result; the gap widens at lower support.

### 8. What is the actual business takeaway — the "so what"?
The cross-task synthesis: a **67-point churn gap** (Dormant ~90% vs Active-loyal ~23%).
The loyal segment buys in **color-coordinated pairs** (e.g., pink→red lunchbag, lift
7.4; pink→red jumbo bag, 6.4; red→white heart t-light, 5.3). **Action:** target the
23% loyal-segment churners with color-pair bundles (high-ROI, since retention is far
cheaper than acquisition) and reserve broad win-back campaigns for the dormant
majority. No single model surfaces this — it needs segmentation + churn + baskets together.

### 9. How do we know the numbers are reproducible / trustworthy?
- Single seed (**42**) everywhere; pinned `requirements.txt`.
- One command: `python scripts/run_pipeline.py --stage all`.
- `reports/manifest.json` records seed, cutoff, row counts, metrics, and package versions.
- Report, slides, and README headline are **generated** from one metrics snapshot
  (`src/submission_snapshot.py`), so the document numbers **cannot drift** from the pipeline.
- **43 pytest** cases + a **23-check validator** + GitHub Actions CI.

### 10. Why calibration, and what did it show?
A retention team acts on **probabilities** (who to contact, at what risk threshold),
not just a ranking. We report the **Brier score (0.189)** and a reliability curve that
hugs the diagonal — meaning the predicted churn probabilities are trustworthy enough to
set a campaign threshold, not merely to rank-order customers.

### 11. How is the project architecture organized?
The architecture is documented in `docs/architecture.md` and as an editable FigJam
board linked from the README. The flow is:
Researcher/CI entrypoints -> pipeline CLI / submission builder / test runner ->
preprocessing -> clustering, churn modeling, market basket mining, deep learning,
and cross-task synthesis -> generated data, model, report, and manifest artifacts.
The README link is also stored in `README.template.md`, so re-rendering the README
does not drop the diagram reference.

---

### Bonus — likely preprocessing questions
- **Why drop ~25% of rows (null CustomerID)?** RFM, segmentation, and churn are all
  *per-customer*; anonymous rows can't be attributed, so they're unusable for these tasks.
- **Outliers?** Winsorise `Quantity`/`Price` at the 1%/99% quantiles to tame extreme
  bulk orders/returns without deleting data; also drop cancellation invoices (prefix `C`)
  and non-positive quantity/price as data-quality filters. Result: 1,067,371 → **805,549** rows.
- **Churn window choice (90 days)?** Long enough to distinguish genuine lapse from normal
  purchase gaps in a 2-year span, short enough to leave a post-cutoff window for labelling.
