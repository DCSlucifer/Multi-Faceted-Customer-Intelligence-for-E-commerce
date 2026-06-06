"""One-command, staged pipeline runner for the Customer Intelligence project.

Each stage reads input artifacts and writes output artifacts; missing inputs
raise a clear error pointing at the stage that produces them. Data artifacts
(CSV/parquet/joblib/manifest) are regenerated here so they trace back to code;
the notebooks own the *figures* and narrative.

Usage
-----
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage preprocess
    python scripts/run_pipeline.py --stage classification
    python scripts/run_pipeline.py --stage clustering
    python scripts/run_pipeline.py --stage association
    python scripts/run_pipeline.py --stage deep-learning
    python scripts/run_pipeline.py --stage cross-analysis
    python scripts/run_pipeline.py --stage slides
    python scripts/run_pipeline.py --stage validate
    python scripts/run_pipeline.py --stage all --force   # rebuild even if present
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import artifacts as A  # noqa: E402
from src.utils import MODELS_DIR, REPORTS_DIR, get_logger, seed_all  # noqa: E402

log = get_logger("pipeline")
KMEANS_K = 2  # chosen from elbow/silhouette in notebook 02 (silhouette 0.42)


def _need(key: str, producing_stage: str) -> Path:
    """Return an artifact path, or raise a clear error naming the stage to run."""
    path = A.ARTIFACTS[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Required input '{key}' not found at {path}.\n"
            f"  -> run:  python scripts/run_pipeline.py --stage {producing_stage}"
        )
    return path


def _exists(*keys: str) -> bool:
    return all(A.ARTIFACTS[k].exists() for k in keys)


# --------------------------------------------------------------------------- #
# Stages                                                                      #
# --------------------------------------------------------------------------- #
def stage_preprocess(force: bool) -> None:
    if not force and _exists("transactions_clean", "rfm_features",
                             "churn_labels", "classification_features"):
        log.info("[preprocess] artifacts present — skip (use --force to rebuild)")
        return
    from src import preprocessing as pp

    clean_df, rfm, churn, cls_feats = pp.build_processed_artifacts(save=True)
    A.update_manifest({
        "n_clean_rows": int(len(clean_df)),
        "n_customers_total": int(len(rfm)),
        "n_customers_labelled": int(len(churn)),
        "churn_rate": round(float(churn["churn"].mean()), 4),
        "n_classification_customers": int(len(cls_feats)),
    })
    log.info("[preprocess] done")


def stage_classification(force: bool) -> None:
    if not force and _exists("classical_results", "mlp_vs_classical", "splits") \
            and A.list_models():
        log.info("[classification] artifacts present — skip (use --force to rebuild)")
        return
    from src import classification as cls
    from src.evaluation import run_ablation, run_baselines, run_calibration
    from src.features import build_supervised_frame

    seed_all()
    feats = pd.read_csv(_need("classification_features", "preprocess"))
    churn = pd.read_csv(_need("churn_labels", "preprocess"))
    frame = build_supervised_frame(feats, churn)
    log.info(f"[classification] supervised frame: {frame.shape}, churn={frame['churn'].mean():.2%}")

    outcomes, table, splits, meta = cls.train_classical(frame)
    table.to_csv(A.ARTIFACTS["classical_results"], index=False)
    cls.save_models(outcomes)
    joblib.dump(splits, A.ARTIFACTS["splits"])

    # Honest baselines + ablation + calibration
    run_baselines(frame)
    run_ablation(frame)
    best = max(outcomes, key=lambda o: o.test_metrics["AUC"])
    run_calibration(best.estimator, splits["X_test_raw"], splits["y_test"],
                    model_name=best.name)

    # MLP comparison vs best classical -> mlp_vs_classical.csv
    model, _, val_aucs, scaler = cls.train_mlp(
        splits["X_train_t"], splits["y_train"], splits["X_test_t"], splits["y_test"])
    mlp_eval = cls.mlp_evaluate(model, scaler, splits["X_test_t"], splits["y_test"])
    mlp_row = {"Model": "MLP", **{k: round(v, 4) for k, v in mlp_eval["metrics"].items()}}
    best_row = {"Model": best.name, **{k: round(v, 4) for k, v in best.test_metrics.items()}}
    pd.DataFrame([best_row, mlp_row]).to_csv(A.ARTIFACTS["mlp_vs_classical"], index=False)

    # Persist the trained MLP + scaler + metadata for reproducible reload.
    import json
    import torch
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), A.ARTIFACTS["mlp_model"])
    joblib.dump({"mlp_scaler": scaler}, A.ARTIFACTS["dl_scalers"])
    A.ARTIFACTS["mlp_metadata"].write_text(
        json.dumps(cls.mlp_metadata(meta, scaler=scaler), indent=2), encoding="utf-8")

    A.update_manifest({
        "classification_best_model": meta["best_model_name"],
        "classification_best_auc": round(meta["best_model_auc"], 4),
        "classification_mlp_auc": round(float(mlp_eval["metrics"]["AUC"]), 4),
        "n_customers_labelled": int(len(frame)),
        "feature_columns": meta["feature_columns"],
        "transformed_feature_names": meta["transformed_feature_names"],
    })
    log.info(f"[classification] best={meta['best_model_name']} "
          f"AUC={meta['best_model_auc']:.4f}  MLP AUC={mlp_eval['metrics']['AUC']:.4f}")


def stage_clustering(force: bool) -> None:
    if not force and _exists("segments_kmeans", "cluster_stability"):
        log.info("[clustering] artifacts present — skip (use --force to rebuild)")
        return
    from src import clustering as clu
    from src.features import build_clustering_matrix

    seed_all()
    rfm = pd.read_csv(_need("rfm_features", "preprocess"))
    X, _, _ = build_clustering_matrix(rfm)

    res = clu.run_kmeans(X, KMEANS_K)
    seg = rfm[["CustomerID"]].copy()
    seg["cluster"] = res.labels
    seg.to_csv(A.ARTIFACTS["segments_kmeans"], index=False)

    profile = clu.profile_clusters(rfm, res.labels)
    profile.to_csv(REPORTS_DIR / "cluster_profile_kmeans.csv", index=False)

    stab = clu.cluster_stability(X, KMEANS_K, seeds=range(10))
    stab.to_csv(A.ARTIFACTS["cluster_stability"], index=False)

    summary = stab.iloc[-1]
    A.update_manifest({
        "clustering_k": KMEANS_K,
        "clustering_silhouette": float(summary["silhouette"]),
    })
    log.info(f"[clustering] k={KMEANS_K}  silhouette={res.silhouette:.4f}  "
          f"(stable over {len(stab) - 1} seeds)")


def stage_association(force: bool) -> None:
    if not force and _exists("association_rules"):
        log.info("[association] artifacts present — skip (use --force to rebuild)")
        return
    from src import association as asso

    parquet = _need("transactions_clean", "preprocess")
    df = pd.read_parquet(parquet)
    code_to_desc = (df.dropna(subset=["Description"])
                      .drop_duplicates("StockCode")
                      .set_index("StockCode")["Description"].to_dict())

    # Mirror notebook 04: restrict to popular items, then mine at MAIN_MIN_SUPPORT.
    df_pop, popular = asso.filter_popular_items(df)
    txs = asso.build_transactions(df_pop)
    enc = asso.encode_transactions(txs)
    log.info(f"[association] {len(popular)} popular items, {len(txs):,} transactions")

    rows = []
    rules_keep = None
    for algo in ("apriori", "fpgrowth"):
        freq, rules, rt = asso.mine_rules(enc, algo, asso.MAIN_MIN_SUPPORT,
                                          asso.MIN_CONFIDENCE, asso.MIN_LIFT)
        rows.append({"Algorithm": "Apriori" if algo == "apriori" else "FP-Growth",
                     "Freq itemsets": int(len(freq)),
                     "Rules (lift>=1.5)": int(len(rules)),
                     "Runtime (s)": round(rt, 3)})
        if algo == "fpgrowth":
            rules_keep = rules
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "apriori_vs_fpgrowth.csv", index=False)

    top = asso.prune_redundant(rules_keep).sort_values("lift", ascending=False).head(30)
    top = asso.add_descriptions(top, code_to_desc)
    top.to_csv(A.ARTIFACTS["association_rules"], index=False)

    A.update_manifest({"association_n_rules": int(len(rules_keep)),
                       "association_min_support": asso.MAIN_MIN_SUPPORT})
    log.info(f"[association] {len(rules_keep)} rules (lift>={asso.MIN_LIFT}); "
          f"top-{len(top)} saved")


def stage_deep_learning(force: bool) -> None:
    """Autoencoder anomaly artifact (MLP comparison lives in classification)."""
    out = REPORTS_DIR / "anomaly_customers.csv"
    if not force and out.exists():
        log.info("[deep-learning] artifacts present — skip (use --force to rebuild)")
        return
    from src import classification as cls
    from src.features import NUMERIC_FEATURES

    seed_all()
    feats = pd.read_csv(_need("classification_features", "preprocess"))
    X = feats[NUMERIC_FEATURES]
    ae_model, losses, ae_scaler, err = cls.train_autoencoder(X)

    # Persist the trained autoencoder + scaler (merge into the shared DL scalers).
    import torch
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ae_model.state_dict(), A.ARTIFACTS["autoencoder_model"])
    dl_scalers = joblib.load(A.ARTIFACTS["dl_scalers"]) if A.ARTIFACTS["dl_scalers"].exists() else {}
    dl_scalers["ae_scaler"] = ae_scaler
    joblib.dump(dl_scalers, A.ARTIFACTS["dl_scalers"])

    feats = feats.assign(reconstruction_error=err)
    thresh = pd.Series(err).quantile(0.95)
    anomalies = (feats[feats["reconstruction_error"] >= thresh]
                 .sort_values("reconstruction_error", ascending=False))
    anomalies.to_csv(out, index=False)
    A.update_manifest({"autoencoder_final_loss": round(float(losses[-1]), 5),
                       "n_anomalies_top5pct": int(len(anomalies))})
    log.info(f"[deep-learning] autoencoder loss={losses[-1]:.5f}; "
          f"flagged {len(anomalies)} anomalies (top 5%)")


def stage_cross_analysis(force: bool) -> None:
    out = REPORTS_DIR / "churn_by_cluster.csv"
    if not force and out.exists():
        log.info("[cross-analysis] artifacts present — skip (use --force to rebuild)")
        return
    seg = pd.read_csv(_need("segments_kmeans", "clustering"))
    churn = pd.read_csv(_need("churn_labels", "preprocess"))
    merged = seg.merge(churn, on="CustomerID", how="inner")
    by_cluster = (merged.groupby("cluster")["churn"]
                  .agg(n_customers="count", churn_rate="mean").reset_index())
    by_cluster["churn_rate"] = by_cluster["churn_rate"].round(4)
    by_cluster.to_csv(out, index=False)
    gap = by_cluster["churn_rate"].max() - by_cluster["churn_rate"].min()
    A.update_manifest({"churn_gap_between_segments": round(float(gap), 4)})
    log.info(f"[cross-analysis] churn-by-segment saved; gap={gap:.1%}")


def stage_decision_layer(force: bool) -> None:
    out = REPORTS_DIR / "customer_decision_table.csv"
    if not force and out.exists():
        log.info("[decision-layer] artifacts present — skip (use --force to rebuild)")
        return
    from src import decision_layer as dl

    # Fail loudly if any upstream input is missing.
    _need("classification_features", "preprocess")
    _need("churn_labels", "preprocess")
    _need("segments_kmeans", "clustering")
    _need("association_rules", "association")
    for rel, stage in (("churn_by_cluster.csv", "cross-analysis"),
                       ("anomaly_customers.csv", "deep-learning")):
        if not (REPORTS_DIR / rel).exists():
            raise FileNotFoundError(
                f"Required input '{rel}' not found.\n"
                f"  -> run:  python scripts/run_pipeline.py --stage {stage}")

    meta = dl.run()
    A.update_manifest(meta)
    log.info("[decision-layer] %d customers; %d retention / %d cross-sell / %d review",
             meta["decision_table_customers"], meta["decision_retention_targets"],
             meta["decision_cross_sell_targets"], meta["decision_manual_review_targets"])


def stage_report(force: bool) -> None:
    log.info("[report] rendering overleaf/main.tex from snapshot ...")
    result = subprocess.run([sys.executable, "-m", "src.report_renderer"], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def stage_readme(force: bool) -> None:
    log.info("[readme] rendering README.md from snapshot ...")
    result = subprocess.run([sys.executable, "-m", "src.readme_renderer"], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def stage_slides(force: bool) -> None:
    log.info("[slides] building deck via `python -m src.build_slides` ...")
    result = subprocess.run([sys.executable, "-m", "src.build_slides"], cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError("slide build failed")


def stage_package(force: bool) -> None:
    log.info("[package] building full submission package ...")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_submission_package.py")],
                            cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def stage_validate(force: bool) -> None:
    log.info("[validate] running scripts/validate_artifacts.py ...")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_artifacts.py")],
                            cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


STAGES = {
    "preprocess": stage_preprocess,
    "clustering": stage_clustering,
    "classification": stage_classification,
    "association": stage_association,
    "deep-learning": stage_deep_learning,
    "cross-analysis": stage_cross_analysis,
    "decision-layer": stage_decision_layer,
    "report": stage_report,
    "readme": stage_readme,
    "slides": stage_slides,
    "validate": stage_validate,
    "package": stage_package,
}

# `report`/`readme`/`slides` run after cross-analysis so the snapshot (manifest +
# CSVs) is fully populated; `validate` runs last. `package` is intentionally not
# part of `all` (it *invokes* `all`).
ALL_ORDER = ["preprocess", "clustering", "classification", "association",
             "deep-learning", "cross-analysis", "decision-layer",
             "report", "readme", "slides", "validate"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=["all", *STAGES.keys()],
                        help="pipeline stage to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if output artifacts already exist")
    args = parser.parse_args()

    if args.stage == "all":
        for name in ALL_ORDER:
            log.info("=== %s ===", name)
            STAGES[name](force=args.force)
    else:
        # An explicitly requested stage always rebuilds.
        STAGES[args.stage](force=True)

    log.info("Pipeline finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
