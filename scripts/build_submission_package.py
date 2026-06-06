"""One command to produce a submission-ready bundle.

Rebuilds every pipeline artifact from scratch, re-renders the report
(`overleaf/main.tex`), README, and slide deck from the resulting metrics
snapshot, runs the full validator, and writes `reports/submission_manifest.json`
listing the files that make up the submitted package.

    python scripts/build_submission_package.py

Requires the raw dataset at `data/raw/online_retail_II.xlsx` (the full rebuild
re-runs preprocessing). PDF export of the report/slides remains a manual final
step (no LaTeX/PowerPoint toolchain is assumed here).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that constitute the submitted package (must exist after a clean build).
REQUIRED_FILES = [
    "README.md",
    "overleaf/main.tex",
    "reports/manifest.json",
    "reports/slides.pptx",
    "reports/classical_results.csv",
    "reports/mlp_vs_classical.csv",
    "reports/baseline_results.csv",
    "reports/ablation_results.csv",
    "reports/calibration_results.csv",
    "reports/cluster_stability.csv",
    "reports/association_rules.csv",
    "models/LogisticRegression.joblib",
    "models/RandomForest.joblib",
    "models/mlp.pt",
    "models/autoencoder.pt",
]


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    py = sys.executable
    # `--stage all` rebuilds artifacts, then renders report/readme/slides and
    # validates (report/readme/slides/validate are the trailing stages).
    run([py, "scripts/run_pipeline.py", "--stage", "all", "--force"])

    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        print(f"ERROR: required package files missing after build: {missing}", file=sys.stderr)
        return 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "required_files": REQUIRED_FILES,
        "optional_files": ["reports/slides.pdf", "reports/ieee_report.pdf"],
        "notes": "Export slides.pdf and ieee_report.pdf manually (no LaTeX/PowerPoint toolchain assumed).",
    }
    out = ROOT / "reports" / "submission_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("Submission package is ready (PDF export remains a manual step).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
