"""Render `README.md` headline metrics from the submission snapshot.

`README.template.md` is the source; the headline-results table carries
`{{token}}` placeholders that this renderer fills from the current artifacts.
"""
from __future__ import annotations

from pathlib import Path

from .report_renderer import render_text_template
from .submission_snapshot import SubmissionSnapshot, load_submission_snapshot


def render_readme(
    root: str | Path | None = None,
    snapshot: SubmissionSnapshot | None = None,
    template_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    snapshot = snapshot or load_submission_snapshot(root)
    template = Path(template_path) if template_path is not None else root / "README.template.md"
    output = Path(output_path) if output_path is not None else root / "README.md"
    rendered = render_text_template(template.read_text(encoding="utf-8"), snapshot.tokens())
    output.write_text(rendered, encoding="utf-8")
    return output


def main() -> int:
    out = render_readme()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
