from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from physics_ai.types import NoveltyReport, PaperBuildSpec


def _load_template(name: str) -> Template:
    base = Path(__file__).parent / "templates" / name
    return Template(base.read_text(encoding="utf-8"))


def _format_neighbors(report: NoveltyReport) -> str:
    lines = []
    for neighbor in report.neighbors:
        lines.append(
            f"- {neighbor.identifier}: {neighbor.title} "
            f"(sim={neighbor.composite_similarity:.3f})"
        )
    return "\n".join(lines) if lines else "- none"


def _build_context(
    spec: PaperBuildSpec,
    derivation: dict[str, Any],
    qnm_summary: dict[str, Any],
    novelty: NoveltyReport,
) -> dict[str, str]:
    return {
        "title": spec.title,
        "authors": ", ".join(spec.authors),
        "run_id": spec.run_id,
        "derivation_family": str(derivation.get("family", "unknown")),
        "equations": "\n".join(f"- `{eq}`" for eq in derivation.get("equations", [])),
        "qnm_summary": "\n".join(f"- {k}: {v}" for k, v in qnm_summary.items()),
        "novelty_score": f"{novelty.novelty_score:.3f}",
        "novelty_explanation": novelty.explanation,
        "neighbors": _format_neighbors(novelty),
    }


def build_paper(
    spec: PaperBuildSpec,
    derivation: dict[str, Any],
    qnm_summary: dict[str, Any],
    novelty: NoveltyReport,
) -> dict[str, Path]:
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _build_context(spec, derivation, qnm_summary, novelty)
    written: dict[str, Path] = {}

    if spec.include_latex:
        tex_tpl = _load_template("paper_tex.tpl")
        tex_path = out_dir / "paper.tex"
        tex_path.write_text(tex_tpl.substitute(context), encoding="utf-8")
        bib_path = out_dir / "refs.bib"
        bib_path.write_text(DEFAULT_BIB, encoding="utf-8")
        written["latex"] = tex_path
        written["bib"] = bib_path

    if spec.include_markdown:
        md_tpl = _load_template("paper_md.tpl")
        md_path = out_dir / "paper.md"
        md_path.write_text(md_tpl.substitute(context), encoding="utf-8")
        written["markdown"] = md_path

    return written


DEFAULT_BIB = """@article{regge_wheeler_1957,
  title = {Stability of a Schwarzschild singularity},
  author = {Regge, T. and Wheeler, J. A.},
  journal = {Physical Review},
  year = {1957}
}
"""
