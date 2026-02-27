from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

from physics_ai.types import NoveltyNeighbor, NoveltyReport


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class LiteratureRecord:
    identifier: str
    title: str
    abstract: str
    equations: list[str]
    parameter_regime: dict[str, float]


def _token_set(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def jaccard_similarity(a: str, b: str) -> float:
    aset = _token_set(a)
    bset = _token_set(b)
    if not aset and not bset:
        return 1.0
    union = aset | bset
    if not union:
        return 0.0
    return len(aset & bset) / len(union)


def parameter_overlap_score(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    diffs = []
    for key in keys:
        denom = max(abs(a[key]), abs(b[key]), 1.0)
        diffs.append(abs(a[key] - b[key]) / denom)
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff)


def fetch_arxiv_records(query: str = "black hole modified gravity", max_results: int = 10) -> list[LiteratureRecord]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        text = response.text
    except Exception:
        return default_seed_records()

    # Lightweight parsing (entry chunks) to stay dependency-light.
    entries = text.split("<entry>")
    records: list[LiteratureRecord] = []
    for chunk in entries[1:]:
        raw_id = _extract_tag(chunk, "id")
        title = _extract_tag(chunk, "title").strip().replace("\n", " ")
        summary = _extract_tag(chunk, "summary").strip().replace("\n", " ")
        if not raw_id or not title:
            continue
        records.append(
            LiteratureRecord(
                identifier=raw_id,
                title=title,
                abstract=summary,
                equations=[],
                parameter_regime={},
            )
        )
    return records or default_seed_records()


def _extract_tag(xml_chunk: str, tag: str) -> str:
    start = xml_chunk.find(f"<{tag}>")
    end = xml_chunk.find(f"</{tag}>")
    if start == -1 or end == -1:
        return ""
    start += len(tag) + 2
    return xml_chunk[start:end]


def default_seed_records() -> list[LiteratureRecord]:
    return [
        LiteratureRecord(
            identifier="rw1957",
            title="Regge-Wheeler Stability of Schwarzschild",
            abstract="Axial perturbations and master equation for Schwarzschild black holes.",
            equations=["d2Psi/dr_*2 + (omega^2 - V_RW)Psi = 0"],
            parameter_regime={"mass": 1.0},
        ),
        LiteratureRecord(
            identifier="fr_review",
            title="f(R) Gravity and Black Hole Phenomenology",
            abstract="Higher-curvature corrections and quasinormal mode signatures.",
            equations=["f_R R_mn - 1/2 f g_mn + (...) = kappa T_mn"],
            parameter_regime={"alpha": 0.1, "mass": 1.0},
        ),
    ]


def score_novelty(
    run_id: str,
    hypothesis_text: str,
    equations: list[str],
    parameters: dict[str, float],
    corpus: list[LiteratureRecord] | None = None,
) -> NoveltyReport:
    corpus = corpus or default_seed_records()
    neighbors: list[NoveltyNeighbor] = []
    hypothesis_eq_blob = " ".join(equations)
    for record in corpus:
        concept = jaccard_similarity(hypothesis_text, f"{record.title} {record.abstract}")
        equation = jaccard_similarity(hypothesis_eq_blob, " ".join(record.equations))
        overlap = parameter_overlap_score(parameters, record.parameter_regime)
        composite = 0.5 * concept + 0.3 * equation + 0.2 * overlap
        neighbors.append(
            NoveltyNeighbor(
                identifier=record.identifier,
                title=record.title,
                concept_similarity=concept,
                equation_similarity=equation,
                parameter_overlap=overlap,
                composite_similarity=composite,
            )
        )
    neighbors = sorted(neighbors, key=lambda item: item.composite_similarity, reverse=True)
    nearest = neighbors[0] if neighbors else None
    novelty = 1.0 - (nearest.composite_similarity if nearest else 0.0)
    novelty = max(0.0, min(1.0, novelty))
    explanation = (
        f"Nearest prior work: {nearest.identifier if nearest else 'none'}; "
        f"composite similarity={nearest.composite_similarity:.3f}."
        if nearest
        else "No neighbors available."
    )
    return NoveltyReport(
        run_id=run_id,
        novelty_score=novelty,
        explanation=explanation,
        neighbors=neighbors[:5],
    )
