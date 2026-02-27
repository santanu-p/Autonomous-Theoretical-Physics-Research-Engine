from pathlib import Path

from physics_ai.novelty import default_seed_records, score_novelty
from physics_ai.paper import build_paper
from physics_ai.types import PaperBuildSpec


def test_novelty_score_in_range() -> None:
    report = score_novelty(
        run_id="test_novelty",
        hypothesis_text="Modified gravity black-hole perturbations",
        equations=["d2Psi/dr_*2 + (omega^2 - V)Psi = 0"],
        parameters={"mass": 1.0, "alpha": 0.05},
        corpus=default_seed_records(),
    )
    assert 0.0 <= report.novelty_score <= 1.0
    assert report.neighbors


def test_paper_builder_writes_latex_and_markdown(tmp_path: Path) -> None:
    report = score_novelty(
        run_id="test_paper",
        hypothesis_text="Modified gravity black-hole perturbations",
        equations=["d2Psi/dr_*2 + (omega^2 - V)Psi = 0"],
        parameters={"mass": 1.0},
        corpus=default_seed_records(),
    )
    spec = PaperBuildSpec(
        run_id="test_paper",
        title="Test Report",
        authors=["Tester"],
        output_dir=str(tmp_path),
        include_latex=True,
        include_markdown=True,
    )
    outputs = build_paper(spec, {"family": "GR", "equations": ["Eq(G,T)"]}, {"k": 1}, report)
    assert "latex" in outputs
    assert "markdown" in outputs
    assert (tmp_path / "paper.tex").exists()
    assert (tmp_path / "paper.md").exists()
