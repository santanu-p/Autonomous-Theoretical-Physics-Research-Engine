from pathlib import Path

from physics_ai.symbolic import derive_field_equations, verify_fr_baseline, verify_gr_baseline
from physics_ai.theory import load_theory


def test_einstein_hilbert_derivation_contains_gr_equation() -> None:
    theory = load_theory(Path("examples/theories/einstein_hilbert.yaml"))
    artifact = derive_field_equations(theory, run_id="test_gr")
    assert verify_gr_baseline(artifact)
    assert artifact.metadata["family"] == "GR"


def test_fr_derivation_contains_fr_signature() -> None:
    theory = load_theory(Path("examples/theories/f_r.yaml"))
    artifact = derive_field_equations(theory, run_id="test_fr")
    assert verify_fr_baseline(artifact)
    assert artifact.metadata["family"] == "f(R)"
