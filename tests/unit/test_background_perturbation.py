from pathlib import Path

from physics_ai.background import reduce_to_background_odes
from physics_ai.perturbation import derive_master_equation
from physics_ai.symbolic import derive_field_equations
from physics_ai.theory import load_theory


def test_background_reduction_square_system() -> None:
    theory = load_theory(Path("examples/theories/einstein_hilbert.yaml"))
    derivation = derive_field_equations(theory, run_id="test_bg")
    background = reduce_to_background_odes(theory, derivation, run_id="test_bg")
    assert background.consistency["square_system"]
    assert len(background.reduced_equations) >= 2


def test_regge_wheeler_master_equation_shape() -> None:
    theory = load_theory(Path("examples/theories/einstein_hilbert.yaml"))
    derivation = derive_field_equations(theory, run_id="test_pt")
    background = reduce_to_background_odes(theory, derivation, run_id="test_pt")
    perturb = derive_master_equation(background, run_id="test_pt")
    assert "omega**2" in perturb.master_equation
