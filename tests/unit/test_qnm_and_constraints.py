from pathlib import Path

from physics_ai.physics_rules import evaluate_constraints
from physics_ai.qnm import Method, QNMRunConfig, compare_methods, solve_qnm
from physics_ai.theory import load_theory


def test_qnm_methods_are_close() -> None:
    config = QNMRunConfig(l=2, n=0, mass=1.0, method=Method.SHOOTING)
    results = [
        solve_qnm(config, run_id="test_qnm", method=Method.SHOOTING),
        solve_qnm(config, run_id="test_qnm", method=Method.WKB),
        solve_qnm(config, run_id="test_qnm", method=Method.SPECTRAL),
    ]
    report = compare_methods(results, tolerance=0.05)
    assert report.is_consistent


def test_constraint_filter_flags_wrong_sign_kinetic() -> None:
    theory = load_theory(Path("examples/theories/einstein_hilbert.yaml"))
    theory.parameters["kinetic_coeff"] = -1.0
    report = evaluate_constraints("test_constraints", theory, qnm_results=[])
    assert not report.is_valid
    assert report.wrong_sign_kinetic
