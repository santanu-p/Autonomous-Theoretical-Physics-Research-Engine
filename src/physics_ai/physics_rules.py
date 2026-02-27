from __future__ import annotations

from typing import Iterable

from physics_ai.types import ConstraintReport, QNMResult, TheorySpec


def _get_parameter(theory: TheorySpec, *names: str, default: float = 1.0) -> float:
    for name in names:
        if name in theory.parameters:
            return float(theory.parameters[name])
    return default


def evaluate_constraints(
    run_id: str,
    theory: TheorySpec,
    qnm_results: Iterable[QNMResult] | None = None,
) -> ConstraintReport:
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    kinetic = _get_parameter(theory, "kinetic_coeff", "zeta", default=1.0)
    ghost_instability = kinetic <= 0.0
    if ghost_instability:
        reasons.append("Ghost instability detected: kinetic coefficient <= 0")
    metrics["kinetic_coeff"] = kinetic

    wrong_sign_kinetic = kinetic < 0.0
    if wrong_sign_kinetic:
        reasons.append("Wrong-sign kinetic term detected.")

    alpha = _get_parameter(theory, "alpha", default=0.0)
    beta = _get_parameter(theory, "beta", default=0.0)
    divergent_background = abs(alpha) > 100 or abs(beta) > 100
    if divergent_background:
        reasons.append("Divergence filter triggered by large coupling magnitude.")
    metrics["alpha"] = alpha
    metrics["beta"] = beta

    if qnm_results:
        max_growth = max((result.omega_imag for result in qnm_results), default=-1.0)
        metrics["max_omega_imag"] = max_growth
        if max_growth > 0:
            reasons.append("Unstable mode detected: positive Im(omega).")

    is_valid = len(reasons) == 0
    return ConstraintReport(
        run_id=run_id,
        is_valid=is_valid,
        ghost_instability=ghost_instability,
        wrong_sign_kinetic=wrong_sign_kinetic,
        divergent_background=divergent_background,
        reasons=reasons,
        metrics=metrics,
    )
