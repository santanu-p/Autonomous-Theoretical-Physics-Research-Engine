from __future__ import annotations

import sympy as sp

from physics_ai.types import BackgroundSystem, DerivationArtifact, TheorySpec


def static_spherical_ansatz() -> dict[str, str]:
    return {
        "metric": "ds^2 = -exp(2A(r)) dt^2 + exp(2B(r)) dr^2 + r^2(dtheta^2 + sin(theta)^2 dphi^2)",
        "functions": ["A(r)", "B(r)"],
    }


def reduce_to_background_odes(theory: TheorySpec, artifact: DerivationArtifact, run_id: str) -> BackgroundSystem:
    r = sp.Symbol("r", positive=True)
    A = sp.Function("A")
    B = sp.Function("B")
    rho = sp.Function("rho")
    pr = sp.Function("p_r")

    eq1 = sp.Eq(sp.diff(B(r), r), (1 - sp.exp(2 * B(r))) / (2 * r) + 4 * sp.pi * r * rho(r))
    eq2 = sp.Eq(sp.diff(A(r), r), (sp.exp(2 * B(r)) - 1) / (2 * r) + 4 * sp.pi * r * pr(r))

    family = artifact.metadata.get("family", "generic")
    if family == "f(R)":
        C = sp.Function("C")
        eq3 = sp.Eq(sp.diff(C(r), r, 2) + 2 / r * sp.diff(C(r), r), sp.Symbol("source_fr")(r))
        equations = [str(eq1), str(eq2), str(eq3)]
        unknowns = ["A(r)", "B(r)", "C(r)"]
    else:
        equations = [str(eq1), str(eq2)]
        unknowns = ["A(r)", "B(r)"]

    consistency = {
        "equation_count": len(equations),
        "unknown_count": len(unknowns),
        "square_system": len(equations) >= len(unknowns),
        "bianchi_like_redundancy": max(len(equations) - len(unknowns), 0),
    }
    return BackgroundSystem(
        run_id=run_id,
        theory_name=theory.name,
        ansatz="static_spherical",
        unknown_functions=unknowns,
        reduced_equations=equations,
        consistency=consistency,
    )


def is_consistent(background: BackgroundSystem) -> bool:
    return bool(background.consistency.get("square_system", False))
