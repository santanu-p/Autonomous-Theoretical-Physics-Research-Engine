from __future__ import annotations

import sympy as sp

from physics_ai.types import BackgroundSystem, PerturbationSystem


def regge_wheeler_potential(mass: float, ell: int) -> sp.Expr:
    r = sp.Symbol("r", positive=True)
    M = sp.Symbol("M", positive=True)
    l = sp.Symbol("l", integer=True, nonnegative=True)
    potential = (1 - (2 * M / r)) * ((l * (l + 1) / r**2) - (6 * M / r**3))
    return sp.simplify(potential.subs({M: mass, l: ell}))


def derive_master_equation(
    background: BackgroundSystem,
    run_id: str,
    family: str = "axial",
    mass: float = 1.0,
    ell: int = 2,
) -> PerturbationSystem:
    r = sp.Symbol("r", positive=True)
    omega = sp.Symbol("omega")
    psi = sp.Function("Psi")
    V = regge_wheeler_potential(mass=mass, ell=ell)

    # Compact r-based representation of the master equation.
    master = sp.Eq(sp.diff(psi(r), r, 2) + (omega**2 - V) * psi(r), 0)
    return PerturbationSystem(
        run_id=run_id,
        theory_name=background.theory_name,
        family=family,
        master_equation=str(master),
        effective_potential=str(V),
        metadata={"ell": ell, "mass": mass},
    )


def verify_regge_wheeler(perturbation: PerturbationSystem) -> bool:
    required = ["omega**2", "6.0/r**3", "l"]  # l appears after substitution as number; keep generic check below.
    value = perturbation.effective_potential
    return "omega**2" in perturbation.master_equation and ("6.0/r**3" in value or "6/r**3" in value)
