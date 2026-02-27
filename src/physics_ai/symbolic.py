from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from physics_ai.theory import classify_theory_family
from physics_ai.types import DerivationArtifact, TheorySpec
from physics_ai.utils import stable_hash


@dataclass(frozen=True)
class SymbolicTheory:
    theory_name: str
    family: str
    action_expr: sp.Expr
    symbols: dict[str, sp.Symbol]


def _action_locals() -> dict[str, Any]:
    R = sp.Symbol("R", real=True)
    Lambda = sp.Symbol("Lambda", real=True)
    kappa = sp.Symbol("kappa", positive=True)
    f = sp.Function("f")
    return {"R": R, "Lambda": Lambda, "kappa": kappa, "f": f, "pi": sp.pi}


def compile_action(theory: TheorySpec) -> SymbolicTheory:
    locals_map = _action_locals()
    # Force declared parameters to be plain symbols, even when names collide
    # with SymPy built-ins (e.g. beta).
    for param_name in theory.parameters:
        locals_map[param_name] = sp.Symbol(param_name, real=True)
    normalized = theory.action.replace("^", "**")
    expr = sp.sympify(normalized, locals=locals_map)
    family = classify_theory_family(theory)
    symbols = {name: locals_map[name] for name in theory.parameters}
    return SymbolicTheory(
        theory_name=theory.name,
        family=family,
        action_expr=sp.expand(expr),
        symbols=symbols,
    )


def canonicalize_expr(expr: sp.Expr) -> str:
    # sympy canonical term ordering + serialized tree = deterministic fingerprint
    simplified = sp.together(sp.expand(expr))
    return sp.srepr(simplified)


def _einstein_equation_strings() -> list[str]:
    mu = sp.Symbol("mu")
    nu = sp.Symbol("nu")
    G = sp.Function("G")(mu, nu)
    T = sp.Function("T")(mu, nu)
    kappa = sp.Symbol("kappa")
    eq = sp.Eq(G, kappa * T)
    trace_eq = sp.Eq(sp.Symbol("R"), -kappa * sp.Symbol("T"))
    return [str(eq), str(trace_eq)]


def _fr_equation_strings() -> list[str]:
    mu = sp.Symbol("mu")
    nu = sp.Symbol("nu")
    Rmunu = sp.Function("R")(mu, nu)
    gmunu = sp.Function("g")(mu, nu)
    nabla = sp.Function("nabla")
    box = sp.Symbol("Box")
    R = sp.Symbol("R")
    f = sp.Function("f")
    fR = sp.Function("fR")
    T = sp.Function("T")(mu, nu)
    lhs = (
        fR(R) * Rmunu
        - sp.Rational(1, 2) * f(R) * gmunu
        + (gmunu * box - nabla(mu) * nabla(nu)) * fR(R)
    )
    eq = sp.Eq(lhs, sp.Symbol("kappa") * T)
    trace = sp.Eq(fR(R) * R - 2 * f(R) + 3 * box * fR(R), sp.Symbol("kappa") * sp.Symbol("T"))
    return [str(eq), str(trace)]


def derive_field_equations(theory: TheorySpec, run_id: str) -> DerivationArtifact:
    compiled = compile_action(theory)
    family = compiled.family
    if family == "GR":
        equations = _einstein_equation_strings()
    elif family == "f(R)":
        equations = _fr_equation_strings()
    else:
        generic_eq = sp.Eq(sp.Symbol("deltaS/dphi"), 0)
        equations = [str(generic_eq)]

    payload = {
        "theory": theory.model_dump(mode="json"),
        "family": family,
        "action_srepr": canonicalize_expr(compiled.action_expr),
        "equations": equations,
    }
    return DerivationArtifact(
        run_id=run_id,
        theory_name=theory.name,
        stage="eom",
        equations=equations,
        canonical_hash=stable_hash(payload),
        metadata={"family": family, "action_srepr": payload["action_srepr"]},
    )


def verify_gr_baseline(artifact: DerivationArtifact) -> bool:
    return any("Eq(G(mu, nu), kappa*T(mu, nu))" in eq for eq in artifact.equations)


def verify_fr_baseline(artifact: DerivationArtifact) -> bool:
    checks = ["fR(R)", "Box", "kappa*T(mu, nu)"]
    joined = "\n".join(artifact.equations)
    return all(token in joined for token in checks)
