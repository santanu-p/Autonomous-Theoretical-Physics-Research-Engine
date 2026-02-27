from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from physics_ai.types import TheorySpec
from physics_ai.utils import load_yaml_or_json


def load_theory(path: str | Path) -> TheorySpec:
    file_path = Path(path)
    payload = load_yaml_or_json(file_path)
    return TheorySpec.model_validate(payload)


def validate_theory(path: str | Path) -> tuple[bool, str]:
    try:
        theory = load_theory(path)
    except (ValidationError, ValueError) as exc:
        return False, str(exc)
    summary = (
        f"Theory '{theory.name}' validated. fields={len(theory.fields)}, "
        f"symmetries={len(theory.symmetries)}, parameters={len(theory.parameters)}."
    )
    return True, summary


def classify_theory_family(theory: TheorySpec) -> str:
    action = theory.action.lower().replace(" ", "")
    if "f(r)" in action or "f_r" in action:
        return "f(R)"
    if "einstein-hilbert" in action or "r-2*lambda" in action or "r-2lambda" in action:
        return "GR"
    if "gauss-bonnet" in action or "gb" in action:
        return "Einstein-scalar-GB"
    return "generic_modified_gravity"
