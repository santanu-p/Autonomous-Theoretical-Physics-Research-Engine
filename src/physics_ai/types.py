from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Method(str, Enum):
    SHOOTING = "shooting"
    WKB = "wkb"
    SPECTRAL = "spectral"


class FieldSpec(BaseModel):
    name: str
    field_type: str = Field(default="scalar")
    spin: int | None = None


class SymmetrySpec(BaseModel):
    name: str
    generators: list[str] = Field(default_factory=list)


class AssumptionSet(BaseModel):
    dimension: int = 4
    coordinates: list[str] = Field(default_factory=lambda: ["t", "r", "theta", "phi"])
    static: bool = True
    spherical: bool = True
    notes: list[str] = Field(default_factory=list)


class TheorySpec(BaseModel):
    name: str
    action: str
    fields: list[FieldSpec] = Field(default_factory=list)
    symmetries: list[SymmetrySpec] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)
    assumptions: AssumptionSet = Field(default_factory=AssumptionSet)

    @field_validator("action")
    @classmethod
    def ensure_non_empty_action(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("action must be non-empty")
        return value


class DerivationArtifact(BaseModel):
    run_id: str
    theory_name: str
    stage: str
    equations: list[str]
    canonical_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackgroundSystem(BaseModel):
    run_id: str
    theory_name: str
    ansatz: str = "static_spherical"
    unknown_functions: list[str]
    reduced_equations: list[str]
    consistency: dict[str, Any]


class PerturbationSystem(BaseModel):
    run_id: str
    theory_name: str
    family: str = "axial"
    master_equation: str
    effective_potential: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QNMRunConfig(BaseModel):
    l: int = 2
    n: int = 0
    mass: float = 1.0
    method: Method = Method.SHOOTING
    max_iter: int = 40
    tol: float = 1e-8


class QNMResult(BaseModel):
    run_id: str
    method: Method
    l: int
    n: int
    omega_real: float
    omega_imag: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ScanJob(BaseModel):
    name: str
    run_id: str
    theory: TheorySpec
    parameter_grid: dict[str, list[float]] = Field(default_factory=dict)
    methods: list[Method] = Field(default_factory=lambda: [Method.SHOOTING, Method.WKB, Method.SPECTRAL])
    l_values: list[int] = Field(default_factory=lambda: [2])
    n_values: list[int] = Field(default_factory=lambda: [0])

    @model_validator(mode="after")
    def validate_grid(self) -> "ScanJob":
        for key, values in self.parameter_grid.items():
            if not values:
                raise ValueError(f"parameter grid entry '{key}' is empty")
        return self


class ConstraintReport(BaseModel):
    run_id: str
    is_valid: bool
    ghost_instability: bool
    wrong_sign_kinetic: bool
    divergent_background: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class NoveltyNeighbor(BaseModel):
    identifier: str
    title: str
    concept_similarity: float
    equation_similarity: float
    parameter_overlap: float
    composite_similarity: float


class NoveltyReport(BaseModel):
    run_id: str
    novelty_score: float
    explanation: str
    neighbors: list[NoveltyNeighbor] = Field(default_factory=list)


class PaperBuildSpec(BaseModel):
    run_id: str
    title: str
    authors: list[str]
    output_dir: str
    include_latex: bool = True
    include_markdown: bool = True


class ProposalSpec(BaseModel):
    name: str
    action: str
    parameters: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class CampaignSpec(BaseModel):
    name: str
    run_id: str
    seed_theory: TheorySpec
    proposal_count: int = 3
    enable_llm: bool = False
    max_retries_per_stage: int = 2


class EventRecord(BaseModel):
    run_id: str
    stage: str
    status: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
