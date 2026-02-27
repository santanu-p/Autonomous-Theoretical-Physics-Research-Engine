from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from physics_ai.artifacts import write_dataframe
from physics_ai.background import reduce_to_background_odes
from physics_ai.explore import summarize_scan
from physics_ai.novelty import fetch_arxiv_records, score_novelty
from physics_ai.paper import build_paper
from physics_ai.perturbation import derive_master_equation
from physics_ai.physics_rules import evaluate_constraints
from physics_ai.proposal import ProposalGenerator
from physics_ai.qnm import compare_methods, solve_qnm, spectrum_table
from physics_ai.storage import (
    constraint_reports_table,
    derivations_table,
    events_table,
    init_db,
    novelty_reports_table,
    paper_builds_table,
    qnm_runs_table,
    save_record,
    theories_table,
    write_artifact,
    write_json_artifact,
)
from physics_ai.symbolic import derive_field_equations
from physics_ai.types import (
    CampaignSpec,
    EventRecord,
    Method,
    PaperBuildSpec,
    QNMRunConfig,
    TheorySpec,
)


@dataclass(frozen=True)
class CampaignResult:
    run_id: str
    accepted_theories: int
    rejected_theories: int
    paper_outputs: list[str]
    summary: dict[str, Any]


def _log_event(engine, run_id: str, stage: str, status: str, message: str, payload: dict[str, Any] | None = None) -> None:
    event = EventRecord(run_id=run_id, stage=stage, status=status, message=message)
    save_record(
        engine,
        events_table,
        {
            "run_id": event.run_id,
            "stage": event.stage,
            "status": event.status,
            "message": event.message,
            "payload": payload or event.model_dump(mode="json"),
        },
    )


def _proposal_to_theory(seed: TheorySpec, name: str, action: str, parameters: dict[str, float]) -> TheorySpec:
    theory = seed.model_copy(deep=True)
    theory.name = name
    theory.action = action
    theory.parameters.update(parameters)
    return theory


def run_autonomous_campaign(campaign: CampaignSpec) -> CampaignResult:
    engine = init_db()
    generator = ProposalGenerator(enable_llm=campaign.enable_llm)
    _log_event(engine, campaign.run_id, "propose", "started", f"Campaign {campaign.name} started.")

    proposals = generator.generate(campaign.seed_theory, campaign.proposal_count)
    accepted = 0
    rejected = 0
    paper_outputs: list[str] = []
    corpus = fetch_arxiv_records()
    aggregate_scan_rows: list[pd.DataFrame] = []

    for proposal in proposals:
        theory = _proposal_to_theory(
            campaign.seed_theory,
            name=proposal.name,
            action=proposal.action,
            parameters=proposal.parameters,
        )
        save_record(
            engine,
            theories_table,
            {
                "run_id": campaign.run_id,
                "name": theory.name,
                "payload": theory.model_dump(mode="json"),
            },
        )
        _log_event(engine, campaign.run_id, "validate", "ok", f"Accepted proposal payload: {proposal.name}")

        derivation = derive_field_equations(theory, run_id=campaign.run_id)
        save_record(
            engine,
            derivations_table,
            {
                "run_id": campaign.run_id,
                "stage": "eom",
                "canonical_hash": derivation.canonical_hash,
                "payload": derivation.model_dump(mode="json"),
            },
        )
        write_json_artifact(
            campaign.run_id,
            "derive",
            f"{theory.name}_eom.json",
            derivation.model_dump(mode="json"),
        )

        background = reduce_to_background_odes(theory, derivation, run_id=campaign.run_id)
        if not background.consistency.get("square_system", False):
            rejected += 1
            _log_event(engine, campaign.run_id, "background", "rejected", f"Inconsistent system: {theory.name}")
            continue

        perturb = derive_master_equation(background, run_id=campaign.run_id)
        write_artifact(campaign.run_id, "perturb", f"{theory.name}_master.txt", perturb.master_equation)

        qnm_results = []
        for method in [Method.SHOOTING, Method.WKB, Method.SPECTRAL]:
            config = QNMRunConfig(method=method, l=2, n=0, mass=theory.parameters.get("mass", 1.0))
            qnm = solve_qnm(config=config, run_id=campaign.run_id, method=method)
            qnm_results.append(qnm)
            save_record(
                engine,
                qnm_runs_table,
                {
                    "run_id": campaign.run_id,
                    "method": qnm.method.value,
                    "omega_real": qnm.omega_real,
                    "omega_imag": qnm.omega_imag,
                    "payload": qnm.model_dump(mode="json"),
                },
            )

        disagreement = compare_methods(qnm_results)
        constraints = evaluate_constraints(campaign.run_id, theory, qnm_results=qnm_results)
        save_record(
            engine,
            constraint_reports_table,
            {
                "run_id": campaign.run_id,
                "is_valid": constraints.is_valid,
                "payload": constraints.model_dump(mode="json"),
            },
        )
        if not constraints.is_valid:
            rejected += 1
            _log_event(
                engine,
                campaign.run_id,
                "constraints",
                "rejected",
                f"{theory.name} rejected by physics filters: {constraints.reasons}",
            )
            continue

        table = spectrum_table(qnm_results)
        table["theory_name"] = theory.name
        table["method_disagreement_real"] = disagreement.max_abs_delta_real
        table["method_disagreement_imag"] = disagreement.max_abs_delta_imag
        aggregate_scan_rows.append(table)

        novelty = score_novelty(
            run_id=campaign.run_id,
            hypothesis_text=f"{theory.name}: {proposal.rationale}",
            equations=derivation.equations + [perturb.master_equation, perturb.effective_potential],
            parameters=theory.parameters,
            corpus=corpus,
        )
        save_record(
            engine,
            novelty_reports_table,
            {
                "run_id": campaign.run_id,
                "novelty_score": novelty.novelty_score,
                "payload": novelty.model_dump(mode="json"),
            },
        )

        out_dir = Path("artifacts") / campaign.run_id / "paper" / theory.name
        paper_spec = PaperBuildSpec(
            run_id=campaign.run_id,
            title=f"{theory.name}: Autonomous Modified-Gravity Analysis",
            authors=["Autonomous Physics AI"],
            output_dir=str(out_dir),
            include_latex=True,
            include_markdown=True,
        )
        outputs = build_paper(
            paper_spec,
            derivation={"family": derivation.metadata.get("family"), "equations": derivation.equations},
            qnm_summary={
                "method_consistent": disagreement.is_consistent,
                "max_abs_delta_real": disagreement.max_abs_delta_real,
                "max_abs_delta_imag": disagreement.max_abs_delta_imag,
            },
            novelty=novelty,
        )
        paper_outputs.extend(str(path) for path in outputs.values())
        save_record(
            engine,
            paper_builds_table,
            {
                "run_id": campaign.run_id,
                "payload": {
                    "theory_name": theory.name,
                    "outputs": {key: str(path) for key, path in outputs.items()},
                },
            },
        )
        accepted += 1
        _log_event(engine, campaign.run_id, "paper", "ok", f"Built paper outputs for {theory.name}")

    if aggregate_scan_rows:
        merged = pd.concat(aggregate_scan_rows, ignore_index=True)
        merged_path = write_dataframe(
            merged, Path("artifacts") / campaign.run_id / "scan" / "qnm_spectrum.parquet"
        )
        summary = summarize_scan(merged)
        summary["spectrum_file"] = str(merged_path)
    else:
        summary = {"rows": 0, "valid_rows": 0}

    _log_event(
        engine,
        campaign.run_id,
        "finalize",
        "ok",
        f"Campaign finished accepted={accepted} rejected={rejected}",
        payload=summary,
    )
    return CampaignResult(
        run_id=campaign.run_id,
        accepted_theories=accepted,
        rejected_theories=rejected,
        paper_outputs=paper_outputs,
        summary=summary,
    )
