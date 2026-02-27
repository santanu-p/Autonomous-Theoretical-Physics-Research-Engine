from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from physics_ai.artifacts import write_dataframe
from physics_ai.background import reduce_to_background_odes
from physics_ai.distributed import run_scan_distributed
from physics_ai.explore import run_scan, save_stability_plot, summarize_scan
from physics_ai.hpc import SlurmJobSpec, write_slurm_script
from physics_ai.novelty import default_seed_records, score_novelty
from physics_ai.orchestrate import run_autonomous_campaign
from physics_ai.paper import build_paper
from physics_ai.perturbation import derive_master_equation
from physics_ai.qnm import solve_qnm
from physics_ai.symbolic import derive_field_equations, verify_fr_baseline, verify_gr_baseline
from physics_ai.theory import load_theory, validate_theory
from physics_ai.types import CampaignSpec, Method, PaperBuildSpec, QNMRunConfig, ScanJob
from physics_ai.utils import generate_run_id, load_yaml_or_json


app = typer.Typer(help="Autonomous theoretical physics research engine.")
theory_app = typer.Typer(help="Theory schema operations.")
derive_app = typer.Typer(help="Derivation operations.")
run_app = typer.Typer(help="Numerical run operations.")
novelty_app = typer.Typer(help="Novelty operations.")
paper_app = typer.Typer(help="Paper build operations.")
orchestrate_app = typer.Typer(help="Autonomous orchestration operations.")

app.add_typer(theory_app, name="theory")
app.add_typer(derive_app, name="derive")
app.add_typer(run_app, name="run")
app.add_typer(novelty_app, name="novelty")
app.add_typer(paper_app, name="paper")
app.add_typer(orchestrate_app, name="orchestrate")


@theory_app.command("validate")
def theory_validate(theory_path: Path) -> None:
    ok, message = validate_theory(theory_path)
    if not ok:
        raise typer.Exit(code=1)
    typer.echo(message)


@derive_app.command("eom")
def derive_eom(theory_path: Path, run_id: Optional[str] = typer.Option(None)) -> None:
    run_id = run_id or generate_run_id("derive")
    theory = load_theory(theory_path)
    artifact = derive_field_equations(theory, run_id=run_id)
    output = Path("artifacts") / run_id / "derive" / "eom.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2), encoding="utf-8")

    checks = []
    family = artifact.metadata.get("family")
    if family == "GR":
        checks.append(f"gr_baseline={verify_gr_baseline(artifact)}")
    if family == "f(R)":
        checks.append(f"fr_baseline={verify_fr_baseline(artifact)}")
    typer.echo(f"Wrote {output} {' '.join(checks)}")


@derive_app.command("background")
def derive_background(
    theory_path: Path,
    ansatz: str = typer.Option("static_spherical"),
    run_id: Optional[str] = typer.Option(None),
) -> None:
    if ansatz != "static_spherical":
        raise typer.BadParameter("Only static_spherical is implemented in this baseline.")
    run_id = run_id or generate_run_id("background")
    theory = load_theory(theory_path)
    derivation = derive_field_equations(theory, run_id=run_id)
    background = reduce_to_background_odes(theory, derivation, run_id=run_id)
    output = Path("artifacts") / run_id / "background" / "background.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(background.model_dump(mode="json"), indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output}")


@derive_app.command("perturb")
def derive_perturb(background_id: Path, run_id: Optional[str] = typer.Option(None)) -> None:
    run_id = run_id or generate_run_id("perturb")
    payload = load_yaml_or_json(background_id)
    theory_name = payload.get("theory_name", "unknown")
    from physics_ai.types import BackgroundSystem

    background = BackgroundSystem.model_validate(payload)
    perturb = derive_master_equation(background, run_id=run_id)
    output = Path("artifacts") / run_id / "perturb" / "perturbation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(perturb.model_dump(mode="json"), indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output} for {theory_name}")


@run_app.command("qnm")
def run_qnm(
    perturb_id: Path,
    method: Method = typer.Option(Method.SHOOTING),
    l: int = typer.Option(2),
    n: int = typer.Option(0),
    mass: float = typer.Option(1.0),
    run_id: Optional[str] = typer.Option(None),
) -> None:
    run_id = run_id or generate_run_id("qnm")
    _ = load_yaml_or_json(perturb_id)
    config = QNMRunConfig(l=l, n=n, mass=mass, method=method)
    result = solve_qnm(config, run_id=run_id, method=method)
    output = Path("artifacts") / run_id / "qnm" / f"qnm_{method.value}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output}")


@run_app.command("scan")
def run_scan_cmd(scan_path: Path, distributed: bool = typer.Option(False), workers: int = typer.Option(2)) -> None:
    payload = load_yaml_or_json(scan_path)
    job = ScanJob.model_validate(payload)
    df = run_scan_distributed(job, workers=workers) if distributed else run_scan(job)
    out_dir = Path("artifacts") / job.run_id / "scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "scan.parquet"
    plot_path = out_dir / "stability.png"
    if df.empty:
        df = pd.DataFrame(columns=["omega_real", "omega_imag", "constraints_valid"])
    written_data_path = write_dataframe(df, parquet_path)
    can_plot = {"omega_real", "omega_imag", "constraints_valid"}.issubset(set(df.columns))
    if (not df.empty) and can_plot:
        save_stability_plot(df, plot_path)
    summary = summarize_scan(df)
    summary["scan_file"] = str(written_data_path)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {written_data_path} and {summary_path}")


@run_app.command("slurm-template")
def run_slurm_template(
    command: str = typer.Option("physai run scan examples/scan.yaml"),
    output: Path = typer.Option(Path("examples/campaigns/scan_job.slurm")),
) -> None:
    spec = SlurmJobSpec(job_name="physai_scan", command=command)
    path = write_slurm_script(spec, output)
    typer.echo(f"Wrote {path}")


@novelty_app.command("score")
def novelty_score(run_id: str, equations_path: Optional[Path] = typer.Option(None)) -> None:
    equations = ["d2Psi/dr_*2 + (omega^2 - V)Psi = 0"]
    if equations_path:
        payload = load_yaml_or_json(equations_path)
        equations = payload.get("equations", equations)
    report = score_novelty(
        run_id=run_id,
        hypothesis_text="Autonomous modified gravity candidate",
        equations=equations,
        parameters={"mass": 1.0},
        corpus=default_seed_records(),
    )
    out = Path("artifacts") / run_id / "novelty" / "novelty.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")


@paper_app.command("build")
def paper_build(
    run_id: str,
    title: str = typer.Option("Autonomous Physics AI Report"),
    output_dir: Optional[Path] = typer.Option(None),
) -> None:
    out = output_dir or (Path("artifacts") / run_id / "paper")
    spec = PaperBuildSpec(
        run_id=run_id,
        title=title,
        authors=["Autonomous Physics AI"],
        output_dir=str(out),
        include_latex=True,
        include_markdown=True,
    )
    novelty_payload = {"run_id": run_id, "novelty_score": 0.4, "explanation": "Seed novelty report", "neighbors": []}
    from physics_ai.types import NoveltyReport

    novelty = NoveltyReport.model_validate(novelty_payload)
    outputs = build_paper(
        spec,
        derivation={"family": "GR", "equations": ["Eq(G(mu,nu),kappa*T(mu,nu))"]},
        qnm_summary={"omega_real": 0.3737, "omega_imag": -0.0890},
        novelty=novelty,
    )
    typer.echo(f"Wrote {', '.join(str(v) for v in outputs.values())}")


@orchestrate_app.command("autonomous")
def orchestrate_autonomous(campaign: Path) -> None:
    payload = load_yaml_or_json(campaign)
    campaign_spec = CampaignSpec.model_validate(payload)
    result = run_autonomous_campaign(campaign_spec)
    typer.echo(
        "Campaign complete "
        f"run_id={result.run_id} accepted={result.accepted_theories} rejected={result.rejected_theories}"
    )
    for path in result.paper_outputs:
        typer.echo(f"- {path}")


if __name__ == "__main__":
    app()
