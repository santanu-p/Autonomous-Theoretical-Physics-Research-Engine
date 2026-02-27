from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlurmJobSpec:
    job_name: str
    command: str
    partition: str = "gpu"
    gpus: int = 1
    cpus_per_task: int = 4
    mem_gb: int = 16
    time_limit: str = "02:00:00"


def render_slurm_script(spec: SlurmJobSpec) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.job_name}",
            f"#SBATCH --partition={spec.partition}",
            f"#SBATCH --gpus={spec.gpus}",
            f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
            f"#SBATCH --mem={spec.mem_gb}G",
            f"#SBATCH --time={spec.time_limit}",
            "",
            "set -euo pipefail",
            spec.command,
        ]
    )


def write_slurm_script(spec: SlurmJobSpec, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_slurm_script(spec), encoding="utf-8")
    return path
