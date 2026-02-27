from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from physics_ai.physics_rules import evaluate_constraints
from physics_ai.qnm import compare_methods, solve_qnm, spectrum_table
from physics_ai.types import Method, QNMRunConfig, ScanJob


def _iter_parameter_points(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [grid[key] for key in keys]
    points = []
    for combo in itertools.product(*values):
        points.append({key: float(value) for key, value in zip(keys, combo)})
    return points


def run_scan(job: ScanJob) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    points = _iter_parameter_points(job.parameter_grid)
    for point in points:
        working_theory = job.theory.model_copy(deep=True)
        working_theory.parameters.update(point)

        results = []
        for l in job.l_values:
            for n in job.n_values:
                base_config = QNMRunConfig(l=l, n=n, mass=working_theory.parameters.get("mass", 1.0))
                for method in job.methods:
                    results.append(solve_qnm(base_config, run_id=job.run_id, method=method))

        disagreement = compare_methods(results)
        constraints = evaluate_constraints(job.run_id, working_theory, qnm_results=results)
        table = spectrum_table(results)
        for key, value in point.items():
            table[key] = value
        table["constraints_valid"] = constraints.is_valid
        table["disagreement_real"] = disagreement.max_abs_delta_real
        table["disagreement_imag"] = disagreement.max_abs_delta_imag
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_stability_plot(df: pd.DataFrame, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = df[df["constraints_valid"] == True]  # noqa: E712
    invalid = df[df["constraints_valid"] == False]  # noqa: E712
    ax.scatter(valid["omega_real"], valid["omega_imag"], s=20, c="green", label="valid")
    ax.scatter(invalid["omega_real"], invalid["omega_imag"], s=20, c="red", label="filtered")
    ax.set_xlabel("Re(omega)")
    ax.set_ylabel("Im(omega)")
    ax.set_title("QNM stability map")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def summarize_scan(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "valid_rows": 0}
    if "constraints_valid" in df.columns:
        valid_count = int(df["constraints_valid"].sum())
    else:
        valid_count = int(len(df))
    return {
        "rows": int(len(df)),
        "valid_rows": valid_count,
        "invalid_rows": int(len(df) - valid_count),
        "max_disagreement_real": float(df.get("disagreement_real", pd.Series([0.0])).max()),
        "max_disagreement_imag": float(df.get("disagreement_imag", pd.Series([0.0])).max()),
    }
