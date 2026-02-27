from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from physics_ai.explore import run_scan
from physics_ai.physics_rules import evaluate_constraints
from physics_ai.qnm import compare_methods, solve_qnm
from physics_ai.types import QNMRunConfig, ScanJob

try:
    from dask.distributed import Client, LocalCluster  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Client = None
    LocalCluster = None


def run_scan_distributed(job: ScanJob, workers: int = 2) -> pd.DataFrame:
    if Client is None or LocalCluster is None:
        return run_scan(job)

    keys = list(job.parameter_grid)
    points = [dict(zip(keys, combo)) for combo in itertools.product(*[job.parameter_grid[k] for k in keys])]
    if not points:
        return run_scan(job)

    def task(point: dict[str, float]) -> dict[str, Any]:
        theory = job.theory.model_copy(deep=True)
        theory.parameters.update({k: float(v) for k, v in point.items()})
        results = []
        for l in job.l_values:
            for n in job.n_values:
                config = QNMRunConfig(l=l, n=n, mass=theory.parameters.get("mass", 1.0))
                for method in job.methods:
                    results.append(solve_qnm(config=config, run_id=job.run_id, method=method))
        disagreement = compare_methods(results)
        constraints = evaluate_constraints(job.run_id, theory, qnm_results=results)
        return {
            **point,
            "avg_real": float(sum(r.omega_real for r in results) / len(results)),
            "avg_imag": float(sum(r.omega_imag for r in results) / len(results)),
            "disagreement_real": disagreement.max_abs_delta_real,
            "disagreement_imag": disagreement.max_abs_delta_imag,
            "constraints_valid": constraints.is_valid,
        }

    cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        futures = client.map(task, points)
        rows = client.gather(futures)
        return pd.DataFrame(rows)
    finally:
        client.close()
        cluster.close()
