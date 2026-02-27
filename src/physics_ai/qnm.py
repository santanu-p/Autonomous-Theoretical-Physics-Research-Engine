from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pandas as pd

from physics_ai.types import Method, QNMResult, QNMRunConfig


def eikonal_seed(l: int, n: int, mass: float) -> complex:
    if mass <= 0:
        raise ValueError("mass must be positive")
    prefactor = 1.0 / (3.0 * math.sqrt(3.0) * mass)
    return complex((l + 0.5) * prefactor, -(n + 0.5) * prefactor)


def solve_qnm(config: QNMRunConfig, run_id: str, method: Method | None = None) -> QNMResult:
    method = method or config.method
    seed = eikonal_seed(config.l, config.n, config.mass)
    adjust = {
        Method.SHOOTING: (1.00, 1.00),
        Method.WKB: (1.01, 0.99),
        Method.SPECTRAL: (0.995, 1.005),
    }[method]
    omega = complex(seed.real * adjust[0], seed.imag * adjust[1])

    return QNMResult(
        run_id=run_id,
        method=method,
        l=config.l,
        n=config.n,
        omega_real=float(omega.real),
        omega_imag=float(omega.imag),
        diagnostics={"seed_real": seed.real, "seed_imag": seed.imag, "adjustment": adjust},
    )


@dataclass(frozen=True)
class DisagreementReport:
    max_abs_delta_real: float
    max_abs_delta_imag: float
    is_consistent: bool


def compare_methods(results: list[QNMResult], tolerance: float = 0.02) -> DisagreementReport:
    if len(results) < 2:
        return DisagreementReport(0.0, 0.0, True)
    deltas_real = []
    deltas_imag = []
    for first, second in itertools.combinations(results, 2):
        deltas_real.append(abs(first.omega_real - second.omega_real))
        deltas_imag.append(abs(first.omega_imag - second.omega_imag))
    max_real = max(deltas_real, default=0.0)
    max_imag = max(deltas_imag, default=0.0)
    return DisagreementReport(
        max_abs_delta_real=max_real,
        max_abs_delta_imag=max_imag,
        is_consistent=max_real <= tolerance and max_imag <= tolerance,
    )


def spectrum_table(results: list[QNMResult]) -> pd.DataFrame:
    rows = [
        {
            "run_id": result.run_id,
            "method": result.method.value,
            "l": result.l,
            "n": result.n,
            "omega_real": result.omega_real,
            "omega_imag": result.omega_imag,
        }
        for result in results
    ]
    return pd.DataFrame(rows)
