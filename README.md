# Autonomous Theoretical Physics Research Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santanu-p/Autonomous-Theoretical-Physics-Research-Engine/blob/main/colab_run.ipynb)
[![Open Full Run In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santanu-p/Autonomous-Theoretical-Physics-Research-Engine/blob/main/colab_full_run.ipynb)

This repository provides a runnable baseline implementation of an autonomous
theoretical physics research engine focused on static, spherically symmetric
black holes in 4D modified gravity.

## Features

- Deterministic theory schema and symbolic representation
- Baseline Euler-Lagrange style equation derivation workflows
- Static spherical background reduction to ODE systems
- Regge-Wheeler style perturbation module
- QNM solvers (shooting, WKB, spectral approximation wrappers)
- Parameter-space scanning and physical consistency filters
- LLM-backed proposal generation with strict JSON schema
- Novelty scoring against ingested literature metadata
- Dual paper generation (`paper.tex` and `paper.md`)
- Autonomous orchestrator pipeline (`propose -> paper`)
- CLI, Python package API, and notebook-compatible flow

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
pytest
```

Validate a theory:

```bash
physai theory validate examples/theories/einstein_hilbert.yaml
```

Run autonomous campaign:

```bash
physai orchestrate autonomous --campaign examples/campaigns/default.yaml
```

Run long-horizon autonomous daemon (example: 24h budget):

```bash
physai orchestrate daemon --campaign examples/campaigns/default.yaml --hours 24 --sleep-seconds 300
```

## Layout

- `src/physics_ai/` package code
- `tests/` unit and integration tests
- `examples/` sample theory and campaign specs
- `docs/` architecture and usage notes
- `notebooks/` notebook-compatible workflows

## Notes

This is a foundation implementation with deterministic interfaces and baseline
physics pipelines. Advanced symbolic tensor completeness, high-order numerical
accuracy guarantees, and production-scale corpus ingestion are modularized for
iterative hardening.
