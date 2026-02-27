# Architecture

The implementation is split into deterministic core modules:

- `physics_ai.theory`: schema loading and validation
- `physics_ai.symbolic`: action compilation, canonicalization, EOM derivation
- `physics_ai.background`: static spherical reduction
- `physics_ai.perturbation`: master wave equation extraction
- `physics_ai.qnm`: QNM method wrappers and method-consistency analysis
- `physics_ai.physics_rules`: ghost/sign/divergence filters
- `physics_ai.explore`: parameter scans and stability outputs
- `physics_ai.proposal`: LLM/fallback structured action proposals
- `physics_ai.novelty`: corpus ingestion and novelty scoring
- `physics_ai.paper`: LaTeX + Markdown generation
- `physics_ai.orchestrate`: autonomous DAG execution
- `physics_ai.storage`: metadata and artifact persistence

Data flow:

1. `TheorySpec` input
2. Symbolic derivation -> `DerivationArtifact`
3. Background reduction -> `BackgroundSystem`
4. Perturbations -> `PerturbationSystem`
5. QNM runs + constraints
6. Novelty scoring
7. Paper generation
