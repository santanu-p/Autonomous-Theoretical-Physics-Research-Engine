# Usage

## Validate

```bash
physai theory validate examples/theories/einstein_hilbert.yaml
```

## Derive equations

```bash
physai derive eom examples/theories/einstein_hilbert.yaml
physai derive background examples/theories/einstein_hilbert.yaml --ansatz static_spherical
```

## Perturb and QNM

```bash
physai derive perturb artifacts/<run_id>/background/background.json
physai run qnm artifacts/<run_id>/perturb/perturbation.json --method shooting
```

## Scan

```bash
physai run scan examples/scan.yaml
```

## Novelty and paper

```bash
physai novelty score run_001
physai paper build run_001
```

## Full autonomous campaign

```bash
physai orchestrate autonomous --campaign examples/campaigns/default.yaml
```
