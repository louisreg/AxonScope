# Validation

AxonScope separates fast CI checks from local scientific validation.

## Fast CI

The default GitHub Actions workflow runs only tests that do not require a local
NRV checkout:

```bash
python -m pip install -e ".[dev]"
git diff --check
MPLBACKEND=Agg python -m pytest -q tests/unit --tb=short
```

These checks should run on every pull request and push to the main development
branches.

## Local NRV Validation

NRV validation requires an environment where `nrv` imports successfully. In this
workspace, the full NRV suite currently passes:

```bash
MPLBACKEND=Agg python -m pytest -q tests/nrv --tb=short
```

Current reference result:

```text
116 passed
```

The warnings emitted by this run currently come from NRV/NumPy array-to-scalar
deprecations, not from failing AxonScope assertions.

Useful targeted subsets:

```bash
MPLBACKEND=Agg python -m pytest -q tests/nrv/extracellular tests/nrv/numerics --tb=short
MPLBACKEND=Agg python -m pytest -q tests/nrv/intracellular tests/nrv/velocity_vs_diameter --tb=short
```

## Interpretation

- `tests/unit` guards public API behavior, solver runtime preparation, batch
  execution, units, stimulation, recording, and guardrails.
- `tests/nrv/extracellular` checks point-source stimulation and MRG/unmyelinated
  extracellular behavior against NRV.
- `tests/nrv/numerics` checks morphology, compartment geometry, boundary
  conditions, and solver numerics against NRV-adjacent references.
- `tests/nrv/intracellular` checks membrane dynamics and observable traces
  against NRV.
- `tests/nrv/velocity_vs_diameter` checks conduction velocity trends and
  AxonScope/NRV velocity agreement.
