# Validation

AxonScope separates fast checks from local scientific validation.

## Fast Checks

The fast default check set does not require a local NRV checkout:

```bash
python -m pip install -e ".[dev]"
git diff --check
MPLBACKEND=Agg python -m pytest -q tests/unit --tb=short
```

These checks should run locally before documentation/API cleanup is marked
complete. If this repository adds a CI workflow, that workflow should run this
same fast set on pull requests and pushes to the main development branches.

## Local NRV Validation

NRV validation requires an environment where `nrv` imports successfully. Run the
full suite with:

```bash
MPLBACKEND=Agg python -m pytest -q tests/nrv --tb=short
```

Record only fresh, dated results here or in release notes after rerunning the
suite in an NRV-ready environment. Do not carry forward historical pass counts
without the run date, environment, and command.

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
