# Changelog

AxonFleet is pre-release. This file records release-level changes only;
implementation history remains in Git, active work in `todo.md`, and validated
architecture in `GUIDELINES.md`.

## Unreleased

### Added

- One public `AxonSimulation` workflow for single axons and populations, with
  typed execution, recording, inspection, and estimation policies.
- Source-backed built-in and custom membrane models, including HH-like,
  stateful, composite, Gaines, and Markov Nav1.x families.
- Compact activation, latency, spike-count, bounded spike-time, and VmRaster
  outputs alongside dense recording and post-hoc analysis.
- Threshold and recruitment protocols with native population and numeric-axis
  batching.
- Sampled extracellular footprint integration, including the NRV population
  and footprint bridge.
- Exact CPU and CUDA routes for the retained single- and double-cable models.

### Changed

- Converged the package on one descriptive `Section -> Layout -> Axon` model,
  one `AxonInstance` stimulation contract, and one canonical result family.
- Made generated NumPy/JAX/Triton membrane artifacts the only runtime model
  path; Model IR remains internal compiler machinery.
- Reorganized JAX preparation, input lowering, execution, recording, and result
  assembly behind `axonfleet.runtime.execution`.
- Replaced historical benchmark and validation surfaces with the current
  launchers documented in `benchmark/README.md` and `docs/validation.md`.

### Removed

- Legacy simulation helpers, compatibility aliases, alternate observer and
  threshold APIs, unused runtime backends, rejected solver candidates, and
  test-only production paths.
- Historical phase diaries, obsolete API proposals, generated validation
  figures, and archived benchmark implementations.

Release validation counts will be added only after the final P19 pre-v1 gate.

## Earlier Development

Versions before the first public release were experimental and did not maintain
a stable API. Git history is the authoritative record for that period.
