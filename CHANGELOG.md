# Changelog

All notable changes to this project are documented here.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Changed

- Closed the post-P7 membrane-model cleanup around class-based
  `axs.membranes.Model` authoring. Built-in membrane source files under
  `src/axonscope/membranes/models/` are the source of truth for equations,
  defaults, aliases, and derived parameter logic.
- Kept Model IR as internal compiler/runtime vocabulary. Public docs and
  examples should describe membrane models, equations, parameters, gates,
  currents, and observables instead.
- Flattened benchmark documentation around current benchmark surfaces, including
  model-codegen cold/warm runs for built-in and custom membrane models.
- Extended model-codegen benchmarks with generated NumPy/JAX model-step timing,
  correctness rows, and tiny public `AxonSimulation` first/warm runs for the
  new class-based membrane template families.
- Routed public simulation estimates and pipeline inspection through the
  backend execution boundary for backend-owned benchmark support, instead of
  importing concrete JAX lowering helpers from `performance.py` and
  `inspection.py`.
- Clarified validation policy: NRV runs are required for numerical behavior
  changes, while hotpath/realistic benchmarks are required only for performance
  claims.
- Clarified that observer-only solver execution is the strict VmRaster route:
  threshold-style definitions can produce `observations["vm_raster"]`, while
  activation, latency, velocity, thresholds, and recruitment summaries remain
  post-processing.
- Clarified public API draft and membrane docs so proposal material is labelled
  and `MembraneModel` is not presented as a user-facing concept.

### Removed

- Removed the public `PeakVoltageObserver` surface from `axs` and
  `axs.analysis`. `PeakVoltage` remains available only as a post-hoc analysis
  on recorded membrane-voltage traces.
- Kept deleted pre-P7 paths out of the active changelog narrative, including
  `ModelIRMembrane`, `CompositeICM`, `ExtracellularContext`, `axonscope.icm`,
  root simulation helper aliases, and broad solver-side observer designs.

### Validation

- Fast local validation from the post-P7 audit:
  `python -m compileall -q src tests/unit` and
  `pytest -q tests/unit --tb=short` passed with `587 passed, 1 skipped`.
- NRV validation should be recorded here only after a fresh NRV-ready run.

## [0.2.0] - 2025-11-25

### Added

- Introduced `IonChannelModelBase`.
- Introduced `CompositeICM` for user-defined composite membrane models.
- Added the initial Sundt model.

### Changed

- Updated core axon and solver classes to support custom ion-channel models.
- Upgraded the solver stack to use JAX tridiagonal solvers.
- Improved runtime performance compared with earlier implementations.

### Fixed

- Improved internal solver consistency.

## [0.1.0] and Earlier

Initial development period including:

- Passive membrane implementation
- Hodgkin-Huxley validation
- Rattay-Aberham implementation
- First Crank-Nicholson solver versions
- Early benchmarking experiments

These changes were not formally documented. Please refer to git history for
details.
