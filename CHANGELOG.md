# Changelog

All notable changes to this project are documented here.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Added an advanced axon-model example showing labelled `Composite` membrane
  components and the resulting public recording names.

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
- Added `benchmark/hotpaths/run.py --workload cold_run_micro` as a short local
  P9 cold-run baseline for retained Vm, VmRaster observer-only, and
  point-source extracellular paths, with the first CPU baseline recorded under
  `docs/benchmarks/`.
- Normalized scalar retained-Vm benchmark spans with batch/observer routes so
  cold-run reports include comparable runtime preparation, input lowering,
  kernel enqueue/wait, and result assembly stages.
- Closed the P9 runtime-performance planning slice with explicit hotpath
  `time_chunk_steps` CLI policies (`default`, integer, `none`/`unchunked`), a
  local observer-only chunking smoke, and documented decisions to keep larger
  GPU/amplitude/solver optimization campaigns as future work.
- Reprioritized the roadmap so the NumPy/SciPy reference runtime is a future
  bonus while the active path is model/compiler surface cleanup followed by
  realistic benchmark evidence and optimization of the current JAX solver.
- Started P10 model/compiler cleanup with a first audit note and tightened the
  helper policy: global membrane helpers should be kept only when used across
  multiple model families.
- Made `rates_from_tau_inf(x_inf, tau)` the canonical public tau/inf gate
  helper. Built-in Tigerholm and Schild source models now use tuple assignment,
  while the source compiler lowers the helper to scalar internal alpha/beta
  expressions.
- Rejected `boltzmann` and `nernst` as public helpers for now. One-off formulas
  stay in model source, while repeated model-family logic belongs in private
  model modules.
- Factored the duplicated Schild-family Nernst calculation into the private
  `axonscope.membranes.models.schild_common` module instead of promoting it to
  the public membrane-helper surface.
- Tightened membrane source compiler diagnostics for rejected Python constructs:
  loops, statement-level conditionals, mutation, imports inside equation
  functions, I/O/side-effect calls, arbitrary NumPy/JAX calls, object
  construction, and hidden-global reads now report targeted source-location
  errors.
- Added explicit membrane-source current metadata through
  `@currents(conductances=..., reversals=...)` for currents whose
  conductance/reversal terms cannot be inferred from the simple
  `I_x = g_x * (Vm - E_x)` form.
- Preserved source section and `@mechanism(...)` boundaries in compiled
  membrane metadata and exposed mechanism assignments/dependencies through
  `axs.membranes.explain()`.
- Tightened Model IR semantic validation for current linearization terms and
  source-backed metadata consistency, including `source_outputs`,
  `source_provenance`, `source_sections`, and `source_mechanisms`.
- Made composite recording identity explicit: `Composite` accepts labelled
  mappings, duplicate component kinds in sequences now require explicit labels,
  public gate/state/generic-observable names are qualified as
  `component_label.name`, and current/conductance groups remain the only
  automatic duplicate-name aggregates.
- Extended `axs.membranes.explain()` with a model-level component and recording
  output summary, including composite labels and current/conductance aggregate
  names.
- Added generated `graph.json` and `optimized_graph.json` content hashes to
  membrane generated-code and explanation reports.
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
- Rewrote the README around the current public `AxonSimulation` workflow,
  class-based membrane models, validation policy, benchmark surface, and
  executable learning path.
- Flattened the agent and architecture guidance: `AGENTS.md` now keeps only
  operational rules, `GUIDELINES.md` reflects the current P0-P5 state, and
  long architecture-plan pages are labelled as historical/proposal material.
- Audited public examples after benchmark flattening and added a README path
  guard so benchmark/profiling material remains under `benchmark/`.
- Cleaned current user-facing docs for membranes, stimulation, recording,
  analysis, and pool dispatch so they describe public concepts first and keep
  backend/compiler lowering behind explicit internal boundaries.
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

- Fast local validation from the current P10 helper/compiler pass:
  `python -m compileall -q src tests/unit` and
  `pytest -q tests/unit --tb=short` passed with `611 passed, 1 skipped`.
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
