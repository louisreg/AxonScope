# Changelog

All notable changes to this project are documented here.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Added backend-independent `Stimulus`, `IntracellularCurrentClamp`, and
  `ExtracellularContext` descriptors.
- Added NumPy evaluation helpers on stimulation contexts.
- Added JAX-compatible stimulation runtime helpers in
  `axonscope.stimulation.runtime` and solver-runtime compilation in
  `axonscope.solvers.runtime`.
- Added `PointSourceElectrode` and extracellular context attachment helpers.
- Added package-level exports for `axonscope.axons`, `axonscope.solvers`,
  `axonscope.icm`, and `axonscope.utils`.
- Added private MRG morphology helpers inside the MRG-like axon template and
  NRV morphology/geometry comparison tests.
- Added generic heterogeneous membrane layout support through
  `CompartmentMembraneLayout` and `HeterogeneousMembraneModel`.
- Added MRG myelinated axon support with nodal `AxnodeICM` and passive
  internodal membrane models.
- Added `MembraneStateSpec` for model-owned membrane state variables.
- Added MRG extracellular AxonScope-vs-NRV baseline export under
  `benchmark/nrv_performance/baselines/`.
- Added a shared `Solver` benchmark runner with JSON/CSV output under
  `benchmark/runtime/`.
- Added benchmark run metadata and JSON comparison tooling for performance
  regression checks.
- Added solver runtime preparation dataclasses and helpers to separate axon
  descriptions from solver-side arrays, initial states, and compiled stimuli.
- Added solver-runtime precomputation of imposed extracellular `Vstim` samples
  for Crank-Nicholson extracellular solves.
- Added optional solver-runtime precomputation of intracellular current-density
  samples for future batch/chunk kernels.
- Added explicit `SingleCableKernel` and `DoubleCableKernel` solver kernels
  for the optimized Crank-Nicholson path.
- Added explicit JIT-compiled VM-only paths inside `SingleCableKernel` and
  `DoubleCableKernel` for the default Crank-Nicholson solve.
- Added an HH NRV/AxonScope performance comparison script over `dt`, `Nx`, and
  simulation duration.
- Added NRV/AxonScope performance comparison profiles for HH and MRG
  intracellular and extracellular workloads with blocked AxonScope timings,
  spike metrics, velocity estimates, spatial alignment diagnostics, and optional
  `m` gate comparison metrics.
- Added explicit NRV and AxonScope output materialization timings to the
  NRV/AxonScope performance comparison so runtime comparisons include usable
  traces.
- Added a small NRV performance plotting CLI for AxonScope-vs-NRV timing and
  speedup sweeps.
- Added solve, materialization, total usable-output, and compile-estimate
  timings to the shared solver benchmark runner.
- Added named benchmark suite manifests and runners under `benchmark/runtime/`
  and `benchmark/nrv_performance/`, with NRV performance suites for smoke, full,
  MRG, and MRG extracellular gate diagnostics.
- Added a dedicated `mrg_extracellular_perf` NRV performance suite for warm
  runtime comparisons without gate diagnostics.
- Added `docs/validation.md` to document fast CI and local NRV validation
  commands.
- Added an `examples/advanced/` area for population/dispatcher workflows.
- Added Pint-oriented unit helpers for voltage and microampere plotting
  conversions.
- Added unit and NRV tests for extracellular stimulation, heterogeneous ICM
  backends, membrane dynamics delegation, MRG morphology, and MRG geometry.
- Added runnable examples under `examples/basic/`.
- Added `agent.md` to document project-specific development guidance and coding conventions.
- Added `colab_benchmark_cpu_vs_gpu.ipynb` for AxonScope CPU vs GPU benchmarking with JAX, AxonScope simulation workloads, and performance visualization.
- Added Phase 0 architecture guardrail tests for the root guidelines reference,
  removed compatibility aliases/signatures, remaining raw-string public-domain
  inventory, and internal import boundaries.
- Added the new executable `AxonSimulation` root object with `.run()` support
  for one axon or a small population through the existing scalar/pool execution
  paths.
- Added `examples/advanced/example_08_root_axon_simulation.py` as the didactic
  demo for the executable simulation root concept.
- Added public `AxonPopulation` as the typed cohort container for ordered
  `AxonInstance` rows, including one-row population execution through
  `AxonSimulation`.
- Added `examples/advanced/example_09_axon_population.py` as the didactic demo
  for explicit population construction.
- Added public `Axon.diameter` and `Axon.diameter_values(...)` inspection
  helpers so constructor geometry stays easy to read without reaching through
  `axon.layout`.
- Added typed recording signal selectors under `axs.signals` and the public
  `Signal`/`RecordingSpatial` types.
- Added `examples/advanced/example_10_typed_recording_signals.py` as the
  didactic demo for typed recording selectors.
- Added typed activation position selectors under `axs.positions`, including
  `ALL`, `CENTER`, `DISTAL`, `At(...)`, and `Indices(...)`.
- Added `examples/advanced/example_11_typed_position_selectors.py` as the
  didactic demo for typed activation-position targets.
- Added public `axs.axons.CableFormulation` for typed cable formulation
  selection.
- Added `examples/advanced/example_12_cable_formulation.py` as the didactic
  demo for typed cable formulation selectors.
- Added opaque public `AxonId` and `DriveId` identifiers for typed Phase 2
  API contracts.
- Added `ExtracellularFootprint`, `ExtracellularDrive`,
  `ExtracellularStimulation`, and explicit dense `ExtracellularPotential`
  objects for factorized extracellular stimulation.
- Added analytical footprint builders on `AnalyticalExtracellularContext` and
  `PointSourceElectrode`.
- Added `examples/advanced/example_13_extracellular_footprint_drive.py` as the
  didactic demo for factorized extracellular footprints and drives.
- Added opt-in hotpath benchmarking through `axs.enable_benchmark(...)`,
  `axs.disable_benchmark(...)`, `axs.benchmark_report(...)`,
  `axs.reset_benchmark()`, and `with axs.benchmark(...)`.
- Added hotpath benchmark JSONL, CSV, and metadata outputs with stage timings,
  array shape/dtype/byte metadata, backend/device metadata, and explicit JAX
  device synchronization at the kernel boundary.
- Added `examples/advanced/example_14_hotpath_benchmarking.py` as the didactic
  demo for Phase 2.5 diagnostic traces.
- Added `benchmark/hotpaths/` as the cataloged location for Phase 2.5 workload
  scripts, including `run.py --list`, a README registry, and smoke/scale size
  presets for hotpath traces.
- Added `benchmark/hotpaths/COLAB.md` to document the manual Google Colab GPU
  trace protocol when local GPU execution is unavailable.
- Added Phase 3 preparation signatures for arrays, stimuli, extracellular
  footprints, drives, and stimulation collections under `axs.preparation`.
- Added `examples/advanced/example_15_preparation_signatures.py` as the
  didactic demo for reusable-preparation signatures.

### Changed

- Rewrote `README.md` as a short current API entry point and moved detailed
  batch, recording, validation, and benchmark contracts to dedicated docs.
- Updated `agent.md` to require example updates for public API/workflow changes
  and to prefer clean pre-release user interfaces over compatibility shims.
- Registered root `GUIDELINES.md` as the project philosophy and master
  product/solver architecture direction in `agent.md` and `todo.md`.
- Expanded `todo.md` with the guideline-derived implementation queue through
  the object-model, typed API, extracellular footprint, planning, backend,
  result, analysis, performance, study, and serialization phases.
- Clarified in `GUIDELINES.md` and `agent.md` that each new advanced concept
  or non-trivial workflow needs a runnable didactic demo in `examples/advanced/`.
- Clarified in `agent.md` that examples should favor a line-by-line tutorial
  flow with short comments over extra helper-function scaffolding.
- Replaced public `Recording(variables=...)` and `spatial_mode=...` string
  selectors with typed `signals=...` and `spatial=...` inputs.
- Replaced public `ActivationCriterion(positions=..., indices=...)` selectors
  with typed `target=axs.positions.*` selectors.
- Replaced public raw formulation strings on axon constructors with
  `axs.axons.CableFormulation` values.
- Replaced the new extracellular drive identifier surface with typed
  `axs.DriveId(...)` values instead of raw strings before release.
- Instrumented the pool dispatcher around planning, group execution, runtime
  preparation, input materialization, kernel enqueue/wait, batch splitting, and
  public-result packaging when hotpath benchmarking is active.
- Updated NRV validation tests to use unit-bearing `Stimulus` time arguments
  with the stabilized public API.
- Renamed the current executable per-axon protocol object from
  `AxonSimulation` to `AxonInstance`, including the source module
  `axon_instance.py`, public exports, examples, docs, and tests; no
  compatibility alias is kept for the old prototype name.
- Replaced internal imports that went through the public `axonscope` facade with
  package-internal imports.
- Removed package-level `axs.analysis` and `axs.visualization` aliases; use
  `axs.results.analysis` and `axs.results.visualization` instead.
- Removed old `threshold` and `min_distance` aliases from post-hoc result
  analysis helpers; use `threshold_mV` and `min_distance_ms`.
- Removed the `AxonInstance.intracellular_clamps` alias; use
  `intracellular_contexts`.
- Removed `axs.run_batch`; use `axs.simulate_pool` as the single public pool
  simulation wrapper.
- Changed public simulation wrappers to `duration`/`dt` keyword arguments and
  removed `duration_ms`/`dt_ms`/`tsim` compatibility names from that facade.
- Removed `duration_ms`/`dt_ms` aliases from direct solver calls; use
  solver-level `tsim`/`dt`.
- Renamed public membrane template geometry arguments from `diameter_um` to
  `diameter` for `Tigerholm`, `Schild94`, and `Schild97`; these now require
  unit-bearing lengths while retaining internal `diameter_um` parameters.
- Renamed the public recording temporal filter input from `sample_dt_ms` to
  `sample_dt`; explicit sampling intervals now require time units while the
  internal canonical field remains `sample_dt_ms`.
- Renamed public recording spatial filter input from `positions_um` to
  `positions`; explicit recording positions now require length units while the
  internal canonical field remains `positions_um`.
- Renamed public intracellular clamp placement inputs from `position_um` to
  `position`; explicit clamp positions now require length units while runtime
  clamp state remains `position_um`.
- Changed explicit analytical extracellular medium conductivity to require
  units through `AnalyticalExtracellularContext(sigma=...)`; omitting `sigma`
  still uses the default `0.3 S/m` medium.
- Replaced `PointSourceElectrode` constructor coordinate aliases with
  quantity-oriented `x`, `y`, `z`, and `min_distance` inputs; internal
  canonical coordinate fields remain in micrometers with read-only meter
  properties.
- Renamed public simulation placement inputs from `x_offset_um`/`y_um`/`z_um`
  to `x_offset`/`y`/`z`; explicit positions now require length units while
  internal runtime/result fields remain in micrometers.
- Changed activation threshold and recruitment protocol current inputs to
  require units for bounds, tolerances, callable/vector bounds, and recruitment
  amplitudes instead of interpreting plain numbers as microamperes.
- Changed public `Stimulus` constructors to require unit-bearing time inputs
  for explicit starts, durations, sample grids, and shifts, while keeping
  generic waveform amplitudes normalized by their consuming clamp or electrode.
- Split the old flat modules into packages:
  `axons/`, `solvers/`, `icm/`, `benchmarking/`, and `utils/`.
- Removed the old monolithic `axons.py`, `solvers.py`, `icm_compute.py`, and
  `math_functions.py` modules in favor of the package layout.
- Moved NumPy stimulus evaluation out of the solver runtime.
- Kept solver/backend-specific JAX compilation in the solver runtime.
- Moved shared solver recording helpers out of `CrankNicholson` so reference
  and prototype solvers do not depend on Crank-Nicholson private internals.
- Changed extracellular Crank-Nicholson stepping to index precomputed imposed
  `Vstim` samples instead of re-evaluating electrode contexts inside the time loop.
- Changed optimized Crank-Nicholson to dispatch to specialized single-cable and
  double-cable kernels instead of carrying both voltage layouts in one scan body.
- Changed optimized Crank-Nicholson to precompute intracellular current-density
  samples before entering the default VM-only JIT kernel.
- Specialized the double-cable 2x2 block-tridiagonal solve to operate on scalar
  coefficient arrays instead of materialized `(Nx, 2, 2)` block matrices.
- Reworked the scalar double-cable block-tridiagonal forward/backward sweep to
  use `jax.lax.scan`, reducing indexed array updates inside the JIT time loop.
- Hoisted double-cable VM-only invariant terms and time-sampled drive inputs
  outside the scan body to simplify the optimized Vi/Ve kernel.
- Skipped unused membrane-current planning work in the stateless double-cable
  VM-only path, keeping the optimized Vi/Ve loop focused on gate prediction and
  block solves.
- Consolidated multicompartment axons so section layouts share common clamp
  handling from `Axon`.
- Replaced the MRG-specific masked ICM layout with the generic heterogeneous
  membrane layout.
- Moved MRG node-count/length construction helpers to the myelinated axon layer;
  MRG morphology tables are now private implementation details of the MRG-like
  double-cable template.
- Simplified `IonChannelModelBase` so sodium, potassium, and calcium-specific
  dynamic helpers are no longer defined on every membrane model.
- Updated README and examples to the current package API.
- Updated solver benchmark comparison output guards to tolerate tiny numerical
  output drift before reporting `changed_output`.
- Reorganized benchmark scripts into `benchmark/runtime/` and
  `benchmark/nrv_performance/`, removing legacy wrappers and stale exploratory
  entry points while the project is still pre-use.
- Simplified the axon stimulation API so intracellular clamps now receive an
  explicit `Stimulus` object.
- Renamed the production solver module to lowercase `crank_nicholson.py`.
- Moved Crank-Nicholson reference and prototype variants to
  `axonscope.solvers.experimental`, leaving `axonscope.solvers` focused on
  production solvers and runtime helpers.
- Renamed extracellular context helpers to `add_extracellular_context`,
  `clear_extracellular_context`, and `extracellular_potential_mV`.
- Interpreted `PointSourceElectrode` coordinates in the same global frame as
  `axon.set_position(...)`; scalar, NumPy evaluation, and pool batch paths now
  convert point-source y/z coordinates to each axon's local transverse offsets
  internally.
- Made fixed-step solver grids exact: `duration_ms` must be an integer multiple
  of `dt_ms` instead of silently rounding up and simulating past the requested
  final time.
- Ignored generated benchmark logs, benchmark figures, NRV figures, caches, and
  local build artifacts.

### Fixed

- Fixed double-cable extracellular Crank-Nicholson RHS handling by including the
  previous imposed extracellular potential term for capacitive extracellular
  coupling.
- Fixed MRG extracellular validation drift visible in the gating variable `m`
  against NRV.
- Fixed examples that still called removed stimulus/context convenience methods.
- Fixed stale internal imports that referenced the old flat module layout.
- Removed stale playground diagnostics and Crank-Nicholson backend experiments
  that referenced older runtime internals.
- Added explicit guardrails rejecting stateful membrane components in generic
  `CompositeICM` and heterogeneous membrane layouts until those paths propagate
  arbitrary membrane state variables.
- Fixed the MRG point-source threshold validation so it preserves the nominal
  MRG diameter and aligns the electrode to the central node of Ranvier before
  checking diameter-dependent threshold monotonicity.
- Reorganized examples so `examples/basic/` stays didactic, pool dispatch lives
  under `examples/advanced/`, and timing/profiling demos live under
  `benchmark/runtime/`.
- Numbered the example scripts, added explicit units to example variables and
  axes, and added a minimal no-NRV pool dispatch example before the advanced NRV
  workflow.
- Updated examples to construct parameters as Pint quantities through
  `axonscope.units`, converting to plain arrays only at plotting boundaries.
- Allowed direct solver time arguments and axon temperatures to accept
  Pint-like quantities.
- Removed skipped NRV threshold placeholder tests now covered by extracellular
  threshold scans and existing systematic NRV validations.

### Validation

- Unit tests cover the current stimulus API, electrode evaluation, solver
  extracellular behavior, heterogeneous ICM layouts, membrane state specs, and
  public package exports.
- NRV comparison tests cover MRG morphology, compartment geometry, intracellular
  models, extracellular models, and velocity/numerical guardrails.
- The `tests/nrv/extracellular` and `tests/nrv/numerics` subsets pass against
  the current global point-source and exact fixed-step solver contracts.
- The `tests/nrv/intracellular` and `tests/nrv/velocity_vs_diameter` subsets
  also pass against the current solver/runtime organization.

### Known Notes

- The generic heterogeneous MRG membrane layout is cleaner than the former masked
  model but currently slower. Profiling and backend specialization are the next
  planned cleanup step.
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
