# Changelog

All notable changes to this project are documented here.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Added backend-independent `Stimulus`, `IntracellularCurrentClamp`, and
  `ExtracellularContext` descriptors.
- Added NumPy evaluation helpers in `axonscope.stimulation.evaluation`.
- Added JAX solver-runtime stimulus compilation in
  `axonscope.solvers.stimulus_runtime`.
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
- Added a GitHub Actions CI workflow for install, whitespace checking, and the
  fast unit suite.
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

### Changed

- Split the old flat modules into packages:
  `axons/`, `solvers/`, `icm/`, `benchmarking/`, and `utils/`.
- Removed the old monolithic `axons.py`, `solvers.py`, `icm_compute.py`, and
  `math_functions.py` modules in favor of the package layout.
- Moved NumPy stimulus evaluation out of the solver runtime.
- Kept solver/backend-specific JAX compilation in the solver runtime.
- Moved shared solver recording helpers out of `CrankNicholson` so `Euler`
  no longer depends on Crank-Nicholson private internals.
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
- Renamed solver modules to lowercase `crank_nicholson.py` and `euler.py`.
- Moved Crank-Nicholson reference and prototype variants to
  `axonscope.solvers.experimental`, leaving `axonscope.solvers` focused on
  production solvers and runtime helpers.
- Renamed extracellular context helpers to `add_extracellular_context`,
  `clear_extracellular_contexts`, and `extracellular_potential_mV`.
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
