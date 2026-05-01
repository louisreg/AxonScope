# Changelog

All notable changes to this project are documented here.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Added backend-independent `Stimulus`, `IntracellularCurrentClamp`, and
  `ExtracellularContext` descriptors.
- Added NumPy evaluation helpers in `axonscope.stimulus_eval`.
- Added JAX solver-runtime stimulus compilation in
  `axonscope.solvers.stimulus_runtime`.
- Added `PointSourceElectrode` and extracellular context attachment helpers.
- Added package-level exports for `axonscope.axons`, `axonscope.solvers`,
  `axonscope.icm`, `axonscope.morphology`, and `axonscope.utils`.
- Added MRG morphology helpers and NRV morphology/geometry comparison tests.
- Added generic heterogeneous membrane layout support through
  `CompartmentMembraneLayout` and `HeterogeneousMembraneModel`.
- Added MRG myelinated axon support with nodal `AxnodeICM` and passive
  internodal membrane models.
- Added `MembraneStateSpec` for model-owned membrane state variables.
- Added MRG extracellular AxonScope-vs-NRV baseline export under
  `benchmark/solver_baseline/`.
- Added a shared `Solver` benchmark runner with JSON/CSV output under
  `benchmark/solver_runtime/`.
- Added unit and NRV tests for extracellular stimulation, heterogeneous ICM
  backends, membrane dynamics delegation, MRG morphology, and MRG geometry.
- Added runnable examples under `examples/basic/`.

### Changed

- Split the old flat modules into packages:
  `axons/`, `solvers/`, `icm/`, `morphology/`, `benchmarking/`, and `utils/`.
- Removed the old monolithic `axons.py`, `solvers.py`, `icm_compute.py`, and
  `math_functions.py` modules in favor of the package layout.
- Moved NumPy stimulus evaluation out of the solver runtime.
- Kept solver/backend-specific JAX compilation in the solver runtime.
- Consolidated multicompartment axons so `AxonMultiCompBase` inherits common
  clamp handling from `AxonBase`.
- Replaced the MRG-specific masked ICM layout with the generic heterogeneous
  membrane layout.
- Moved MRG node-count/length construction helpers to the myelinated axon layer;
  `morphology.mrg` now focuses on morphology tables and interpolation.
- Simplified `IonChannelModelBase` so sodium, potassium, and calcium-specific
  dynamic helpers are no longer defined on every membrane model.
- Updated README and examples to the current package API.
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

### Validation

- Unit tests cover the current stimulus API, electrode evaluation, solver
  extracellular behavior, heterogeneous ICM layouts, membrane state specs, and
  public package exports.
- NRV comparison tests cover MRG morphology, compartment geometry, intracellular
  models, extracellular models, and velocity/numerical guardrails.

### Known Notes

- The generic heterogeneous MRG membrane layout is cleaner than the former masked
  model but currently slower. Profiling and backend specialization are the next
  planned cleanup step.
- The `playground/` and legacy `benchmark/CrankNicholson_runtime/` directories
  contain active experiments and are not stable public API.

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
