# AxonScope TODO

Living execution plan for cleanup, stabilization, examples, runtime policy, and
solver/backend work.

`GUIDELINES.md` is the architecture reference. `agent.md` is the agent working
guide. `todo.md` should stay short: current work, next decisions, and the
minimum references needed to act. Long benchmark narratives belong in
`benchmark/reports/`; design alternatives belong in `ideas/`.

## Snapshot

Updated on 2026-06-22.

The solver optimization campaign is closed. The active work is now a
stabilization pass:

- keep only retained solver routes in active runtime code;
- preserve the separation between public/runtime layers and JAX backend code;
- make runtime/device/precision policies executable, not estimate-only;
- make planning, batching, preparation, lowering, kernel route, and result
  assembly inspectable;
- keep scalar and population runs on one coherent public result model;
- remove real-world coordinate ownership from core/public simulation objects;
- flatten all examples against the public API;
- remove stale docs and obsolete branches.

## Non-Negotiables

- AxonScope is pre-release: prefer clean breaking changes over compatibility
  shims.
- Public double-cable solver options are exactly `auto`, `thomas`, `pcr`,
  `pcr_soa`, and `pcr_adaptive`.
- `auto` resolves from the effective execution device: CPU/default -> `thomas`,
  GPU-like -> `pcr_adaptive`.
- `pcr_adaptive` uses `pcr_soa` for `B <= 4096`, then matrix-layout `pcr`.
- Pallas, Triton, JAX-Triton, CUDA FFI, split iterative, associative-transfer,
  and pseudo-double candidates are archived/standby evidence, not public solver
  routes.
- Solver-side observer-only execution produces strict
  `observations["vm_raster"]`; activation, latency, velocity, threshold, and
  recruitment are post-processing.
- `PeakVoltage` remains post-hoc on recorded Vm unless a dedicated benchmarked
  solver-side implementation is designed.
- Factorized `Vext` is an internal optimization. Do not expose it as a public
  mode and do not re-densify when the caller only needs observer output.
- Public examples must not import solver/backend internals.
- World/anatomical coordinates, trajectories, nerve geometry, and electrode CAD
  stay outside AxonScope core; AxonScope consumes intrinsic positions and
  sampled footprints.

## Active Queue

Work should start here unless the user asks otherwise.

### 1. Solver Route Cleanup

- [x] Trace every active scalar, pool, batch, single-cable, double-cable,
  VmRaster, dense/factorized Vext, and fallback route.
- [x] Move split/associative/custom-kernel candidates out of
  `src/axonscope/solvers/` into benchmark/archive locations.
- [x] Keep active benchmark runners focused on retained public solver options;
  split archived reproduction scripts if needed.
- [x] Add guardrails that prevent standby solver choices from reappearing in
  public options or runtime dispatch.

### 2. Runtime And Backend Boundary

- [x] Split solver-facing contracts from JAX implementation details.
- [x] Move fixed-step time-grid validation out of JAX-heavy solver helpers into
  neutral `axonscope.timebase`.
- [x] Restrict `axonscope.solvers` to the stable solver facade; import
  kernels/runtime helpers through explicit internal modules.
- [x] Move or wrap JAX runtime/kernels/lowering under `backends/jax` where the
  target architecture requires it.
- [x] Keep `solvers` as the public facade for stable solver classes/options, not
  as a catch-all for backend internals.
- [x] Add guardrails for forbidden dependencies, especially high-level modules
  importing solver-specific lowering.

### 3. Recording And VmRaster

- [x] Introduce or finalize a `RecordingPlan` boundary so `recording.py` no
  longer imports solver batch options directly.
- [x] Move VmRaster result containers and CPU decoders to a result/analysis
  boundary; keep JAX bit-packing in backend/runtime code.
- [x] Update protocols to consume backend-neutral VmRaster output rather than
  importing solver observer internals.
- [x] Remove stale documentation for superseded broad observer design.

### 4. Vext, Stimulation, And Placement API

- [x] Audit stimulation/context names against the product boundary in
  `GUIDELINES.md`.
- [x] Keep public concepts simple: clamps, point-source electrodes,
  extracellular footprints, drives, stimulation, populations.
- [x] Preserve the static-footprint/dynamic-stimulus split internally.
- [x] Keep factorized/dynamic Vext reuse gated by dense-equivalence tests.

### 5. Runtime Policy

- [x] Promote `axs.Runtime`, `axs.Device`, and `axs.PrecisionPolicy` from
  estimate-only values to real execution policy.
- [x] Decide final spelling: keyword arguments on `run(...)` or an
  `ExecutionPolicy` object.
- [x] Resolve requested JAX device explicitly and fail clearly if unavailable.
- [x] Make precision part of preparation/compile cache identity.
- [x] Handle `float64` requests explicitly with JAX `jax_enable_x64`.
- [x] Ensure solver `auto` routing uses the effective requested device.

### 6. Pipeline Inspection

- [x] Define host-side inspection records for planning, dispatch/batch, and
  preparation.
- [x] Add `print()` summaries for planning, dispatch/batch, and preparation.
- [x] Extend inspection to input lowering, observer/recording lowering, kernel
  routing, and result assembly.
- [x] Add a lightweight plot for dispatch groups, retained Vm width, and
  materialization choices.
- [x] Add deeper plots where useful: padding, memory, probe positions, and
  result assembly.
- [x] Keep inspection opt-in and zero-overhead by default.
- [x] Avoid device-to-host transfers unless explicitly requested by an
  execution-time capture.
- [ ] Redesign pool progress reporting around dispatch groups, cohort rows, or
  callback events; examples may still use temporary `progress=True` prints, but
  the current boolean progress does not expose enough per-simulation visibility.

### 7. Examples Flattening

- [x] Audit `examples/basic/`, `examples/advanced/`, `examples/with_nrv/`,
  `examples/tutorials/`, `benchmark/`, README commands, and
  `examples/README.md` together.
- [x] Replace direct `CrankNicholson`, solver/backend internals, or observer
  runtime imports in public examples with public API usage.
- [x] Move benchmark/profiling-only material under `benchmark/` or rewrite it as
  a public inspection/runtime-policy tutorial.
- [x] Add a concise runtime-policy example after the execution policy surface is
  real.
- [x] Keep pseudo-double/custom-kernel/archive experiments out of user-facing
  examples.
- [x] Split `with_nrv/` out of `advanced/` for NRV-owned geometry workflows.
- [x] Add a tutorial index for future notebook mini-courses.
- [ ] Write real notebook tutorials under `examples/tutorials/` following the
  indexed mini-course sequence.
- [x] Flatten `examples/basic/` into mostly linear `main()` scripts with
  didactic comments and minimal helper functions.
- [x] Flatten `examples/advanced/` so each public script has one top-level
  `main()` and keeps callbacks/helpers local to the concept being taught.
- [x] Split advanced protocol examples into threshold-parameter and recruitment
  waveform workflows instead of mixing both in one script.
- [x] Rewrite each moved example for the new didactic order instead of only
  preserving the previous script body.
- [ ] Provide realistic example with NRV built-in geometry capabilities.

### 8. Simulation Results Model

- [x] Introduce `RecordedAxis` as the shared scalar/pool interpretation of
  recorded intrinsic positions and original layout indices.
- [x] Add descriptor-based `signal(...)` access to one-axon views and population
  results.
- [x] Decide and enforce the scalar return contract:
  `simulate(...) -> AxonSimulationResult`, one row through `.single` or `[0]`.
- [x] Remove public scalar-solver-result export and public view conversion paths.
- [x] Audit every remaining public result class/path: `AxonSimulationResult`,
  `AxonResultView`, `RecordedSignal`, `RecordingManifest`, `VmRasterResult`,
  analysis reports, protocol return types, and internal dense storage blocks.
- [x] Finish one-axon and population result semantics: signal access,
  indexing, iteration, metadata, diagnostics, observations, final state, and
  analysis/report APIs.
- [x] Keep analyses separate from raw numerical results while making common
  workflows ergonomic: `result.analyze(...)`, `result.report(...)`, protocol
  summaries, and population denominators.
- [x] Remove duplicate result containers, forwarding aliases, or shape-specific
  convenience paths that conflict with the canonical model.
- [x] Update basic/advanced examples and primary docs so result usage teaches
  the final contract.
- [x] Clean stale proposal/secondary docs that still mention historical
  public result APIs.

### 9. Remove World Coordinate Ownership

- [ ] Audit public objects and examples for `x`, `y`, `z`, trajectory,
  anatomical-frame, nerve-geometry, electrode-geometry, or CAD assumptions.
- [ ] Keep intrinsic axon position `s = 0 ... length` as the core coordinate used
  for layout, clamps, recording selectors, event positions, and footprints.
- [ ] Ensure `Axon`, `AxonInstance`, `ExtracellularFootprint`, and solver
  preparation do not require world coordinates or external geometry ownership.
- [ ] Move analytical point-source helpers toward footprint generation/examples
  only; solver execution should consume sampled footprints/drives.
- [x] Add a small `axs.analytical.local_point_source_context(...)` helper so
  public examples can keep transverse point-source geometry out of
  `AxonInstance`.
- [ ] Keep NRV and other geometry frameworks behind `examples/with_nrv/` or
  adapter-style inputs that hand AxonScope intrinsic footprints and metadata.
- [ ] Rename or remove user-facing fields/docs that imply AxonScope owns
  real-world placement beyond temporary analytical examples.

### 10. Docs

- [x] Fix stale solver docs, including `pcr_adaptive` cutoff `4096`.
- [x] Update recording/analysis docs to strict VmRaster and post-hoc rich
  analyses.
- [x] Update dispatch docs so advanced snippets match the actual backend/input
  lowering modules.
- [ ] Keep proposal docs clearly labeled when they show future APIs.
- [ ] Prepare proper Sphinx Documentation
- [ ] Do/update all docstrings 

### 11. Validation

- [x] Run focused unit tests after each cleanup slice.
- [x] Run architecture guardrails after dependency-boundary changes.
- [x] Run example import/smoke tests after examples flattening.
- [ ] Run NRV validation only for numerical behavior changes.
- [ ] Re-run hotpath/realistic benchmarks only when making performance claims.
  
### 12. Misc
- [ ] Clean Benchmark solver

## Later

- GPU dispatch scheduling: memory-aware bucket/coalesce first, optional async
  enqueue second, only after memory budgets and group-route inspection exist.
  see axonscope_dispatch_scheduling_gpu_note.md
- Improve GPU solver: see axonscope_gpu_tridiagonal_solver_literature_synthesis.md and update axonscope_double_cable_exact_gpu_solver_roadmap.md
- Improve cold run and not only warm (althrough warm is more important)
- Provide scipy runtime for scalar/tiny simulations
- Studies: callable threshold curves, recruitment curves, conduction validation,
  parameter sweeps, reuse policies, retention policies, and study results.
- Serialization: final schemas, typed serialization, etc
- Work on HPC integration
- DSL model AXONSCOPE_RUNTIME_AGNOSTIC_DSL_ARCHITECTURE.md
- Work on FEM footprint integration, see fem_axon_gpu_coupling_design.md

## Key References

- Architecture reference: `GUIDELINES.md`
- Agent guide: `agent.md`
- Solver organization: `docs/solver_organization.md`
- Axon model organization: `docs/axon_model_organization.md`
- Stimulation model: `docs/stimulation.md`
- Pool dispatch: `docs/pool_dispatch.md`
- Recording/results/analysis: `docs/results_recording_analysis.md`
- Active solver README: `benchmark/solvers/README.md`
- Solver campaign report:
  `benchmark/reports/double_cable_solver_optimization_2026_06.md`
- Vm observer report:
  `benchmark/reports/compact_activation_observer_2026_06_20.md`
- Pseudo-double standby: `benchmark/pseudo_double/README.md`
- Archived solver spikes:
  `benchmark/archived_solver_spikes/`, `benchmark/triton_solver/`,
  `benchmark/jax_triton_solver/`, `benchmark/cuda_ffi_solver/`,
  `tests/archive/solver_spikes/`
