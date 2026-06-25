# AxonScope TODO

Living execution plan for cleanup, stabilization, examples, runtime policy, and
solver/backend work.

`GUIDELINES.md` is the architecture reference. `agent.md` is the agent working
guide. `todo.md` should stay short: current work, next decisions, and the
minimum references needed to act. Long benchmark narratives belong in
`benchmark/reports/`; design alternatives belong in `ideas/`.

## Snapshot

Updated on 2026-06-23.

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
- [x] Move JAX-dependent stimulation compilation under
  `backends/jax/stimulation_runtime.py`; `axonscope.stimulation` remains
  descriptive.
- [x] Hide direct JAX backend adapters from the public simulation entry layer:
  `simulation.py` enters through `axonscope.backends.execution`.
- [x] Move host-side batch row helpers into
  `preparation/runtime_batches.py`; backend array lowering stays under
  `backends/jax`.

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
- [x] Keep public concepts simple: clamps, analytical quick-start helpers,
  extracellular footprints, drives, stimulation, populations.
- [x] Preserve the static-footprint/dynamic-stimulus split internally.
- [x] Keep factorized/dynamic Vext reuse gated by dense-equivalence tests.
- [x] Audit low-level point-source context usage in tests and benchmarks; migrate
  active paths to typed stimulation and mark the remaining pseudo-double case as
  reference-only validation, not an alternate user route.

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
- [x] Redesign pool progress reporting around structured dispatch/backend
  events: route choice, preparation, input lowering, kernel progress, and result
  assembly.

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
- [x] Provide realistic example with NRV built-in geometry capabilities.
- [ ] Add a didactic basic example for high-frequency block. This first requires
  a block-detection analysis path so the example can distinguish propagation,
  activation failure, and true conduction block instead of relying on ad-hoc
  visual inspection.

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

- [x] Audit public objects and examples for `x`, `y`, `z`, trajectory,
  anatomical-frame, nerve-geometry, electrode-geometry, or CAD assumptions.
- [x] Remove `AxonInstance` placement arguments/properties and
  `set_position(...)`; reject the old public path explicitly.
- [x] Keep intrinsic axon position `s = 0 ... length` as the core coordinate used
  for layout, clamps, recording selectors, event positions, and footprints.
- [x] Ensure `Axon`, `AxonInstance`, `ExtracellularFootprint`, and solver
  preparation do not require world coordinates or external geometry ownership.
- [x] Move analytical point-source helpers under `axs.analytical`; public
  examples now build sampled footprints/drives/stimulation, and solver
  execution consumes the generic footprint contract.
- [x] Remove the legacy `local_point_source_context(...)` shortcut; the only
  public point-source path is helper -> sampled footprint/drive/stimulation.
- [x] Keep NRV and other geometry frameworks behind `examples/with_nrv/` or
  adapter-style inputs that hand AxonScope intrinsic footprints and metadata.
- [x] Rename or remove user-facing fields/docs that imply AxonScope owns
  real-world placement beyond temporary analytical examples.
- [x] Add architecture guardrails so public examples/docs cannot reintroduce
  `AxonInstance(..., y=...)`, `z=...`, `x_offset=...`, or
  `set_position(...)`.

### 10. Docs

- [x] Fix stale solver docs, including `pcr_adaptive` cutoff `4096`.
- [x] Update recording/analysis docs to strict VmRaster and post-hoc rich
  analyses.
- [x] Update dispatch docs so advanced snippets match the actual backend/input
  lowering modules.
- [x] Refresh stale factorized-Vext wording in `GUIDELINES.md`,
  `docs/solver_organization.md`, and backend docstrings: the code now uses
  generic `factorized_footprint` lowering for sampled footprints.
- [x] Remove stale "localize context" language from `GUIDELINES.md`; the public
  path is analytical helper -> sampled footprint/drive/stimulation.
- [ ] Keep proposal docs clearly labeled when they show future APIs.
- [ ] Prepare proper Sphinx Documentation
- [ ] Do/update all docstrings 

### 11. Validation

- [x] Run focused unit tests after each cleanup slice.
- [x] Run architecture guardrails after dependency-boundary changes.
- [x] Run example import/smoke tests after examples flattening.
- [ ] Run NRV validation only for numerical behavior changes.
- [ ] Re-run hotpath/realistic benchmarks only when making performance claims.
- [x] Use `benchmark/nrv_performance/population_tsim_scaling.py` to diagnose
  AxonScope-vs-NRV population timing before optimizing long realistic runs:
  point source, <=100 fibers, observer-only vs full-Vm, runtime vs `tsim`.
  Local validation `population_tsim_20260625_081718` shows the retained
  default path is faster than NRV on first-run and warm-run timings.
- [x] Run the matching Kaggle GPU validation preset:
  `benchmark/kaggle/run_kernel.py --benchmark population_tsim_gpu`, then record
  the downloaded CSV/JSON/profile directory and summarize first/warm GPU timing.
  Kaggle P100 run `20260625_083510_population_tsim_gpu_NvidiaTeslaP100`
  completed on commit `344b6d2` with `jax_backend=gpu`, `groups=2`,
  `padded=1`; AS first-run `7.18-8.10 s`, warm-run `0.088-0.239 s` for
  synthetic mixed populations of `25/50/100` fibers at `tsim=0.5/1 ms`.
- [x] Run the large Kaggle GPU population check:
  `benchmark/kaggle/run_kernel.py --benchmark population_tsim_gpu_1000`.
  Kaggle P100 run `20260625_125433_population_tsim_gpu_1000_NvidiaTeslaP100`
  completed on commit `90e62c8` with `jax_backend=gpu`, `groups=2`,
  `padded=1`; AS first-run `24.50 s` / warm-run `1.92 s` at `tsim=0.5 ms`,
  and AS first-run `23.78 s` / warm-run `2.23 s` at `tsim=1 ms` for a
  synthetic mixed population of `1000` fibers.
- [ ] Use the cold-path profile mode before making performance claims from
  first-call timings: `population_cold_path_smoke`, `--profile-cold-path`,
  `--profile-warm-path`, and `--clear-jax-caches`. Track at least
  `runtime.prepare`, `kernel.enqueue`, `kernel.dispatch_jax`, cache hits/misses,
  and first-vs-warm deltas.

### 12. Cold-Run Optimization

- [x] Add this as a dedicated workstream instead of burying it under validation.
- [x] Keep observer-only singleton groups on compact batch observers instead of
  scalar fallback.
- [x] Split batch runtime caching into a static structural cache plus a
  per-time-grid cache so repeated `tsim/dt` sweeps reuse prepared membrane,
  cable, and extracellular runtime data.
- [x] Prepare padded double-cable extracellular arrays host-side and transfer
  one stacked array per field instead of per-row JAX preparation.
- [x] Group shifted MRG/double-cable rows by membrane-family set rather than
  membrane-prefix compatibility, so NRV `node_shift`/AxonScope `x_shift`
  populations do not fragment into many solver routes.
- [x] Add nested cold-path benchmark spans for batch runtime base preparation,
  cable stack, extracellular stack, and membrane stack.
- [x] Reduce double-cable population cold `runtime.prepare` time, especially MRG
  parameter batches with padding. The current path now uses a minimal
  AxNode/passive base runtime, direct solver-axon membrane encoding, NumPy
  AxNode initial gates, and a row-parametric AxNode/passive family backend.
  Smoke profile (`50/100` fibers, `tsim=0.5 ms`) shows double-cable
  `stack_membrane` around `37/75 ms` and 100-fiber total `runtime.prepare`
  around `469 ms`.
- [x] Avoid preparing a full representative cable/extracellular runtime for
  parameter batches whose row-stacked arrays replace those fields.
- [x] Reduce first-call membrane/channel preparation time by compiling repeated
  compartment models once, grouping heterogeneous backends by cached signatures,
  preparing heterogeneous initial arrays host-side, and broadcasting eligible
  Rattay/Rattay+passive initial gates from a NumPy host calculation.
- [x] Use static gate metadata instead of calling `init_gates(...)` just to
  discover gate counts; keep actual gate equations owned by channel models.
- [x] Reduce observer-only duration-sweep JAX compile pressure by defaulting
  `BatchOptions.none()` to stable time chunks and assembling local VmRaster
  chunk states into the public full-duration raster.
- [x] Reduce first-call JAX compile pressure from row-indexed MRG/static
  membrane identity: encode AxNode/passive family parameters in dynamic gate
  rows and keep the JAX static backend/membrane stable for eligible MRG-like
  double-cable batches.
- [ ] Keep shape bucketing internal and opt-in until benchmarks show an end to
  end cold-run win. After the family backend, smoke runs still showed no CPU
  total-time gain versus default (`25/50/100` fibers, `tsim=0.5 ms`), even
  though `kernel.dispatch_jax` can be slightly lower.
- [x] Close the remaining cold-run cleanup for `kernel.dispatch_jax` and
  single-cable population preparation. The retained changes are deliberately
  narrow: Rattay/Rattay+passive host-side initial gates for the single-cable
  mixed-population group, and pre-lowering factorized single-cable footprints to
  diffusion forcing footprints before the JIT call. Rejected broader attempts
  (`linear_membrane_only`, generic uniform gate broadcast) because they
  increased cold compile time or fragmented the internal route.
- [x] Validate the retained default path on NRV-vs-AxonScope population timing:
  `population_tsim_20260625_081718` reports 2 dispatch groups / 1 padded group,
  AS first-run `3.6-5.7 s` versus NRV `12.0-19.9 s`, and AS warm-run
  `0.13-0.60 s` on `25/50/100` fiber grids with `tsim=0.5/1 ms`.
- [ ] Future performance work: investigate persistent JAX compilation/cache
  policy or a dedicated compiler-level strategy if cold `kernel.dispatch_jax`
  remains a product requirement. Keep this separate from solver-route cleanup.
- [x] Add an optional human-readable cold-run progress display that reports the
  active step (`building plan`, `preparing runtime`, `compiling JAX kernel`,
  `solving`, `assembling results`, etc.). Prefer wiring it through the existing
  structured progress/hotpath events rather than ad-hoc prints, and make it most
  useful for first-call/cold runs where compilation can look stalled.
  `DispatchProgress` now reports dispatch planning, batch recording, runtime
  preparation, input lowering, optional JAX/scalar compilation points, solving,
  and result assembly through one structured path; see
  `examples/advanced/runtime/04_cold_run_progress.py`.
- [x] Keep cold/warm profiling as a first-class benchmark output and compare
  `first`, `warm`, `cold-warm`, cache hits/misses, `kernel.dispatch_jax`, and
  `runtime.prepare` before/after each optimization.
- [x] Preserve the one-path user contract: population observer-only execution
  should stay on compact batch observers, not scalar fallbacks.

### 13. Misc
- [ ] Clean Benchmark solver

## Later

- Prepare a publication-grade benchmark campaign for AxonScope versus baselines.
  The campaign should cover velocity, activation-threshold curves, and
  recruitment curves across: `dt`, `Nx`, `Naxons`, FP32 versus FP64; full Vm,
  single/probe Vm, and observer-only outputs; single-cable, double-cable, and
  mixed populations; same-diameter versus different-diameter cohorts within the
  same model family; CPU versus GPU versus NRV. Keep this as a reproducible
  campaign plan with fixed presets, saved raw data, plots, and publication-ready
  summary tables.
- Explore recruitment amplitude micro-batching as a benchmark axis, but keep
  the runtime/protocol default at one amplitude per solver call until evidence
  says otherwise. Benchmark candidate `amplitude_batch_size` values such as 1,
  2, 4, and 8 against peak memory, footprint duplication
  `footprint[B, Nx]`, cold/warm time, and observer-only result assembly.
- Benchmark and formalize `time_chunk_steps` policy for observer/result
  assembly. Compare unchunked, 250, 500, 1000, and adaptive values across
  full Vm, probe Vm, and observer-only outputs; track peak memory, chunk
  overhead, cold/warm time, GPU utilization, result equivalence, and whether
  defaults should depend on `nt`, `Naxons`, recording mode, or backend.
- Promote reusable NRV integration pieces out of examples when they stabilize:
  realistic fascicle/fiber-table extraction, NRV LIFE/FEM footprint sampling
  into AxonScope `ExtracellularStimulation`, NRV recruitment-result decoding by
  fiber row, and compact AxonScope-versus-NRV recruitment summaries. Keep them
  under `axonscope.integrations.nrv` so NRV still owns external geometry while
  AxonScope owns intrinsic axon dynamics.
- Park performance optimization for now. When optimization resumes, start with
  a cold-path audit for large synthetic/GPU populations (`n=1000`): split
  `build pool`, `dispatch.build_plan`, `runtime.prepare`, and
  `kernel.dispatch_jax`; investigate row-by-row planning/preparation overhead
  before changing kernel routes or adding scheduling complexity.
- GPU dispatch scheduling: memory-aware bucket/coalesce first, optional async
  enqueue second, only after memory budgets and group-route inspection exist.
  see axonscope_dispatch_scheduling_gpu_note.md
- Improve GPU solver: see axonscope_gpu_tridiagonal_solver_literature_synthesis.md and update axonscope_double_cable_exact_gpu_solver_roadmap.md
- Provide scipy runtime for scalar/tiny simulations
- Studies: callable threshold curves, recruitment curves, conduction validation,
  parameter sweeps, reuse policies, retention policies, and study results.
- Serialization: final schemas, typed serialization, etc
- Work on HPC integration
- DSL model AXONSCOPE_RUNTIME_AGNOSTIC_DSL_ARCHITECTURE.md
- Work on FEM footprint integration, see `ideas/fem_axon_gpu_coupling_design.md`.
  Start with the CPU/NRV path before thinking GPU FEM: split benchmarks into
  FEM solve, first footprint, cached footprint sampling, and AxonScope solve;
  cache reusable FEM field bases; avoid repeated point-location by introducing
  an axon embedding/projection representation; then choose between full
  precomputed footprints, chunked projection, and future fused projection-solver
  paths by memory budget.

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
