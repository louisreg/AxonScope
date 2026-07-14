# AxonScope TODO

Living execution plan for AxonScope cleanup, benchmark evidence, validation,
documentation, and next runtime phases.

`GUIDELINES.md` is the architecture reference. `AGENTS.md` is the agent working
guide. Source, tests, runnable examples, and fresh benchmark reports remain the
implementation truth.

This file is intentionally compact. The full pre-cleanup ledger was archived in
`docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.

## Snapshot

Updated on 2026-07-14 during the P12 GPU cold/warm optimization closeout.

Current state:

- P7 is closed: public membrane authoring is class-based through
  `axs.membranes.Model`; built-ins live under
  `src/axonscope/membranes/models/`.
- Historical `channel_models`, `icm`, `model_ir/models`, and
  `model_ir/builtins.py` paths are removed and must stay absent.
- Model IR remains internal compiler/runtime vocabulary. Users write membrane
  models, equations, parameters, gates, currents, and observables.
- `axs.runtime.numpy` is reserved for a future real NumPy/SciPy reference
  runtime. It must not become a JAX-backed compatibility path.
- Solver-side observer-only execution is the strict VmRaster path under
  `observations["vm_raster"]`; activation, latency, velocity, threshold, and
  recruitment summaries remain post-processing.
- `PeakVoltage` remains post-hoc on recorded Vm unless a dedicated benchmarked
  solver-side design is accepted.
- P3 is paused after current-docs cleanup. Tutorials/Sphinx/docstrings remain
  open.
- P11 is closed for the current JAX runtime, benchmark, and solver-policy
  stabilization pass. Deferred runtime, benchmark, and solver-policy work is
  tracked in `docs/architecture/p11_closeout_2026_07_12.md`.
- P12 is closed for runtime/JAX cleanup, targeted GPU warm/cold optimization,
  and Graphify-guided runtime dead-code cleanup. A final cold-start extension
  now persists the supported double-cable Triton compiled call. Future work
  keeps broader dense-observable benchmarks, generalized compile-cache policy,
  GPU async scheduling, and larger public speed-claim evidence out of the P12
  closeout path.
- Current solver-policy decisions are tracked in
  `docs/architecture/p11_solver_policy_cleanup_decisions_2026_07_11.md`:
  CPU double-cable keeps only Thomas as a production route; GPU double-cable
  currently resolves `auto` through the Triton/tiled-Thomas route while the
  full policy matrix remains the benchmark gate; single-cable stays on the JAX
  tridiagonal route for now.

Fresh local validation from the 2026-07-02 audit:

```text
python -m compileall -q src tests/unit
pytest -q tests/unit --tb=short
587 passed, 1 skipped in 424.89s
```

## Non-Negotiables

- AxonScope is pre-release with one active user. Prefer clean deletion and
  direct convergence over retrocompatibility, shims, aliases, or deprecated
  wrappers.
- One concept, one public name, one execution path, one canonical public result
  model.
- Public examples must not import solver/runtime internals.
- Every public feature, option, workflow, analysis, runtime mode, inspection
  view, or advanced concept must be documented in runnable examples or removed
  from the public surface.
- World/anatomical coordinates, trajectories, nerve geometry, electrode CAD,
  surgical placement, and FEM solving stay outside AxonScope core. AxonScope
  consumes intrinsic positions and sampled footprints.
- Do not remove unfinished TODO items unless they are completed, rejected, or
  moved to a named tracking document.

## Active Plan

### P12 - Runtime Cleanup, Studies, Serialization, Integration

Final P12 performance objective: make warm GPU runs as solver-bound as possible,
especially by reducing JAX/Python dispatch, input lowering, observer packing,
and host/device transfer overheads around the solver. In parallel, reduce
cold-run latency as much as possible through preparation, membrane/runtime
caching, compilation/lowering reuse, and targeted persistent caches. Keep
benchmark evidence separated between warm solver-bound claims and cold-start
improvements.

- [x] P12A runtime contract and sanity benchmark gate:
  use `docs/architecture/p12_runtime_contract_2026_07_12.md` and
  `docs/architecture/p12a_jax_runtime_audit_2026_07_12.md` as the completed
  local CPU plus Kaggle GPU smoke gate. This validates that the initial
  runtime-contract cleanup still runs on the P11-sensitive single-cable and
  double-cable observer-only paths.
- [x] P12B runtime/JAX cleanup:
  use `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md` as the active
  migration note. Homogenize non-solver preparation, recording/observer
  lowering, input semantics, benchmark metadata, and result assembly between
  single-cable and double-cable paths as much as possible without losing P11
  performance.
  - [x] Remove unused rate-table option, the direct public
    `CrankNicholson`/`Solver` execution facade, and the JAX scalar fallback;
    one-row public simulations use the batch route with `B=1`.
  - [x] Split the former JAX batch-kernel monolith into explicit
    `runtime/jax/kernels/single_cable.py`,
    `runtime/jax/kernels/double_cable.py`, shared chunking/factorized/input
    helpers, and `runtime/jax/recording/results.py`; remove the old
    `runtime/jax/batch_kernels.py`/`kernels/batch.py` path.
  - [x] Split active shared numerical primitives out of the old
    `runtime/jax/kernels/common.py` bucket into
    `runtime/jax/cable_geometry.py`,
    `runtime/jax/kernels/double_cable_linear.py`, and
    `runtime/jax/kernels/block_tridiagonal.py`; remove legacy PCR/PCR-SoA and
    diagnostic batched-Thomas helpers from active runtime code, and keep
    double-cable scan bodies split into CPU Thomas and GPU tiled-Thomas/Triton
    files.
  - [x] Review `chunking`, `factorized`, `inputs`, `results`, `core`, and
    `cable_geometry` one by one: keep `chunking`, `factorized`, and `inputs`
    as kernel-only shared helpers; move JAX geometry to
    `runtime/jax/cable_geometry.py`, move result synchronization to
    `runtime/jax/recording/results.py`, and rename the former vague kernel
    `core.py` to `runtime/jax/kernels/double_cable_step.py`.
  - [x] Archive historical P11B/P11C solver probes under
    `benchmark/legacy/p11_solver_exploration/`, delete the
    `jax_triton_cold_start_audit` runner/test surface, and keep the active
    double-cable routes limited to CPU Thomas and GPU looped Triton/tiled
    Thomas.
  - [x] Clean the single-cable JAX kernel surface: split scan bodies into
    `runtime/jax/kernels/single_cable_scans.py`, remove the unsupported
    observer-only sparse-current plus dense-Vstim route, and keep only dense,
    factorized, factorized-sparse, and zero-sparse routes that are reachable
    from the runtime lowering contract.
  - [x] Reorganize the remaining JAX runtime modules by responsibility:
    typed runtime policy in `runtime/jax/policy/`, input payload/build/lowering
    in `runtime/jax/inputs/`, host-side batch preparation and caches in
    `runtime/jax/preparation/`, observer/recording/result synchronization in
    `runtime/jax/recording/`, and JAX profiling/metadata helpers in
    `runtime/jax/benchmarking/`.
  - [x] Reintroduce dense recording only as a batch-native result path for
    single-cable groups: `Recording.full()`, gates, currents, conductances, and
    available state variables lower through the batch route for `B=1` and `B>N`,
    with explicit signal names, result manifests, memory accounting, focused
    tests, and a runnable public example. Do not reintroduce scalar fallback
    execution for this.
  - [x] Dense observable recording was restored as a public batch-native path,
    but its dedicated benchmark evidence is deferred out of P12 because the
    P12 closeout target shifted to observer-only hotpath and cold-start
    runtime cleanup. Benchmark dense observables when they become an active
    performance target rather than holding P12 open.
- [x] Audit `src/axonscope/runtime/jax/` for dead, duplicate, or cable-specific
  host-side code. Delete unused paths, keep solver/kernel-specific code inside
  the JAX runtime, and move semantic-only reusable contracts to
  `src/axonscope/runtime/` when they can support a future NumPy/SciPy runtime.
  Use `docs/architecture/p12b_jax_runtime_reorganization_proposal_2026_07_12.md`
  as the proposed file-responsibility map before moving more modules.
  Method: do the whole audit/move/delete pass first, without running the full
  test suite after each file or folder. Validate once at the end with
  `compileall`, `tests/unit`, `git diff --check`, `vulture`, and only then
  targeted benchmarks if hotpath behavior changed.
  For every directory or root file below, verify that all retained paths are
  still used, that there is no dead or duplicate code, that responsibility is
  not split across redundant routes, and that the code is genuinely
  JAX-specific. If a contract, planning rule, metadata shape, or host-side
  semantic helper is runtime-neutral, move it to `src/axonscope/runtime/` so it
  can serve the future NumPy/SciPy runtime too.
  - [x] Root JAX files: `__init__.py`, `group_runner.py`, `types.py`, and
    `cable_geometry.py`.
  - [x] `runtime/jax/policy/`: typed JAX solver requests, execution context,
    device/precision lowering, and solver-engine resolution.
  - [x] `runtime/jax/inputs/`: payload dataclasses, dense/sparse/factorized
    builders, footprint caches, and semantic input lowering.
  - [x] `runtime/jax/preparation/`: batch runtime materialization, caches,
    shape bucketing, host-to-device array preparation, and row stacking.
  - [x] `runtime/jax/recording/`: VmRaster observer plan/state/update,
    recording lowering, result synchronization, waits, trimming, and
    finalization.
  - [x] `runtime/jax/kernels/`: single-cable, double-cable CPU/GPU, shared
    chunking/factorized/input helpers, double-cable linear-system helpers, and
    Triton integration.
  - [x] `runtime/jax/membranes/`: membrane compiler bridge, Model IR lowering,
    membrane backend implementations, layout aggregation, programs, and
    stacking optimizations.
  - [x] `runtime/jax/benchmarking/`: JAX profiling hooks, memory profiling,
    benchmark metadata, and estimate/inspection support helpers. Source audit
    pass started: the runtime-neutral batch memory-estimate math now lives in
    `runtime/memory_estimates.py`.
- [x] Audit root `src/axonscope/runtime/` for the future NumPy/SciPy runtime
  contract. Group files by responsibility, verify every retained path is used,
  delete dead or duplicate code, and keep only runtime-neutral contracts,
  host-side semantic planning, public policy, and the concrete-runtime
  execution boundary at this level. Concrete numerical lowering, device
  profiling, and solver implementation details must stay under the concrete
  runtime namespace such as `runtime/jax/`.
  Method: do the audit/move/delete pass first, without running the full test
  suite after each file. Validate once at the end with `compileall`,
  `tests/unit`, `git diff --check`, `vulture`, and targeted benchmarks only if
  a hot path changes.
  - [x] Public runtime namespace and policy: `runtime/__init__.py`,
    `runtime/policy.py`, and `runtime/execution.py`.
  - [x] Runtime input/output contracts: `runtime/input_contract.py`,
    `runtime/input_payloads.py`, `runtime/output_contract.py`,
    `runtime/recording.py`, `runtime/row_output.py`, and
    `runtime/solver_axon.py`.
  - [x] Runtime-neutral host planning and preparation:
    `runtime/input_planning.py`, `runtime/host_preparation.py`,
    `runtime/group_preparation.py`, `runtime/result_assembly.py`, and
    `runtime/memory_estimates.py`.
  - [x] Benchmarking interface and runtime-specific profiling boundary:
    `runtime/benchmarking.py`.
- [x] Run a final Graphify-guided runtime cleanup pass before closing P12.
  Focus on `src/axonscope/runtime/` and `src/axonscope/runtime/jax/`, plus only
  the direct public boundaries that call into runtime (`simulation.py`,
  `performance.py`, `inspection.py`, and dispatcher/result assembly) when they
  expose duplication or dead routes. Inspect runtime high-noise hubs, thin
  communities, isolated nodes, and parallel modules that look duplicated, then
  confirm candidates with direct usage search, `vulture`, tests, and examples
  before deleting or moving code. The 2026-07-13 `graphify cluster-only` refresh
  wrote `graphify-out/GRAPH_REPORT.md` from commit `8a97deb6` with 8906 nodes,
  20840 edges, 484 communities, 77 thin communities omitted from the report,
  and 1077 isolated nodes; filter it down to runtime cleanup evidence, not a
  repo-wide cleanup pass.
  First pass: Graphify plus usage search found
  `runtime/jax/cable_geometry.py::extracellular_absolute_arrays` unused after
  the host-side `extracellular_runtime_numpy` path took over double-cable
  absolute-array preparation, so the JAX helper was deleted. `shape_bucketing`,
  runtime memory snapshots, recording/output lowering, runtime caches, and
  membrane backend construction remain used or intentionally covered. After the
  edit, `graphify update` rebuilt `graphify-out/GRAPH_REPORT.md` with 8935
  nodes, 20942 edges, and 467 communities.
  Second pass: removed runtime helpers whose only live callers were tests,
  legacy benchmarks, or replaced dense-preparation paths:
  `build_membrane_backend_from_axon`,
  `precompute_extracellular_potential_mV`,
  `build_vstim_initial_previous_batch`, and the dense direct-footprint Vstim
  helpers (`build_footprint_vstim_batch`,
  `build_footprint_vstim_midpoint_batch`,
  `build_footprint_vstim_initial_previous_batch`, `FootprintEngine`, and their
  private batch-shape helpers). Tests now use active primitives:
  `compile_axon_membrane`/`backend_from_membrane`,
  `build_extracellular_potential_fn` plus
  `sample_extracellular_potential_mV`, and
  `build_vstim_midpoint_and_initial_previous_batch`. A follow-up
  Graphify update rebuilt the runtime map with 8918 nodes, 20864 edges, and
  466 communities. The final usage scan found zero public runtime definitions
  whose only remaining callers were tests or legacy benchmarks.
- [x] Define and enforce the runtime input contract before implementing
  `axs.runtime.numpy`: prepared batches must expose one cable formulation, one
  padded `Nx`, a dtype/time grid, typed per-cable solver policy, recording and
  observer plans, intracellular modes, and extracellular modes
  (`zero`, `shared_current`, `scaled_shared_waveform`, `current_table`,
  `dense`). The JAX runner now validates and records a runtime-neutral prepared
  input summary before kernel enqueue.
- [x] Before claiming P12 cleanup has no performance loss, re-run the relevant
  P11 hotpath/realistic benchmark slices for single-cable and double-cable,
  CPU/GPU where applicable, with fresh artifact directories and git metadata.
  Fresh local CPU guard rerun on 2026-07-12 wrote
  `benchmark/results/p12_current_repeats2_single_cpu` and
  `benchmark/results/p12_current_repeats2_double_cpu`. It shows no solver-side
  regression versus the recording-contract gate; remaining P12 optimization
  targets are cold/preparation spans before broader P11/GPU slices can close
  the performance-loss claim.
  Kaggle P100 GPU rerun on commit `aab5384` wrote
  `benchmark/results/p12_final_gpu_single_aab5384` and
  `benchmark/results/p12_final_gpu_double_jt_aab5384`. Later Kaggle P100
  1024-axon runs and targeted P12 optimization runs confirmed the current
  closeout state: single-cable cold improved from
  `benchmark/results/kaggle/20260713_111647_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axsp12-cold-single`
  to
  `benchmark/results/kaggle/20260713_113800_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axsp12-cold-single-hostzero`,
  while warm single-cable remains dispatch-bound and double-cable warm/cold is
  good enough for P12 after rejected trim-static and batch-size experiments.
  Broader P11-style benchmark slices remain future evidence for public speed
  claims, not a blocker for closing P12 runtime cleanup.
- [x] Post-P11 runtime/benchmark backlog:
  continue only the deferred items tracked in
  `docs/architecture/p11_closeout_2026_07_12.md`. Main follow-ups are GPU
  double-cable Triton/tiled-Thomas policy thresholds, shared-waveform/scaled
  extracellular input lowering, adaptive time-chunk policy, GPU dispatch
  scheduling, model/compiler optimizer closeout, dense fallback decisions, and
  NRV validation only when numerical behavior changes. This backlog is
  explicitly deferred out of P12.
- [x] Evaluate targeted GPU kernels for remaining non-solver device-side
  bottlenecks, without turning the whole host/runtime path into Triton:
  first prototype an `extracellular_scaled_shared_waveform` path that writes
  forcing directly in the solver layout, then prototype observer-only
  VmRaster/probe packing that extracts or aggregates on GPU without CPU
  round-trips. Keep this behind the JAX GPU runtime boundary and accept it only
  with before/after stage benchmarks showing that the cost is device-side and
  not just Python/JIT/transfer overhead. P12 did not identify an obvious small,
  safe GPU-kernel promotion; keep this as a future optimization track.
- [x] After the runtime contract, benchmark surface, and hot-path cleanup are
  stable, revisit cold-run optimization separately. Focus on JIT/lowering,
  membrane/runtime preparation caches, pool rebuild costs, and optional
  persistent compilation caches; do not mix cold-start policy decisions into
  the current hot-path cleanup.
  First targeted P12 optimization: uniform stateless Model IR membrane initial
  arrays now use the NumPy interpreter for cold `Vm0`/`gates0` construction.
  Local CPU single-cable `runtime.prepare.membrane_init` dropped from 708.2 ms
  to 3.3 ms in `p12_opt_uniform_init_single_cpu`; keep the broader cold-start
  item open for double-cable, compile/backend, GPU, and persistent-cache work.
  Second targeted P12 optimization: stateless heterogeneous Model IR membrane
  groups now use the same NumPy init path while keeping the JAX backend for
  execution. Local CPU double-cable `runtime.prepare.membrane_init` dropped from
  914.5 ms to 4.6 ms, and `runtime.prepare` from 2005.9 ms to 1211.6 ms, in
  `benchmark/results/p12_opt_heterogeneous_init_double_cpu`. Keep the broader
  cold-start item open for compile/backend, GPU confirmation, dispatch/enqueue,
  and persistent-cache work.
  Third targeted P12 optimization: shared double-cable base runtime now builds
  cable coefficients and extracellular absolute arrays with the runtime-neutral
  NumPy host-preparation helpers before compact JAX materialization. Local CPU
  double-cable `runtime.prepare.base_runtime` dropped from 1209.0 ms to
  337.5-351.0 ms, and `runtime.prepare` from 1211.6 ms to 341.0-353.9 ms, in
  `benchmark/results/p12_opt_host_double_only_double_cpu_serial*`. Remaining
  local costs are mostly enqueue/dispatch, cold membrane compile, and kernel
  state/observer preparation; confirm this path on GPU before claiming the P12
  GPU cold-start target improved.
  Kaggle P100 RSS rerun on commit `569e8d4` wrote
  `benchmark/results/kaggle/20260713_002054_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12-cold-single-rss-569e8d4`
  and
  `benchmark/results/kaggle/20260713_002323_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12-cold-double-rss-569e8d4`.
  Double-cable cold preparation is confirmed on GPU:
  `runtime.prepare.base_runtime` dropped from 2265.5 ms to 652.4 ms and
  `runtime.prepare.membrane_init` from 936.2 ms to 4.7 ms versus the previous
  `aab5384` gate. Warm GPU repeat means are not improved yet
  (`double-cable curve.simulate` 23.1 ms to 26.6 ms), so keep the next P12 GPU
  optimization focused on dispatch/enqueue/input/observer overhead. Matching
  `--memory-trace device` probes were run too, but use them for memory evidence,
  not timing claims, because device-memory snapshots dominate small warm spans.
  Kaggle P100 1024-axon rerun accepted NumPy cable preparation for single-cable
  too: commit `1a1183e` changed single-cable from `cable_runtime_source=jax` to
  `cable_runtime_source=numpy`, and
  `benchmark/results/kaggle/20260713_003301_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12-1024-single-numpy-1a1183e`
  shows `runtime.prepare.base_runtime` dropping from 1265.9 ms to 699.6 ms
  versus the JAX baseline
  `benchmark/results/kaggle/20260713_002938_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12-1024-single-jax-24efe36`,
  while warm `curve.simulate` changes only from 36.35 ms to 37.29 ms. The
  matching 1024-axon double-cable NumPy run is
  `benchmark/results/kaggle/20260713_003531_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12-1024-double-numpy-1a1183e`;
  its all-phase time is still dominated by `kernel.enqueue` and
  `kernel.dispatch_jax`. Final targeted P12 cold runs then kept the single-cable
  wins and rejected double-cable trim-static/batch-size tweaks as not worth
  carrying. Persistent compile caches, larger policy sweeps, and specialized
  double-cable JITs remain future work.

### P13 - Evaluation Du Time Chunk

- [x] Evaluate `time_chunk_steps` as a first-class runtime/performance policy
  instead of a hidden observer-only default.
- [x] Benchmark observer-only VmRaster routes across `time_chunk_steps=None`,
  128, 256, 512, 1024, and full-duration chunks for representative
  single-cable and double-cable groups; separate cold compile/lowering,
  warm solve, host enqueue/dispatch, `kernel.wait`, observer finalization, and
  memory/RSS.
- [x] Compare CPU and GPU behavior separately. On GPU, prioritize whether
  chunking improves memory pressure without making warm runs less solver-bound;
  on CPU, check whether chunking mostly adds Python/JAX dispatch overhead.
  Initial P13 matrix on 2026-07-13 covered observer-only single/double cable
  with `Naxon={1,64,1024}` and policies
  `default,unchunked,128,256,512,1024`. Local CPU results are in
  `benchmark/results/p13_time_chunk_cpu_matrix`. Kaggle CPU results are in
  `benchmark/results/kaggle/20260713_131927_time_chunk_sweep_quick_cpu_cpu_axonscope-p13-time-chunk-cpu-a6a1404`.
  Kaggle GPU single-cable results are in
  `benchmark/results/kaggle/20260713_131849_time_chunk_sweep_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p13-time-chunk-gpu-a6a1404`;
  the first combined GPU run intentionally failed for double-cable because
  `jax-triton` was missing. The corrected double-cable GPU rerun with
  `jax-triton` is in
  `benchmark/results/kaggle/20260713_132533_time_chunk_sweep_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p13-time-chunk-gpu-double-jt-8d092fc`.
  Decision: keep the observer/VmRaster default simple and use `128` everywhere.
  The measured effect is globally weak, `512` is not a clear win, and `128`
  gives a conservative bounded chunk without adding a meaningful warm overhead.
- [ ] Evaluate dense Vm/full recording separately from VmRaster. Full Vm may
  still need output/assembly optimizations, but it should not drive the
  default observer-only chunking policy.
- [x] Decide the public/internal policy knobs: keep the default simple as
  `DEFAULT_OBSERVER_TIME_CHUNK_STEPS = 128` for observer/VmRaster routes. Do
  not expose a new public execution-policy option or adaptive default yet.
- [x] Keep progress logs explicit: report time chunks as time chunks, and
  reserve wait/synchronization wording for the final JAX/device wait.
  Verified on 2026-07-13: dispatcher progress uses `solving time chunks` for
  chunk callbacks, while final synchronization remains `waiting for JAX work`
  and benchmark timing keeps `kernel.wait` as the explicit device wait span.

### P3 - Documentation And Examples

- [x] README rewritten after post-P7 stabilization.
- [x] Manual cleanup of `docs/`, `GUIDELINES.md`, and `AGENTS.md`.
- [x] Public examples audited after benchmark flattening.
- [ ] Write real notebook tutorials under `examples/tutorials/` following the
  indexed mini-course sequence.
- [ ] Add a didactic basic example for high-frequency block after block
  detection exists, so the example distinguishes propagation, activation
  failure, and true conduction block.
- [ ] Prepare proper Sphinx documentation.
- [ ] Do/update all public docstrings.

## Future Phases

### Unsorted Future Work

These items are intentionally not ordered or scoped into a phase yet.

- [ ] Continue hardening NRV integration only where the package contract is
  stable: keep geometry construction in `examples/with_nrv` or benchmarks, and
  promote future pieces only when they do not duplicate the canonical
  sampled-footprint path already in `axonscope.integrations.nrv`.
- [ ] Studies: callable threshold curves, block-threshold curves, recruitment
  curves, conduction validation, parameter sweeps, reuse policies, retention
  policies, and study results.
- [ ] Serialization: final schemas, typed serialization, and persistence
  strategy.
- [ ] Work on HPC integration.
- [ ] Work on FEM footprint integration, see
  `ideas/fem_axon_gpu_coupling_design.md`. Start with the CPU/NRV path before
  thinking GPU FEM: split benchmarks into FEM solve, first footprint, cached
  footprint sampling, and AxonScope solve; cache reusable FEM field bases;
  avoid repeated point-location by introducing an axon embedding/projection
  representation; then choose between full precomputed footprints, chunked
  projection, and future fused projection-solver paths by memory budget.
- [ ] Add a KES block example after implementing the missing support needed for
  that workflow, including filtering and the remaining block-analysis pieces.
- [ ] Reorganize `src/`, especially the Python modules still living at package
  root.
- [ ] Revisit the caching strategy: generate artifacts only for the requested
  runtime, keep cache state clean, and decide whether built-in models should be
  built on first call and/or at package install time.
- [ ] Revisit double-cable Triton input/output aliasing together with a serious
  dispatch/copy optimization pass. The retained three-repeat warm medians are
  `run_pool=437.5 ms`, `dispatch_jax=82.7 ms`, and `wait=28.7 ms` without
  aliases versus `565.5/113.5/9.75 ms` with aliases. The wait reduction is
  substantial, but `dispatch+wait` and end-to-end time currently regress.
  Profile XLA buffer assignment/copy insertion and Nsight device activity;
  sweep larger `Nx`, batch sizes, and solver-heavy durations; test alias
  subsets, truly ephemeral custom-call operands, and outer-JIT donation where
  ownership permits it. Retain aliasing only if total `run_pool` and
  `dispatch_jax + wait` improve with identical numerical results, rather than
  accepting a synchronization-boundary shift as solver speedup.
- [ ] Evaluate JAX's native persistent compilation cache for non-Triton JAX
  routes, then enable it under `.axonscope_cache/runtime/jax/xla` only if fresh
  evidence is positive. Cover single-cable CPU/GPU, CPU double-cable Thomas,
  and any future pure-JAX GPU fallback; also test whether it usefully layers on
  top of the Triton compiled-call cache for the complete XLA executable.
  Configure the cache before the first JAX compilation and benchmark a true
  miss followed by a hit in separate processes, splitting trace, lower, XLA
  compile, first execution, and end-to-end cold time. Decide minimum compile
  time/entry size, maximum size/LRU retention, explicit clean/disable behavior,
  trusted-local-directory policy, cache diagnostics, and Kaggle/HPC sharing.
  Prefer JAX's version/device/HLO-keyed executable cache over an AxonScope
  reimplementation, while retaining AxonScope-owned policy and benchmark
  guards around its location and lifecycle.
  The official JAX cache key covers non-optimized HLO, jaxlib version,
  relevant XLA flags, device count/topology (GPU model), compression, and an
  optional custom hook. Runtime values with unchanged abstract shapes/static
  arguments therefore reuse the same executable; a changed HLO/shape/static
  argument produces another complete executable rather than incrementally
  recompiling only one subgraph. Evaluate
  `jax_persistent_cache_min_entry_size_bytes=-1` and compare the default
  per-fusion autotune cache with `jax_persistent_cache_enable_xla_caches=all`
  and `xla_gpu_kernel_cache_file`, including whether the latter usefully
  persists GPU kernel/PTX work around the Triton custom call. Enable
  `jax_explain_cache_misses` in diagnostic benchmark runs and keep the cache in
  a trusted local/shared directory only.
- [ ] After CPU/GPU optimization work is complete, run a broad cleanup pass to
  remove unused code and verify contracts. Treat `runtime/jax` as the first
  completed target, and use Graphify to guide the wider pass.
- [ ] Test GPU async scheduling. Candidate grouping contract: one batch has the
  same model, same padded `Nx`, potentially variable diameter, potentially
  variable footprint, and the same stimulus; incompatible rows form separate
  groups, then groups may be launched with async GPU scheduling if benchmarks
  show it helps.
- [ ] Remove public API surface that is unused or not documented in advanced
  examples.
- [x] Validate example performance for `examples/basic/06_activation_velocity.py`,
  `examples/basic/07_threshold_vs_diameter.py`,
  `examples/basic/08_recruitment_curve_population.py`, and
  `examples/with_nrv/01_synthetic_fascicle_geometry.py` on CPU and Kaggle GPU.
  Record cold and warm timings with global performance counters, then add
  benchmark coverage to identify any remaining bottlenecks before treating
  these examples as stable perf gates.
  - [x] Close the `with_nrv/01_synthetic_fascicle_geometry.py` full-population
    validation slice. On 2026-07-13, commit `0f5860f` added deterministic NRV
    seeding plus `recruitment_result.json` export. Full example-01 runs used
    100 axons per fascicle, 193 AxonScope axons, 21 amplitudes, `duration=3 ms`,
    `dt=0.001 ms`, and `vm_raster/full`. Artifacts:
    `benchmark/results/with_nrv_examples_local_cpu_fullpop_seed0_20260713`,
    `benchmark/results/kaggle/20260713_193414_with_nrv_examples_quick_cpu_cpu_axonscope-with-nrv-01-cpu-fullpop-seed0-0f5860f`,
    and
    `benchmark/results/kaggle/20260713_192746_with_nrv_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-with-nrv-01-gpu-fullpop-seed0-0f5860f`.
    CPU routes stayed on double-cable `thomas`; GPU used double-cable
    `jax_triton_loop_xb`. Local CPU versus Kaggle P100 GPU ratios were
    `6.4x` for `protocol.recruitment_sweep`, `7.5x` for `simulation.run_pool`,
    and `9.3x` for post-compile per-amplitude values. Kaggle CPU versus Kaggle
    P100 GPU ratios were `12.1x`, `14.3x`, and `17.8x`, respectively. Final
    recruitment at 300 uA matched at `120/193`; CPU/GPU differed only by a few
    near-threshold activation decisions, so strict cross-backend checks should
    allow one amplitude step around threshold crossings.
  - [x] Close the `examples/basic/06`, `07`, and `08` validation slice. On
    2026-07-13, local CPU artifact
    `benchmark/results/basic_examples_local_cpu_validate_20260713` and Kaggle
    P100 GPU artifact
    `benchmark/results/kaggle/20260713_203352_basic_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-basic-06-07-08-gpu-7f9b781`
    ran `benchmark/examples/basic_examples.py --examples 06,07,08 --warmups 0
    --repeats 1`. Cold/warm wall timings were: `06` CPU `27.12/20.86 s`,
    GPU `15.60/3.14 s` (`1.7x/6.6x`); `07` CPU `16.63/9.82 s`, GPU
    `9.92/4.65 s` (`1.7x/2.1x`); `08` CPU `6.78/4.12 s`, GPU `5.16/2.73 s`
    (`1.3x/1.5x`). GPU `kernel.wait` is already tiny for `06` and `08`, so
    remaining bottlenecks are mostly JAX/Python dispatch, enqueue, result
    assembly, and example/protocol overhead rather than raw solver time.
  - [x] Revalidate these example gates after the final P12 warm-path cache
    changes at commits `0b76cf7` and `222c504`. Artifacts:
    `benchmark/results/basic_examples_local_cpu_post_p12_20260714`,
    `benchmark/results/kaggle/20260714_114908_basic_examples_quick_cpu_cpu_axonscope-basic-06-07-08-post-p12-cpu`,
    `benchmark/results/kaggle/20260714_021129_basic_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-basic-06-07-08-post-p12-gpu`,
    `benchmark/results/with_nrv_examples_local_cpu_post_p12_fullpop_clean_20260714`,
    and
    `benchmark/results/kaggle/20260714_115613_with_nrv_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-with-nrv-01-post-p12-fullpop-gpu`.
    Kaggle CPU/P100 wall speedups for the committed `06/07/08` examples were
    cold `1.73x/1.67x/1.27x` and warm `6.59x/2.27x/1.44x`. The current
    uncommitted `08` timestep (`dt=0.005 ms`) was validated separately through
    benchmark-only commit `fd09837`: CPU/GPU warm wall was `7.52/2.96 s`
    (`2.53x`), and both routes produced counts
    `6 20 47 67 86 107 130 135`. Full-population `with_nrv/01` improved versus
    the prior artifacts: local CPU/GPU `protocol.recruitment_sweep` became
    `204.66/27.81 s` (`7.36x`) and `simulation.run_pool` `201.92/24.78 s`
    (`8.15x`); median post-compile amplitude time was approximately
    `9.59/0.986 s` (`9.73x`). Final recruitment matched at `120/193`; three
    intermediate amplitudes differed by one near-threshold axon.
- [x] Check whether `recruitment_sweep` can batch amplitude values into one
  expanded compatible run when the pool, model shapes, footprints, and stimulus
  timing are shared. Use `examples/basic/08_recruitment_curve_population.py` as
  the first benchmark target; compare against the current sequential
  per-amplitude observer-only path.
  - [x] Add a native-pool opt-in for observer-only recruitment via
    `batch_amplitudes=True`. It builds an expanded value-major
    `amplitude x axon` AxonScope pool without `deepcopy`, keeps original rows
    unmutated, and validates against the sequential path on real single-cable
    and double-cable/MRG point-source probes.
  - [x] Fix double-cable observer-only compact factorized Vext so row-specific
    waveform scales are applied in the VmRaster path, matching the existing
    dense/probe Vm route. Before this fix, naive expanded MRG pools did not
    reproduce sequential activation counts because `current_row_scales` were
    ignored by the double-cable observer path.
  - [x] Keep `examples/basic/08_recruitment_curve_population.py` on the
    default sequential amplitude path for now. A full native amplitude batch is
    functionally correct but slower for this example on Kaggle P100 at commit
    `9c87208`: `20.01/5.51 s` cold/warm versus the prior sequential
    `5.16/2.73 s`. Warm full-batch overhead is dominated by
    `dispatch.build_plan` (`2.64 s`) and `protocol.sweep.build_amplitude_pool`
    (`0.59 s`), while `kernel.wait` stays around `5 ms`.
    Follow-up at commit `a572742` fixed the main `dispatch.build_plan` cost by
    sharing immutable/source axon objects across native amplitude clones instead
    of shallow-copying each axon. The example remains sequential until the
    public default policy is decided.
  - [x] Add configurable amplitude pool chunking for native recruitment
    batching, e.g. `amplitude_batch_size=1`, `10`, `20`, or `None/full`, so
    large sweeps can choose between sequential amplitudes, medium
    `fibers x amplitude_chunk` pools, and fully expanded
    `fibers x all_amplitudes` pools.
  - [ ] Finalize and analyze compact observer results once per native amplitude
    batch instead of constructing one public `AxonSimulationResult` and
    decoding VmRaster activity after every amplitude. Keep packed/device-local
    outputs until the batch completes, then preserve value ordering, progress
    rows, failure attribution, summary-only retention, and the
    `amplitude_batch_size=1` behavior. Benchmark result-finalization time,
    device-to-host transfers, peak device/host memory, and end-to-end CPU/GPU
    time for sizes `1/10/20/full` before making it the default.
  - [x] Benchmark native amplitude chunk sizes on CPU and Kaggle GPU for
    single-cable and double-cable pools. Report cold/warm timings, compile
    overhead, `kernel.dispatch_jax`, enqueue, wait, result assembly, peak
    memory, and activation-count equivalence against the sequential path.
    Added `benchmark/run.py --script recruitment_amplitude_batch` at commit
    `63cc687`, based on the mixed single/double-cable population from
    `examples/basic/08`. Baseline local CPU artifact
    `benchmark/results/recruitment_amplitude_batch_local_cpu_baseline_20260713`
    showed warm sequential `4.01 s`, `1` `7.85 s`, `10` `7.42 s`, `20`
    `7.29 s`, and full `7.36 s`; native chunks were dominated by
    `dispatch.build_plan` around `2.65-3.25 s`. Baseline Kaggle P100 artifact
    `benchmark/results/kaggle/20260713_212454_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-recruitment-amplitude-batch-gpu-63cc687`
    showed warm sequential `2.61 s`, `1` `5.95 s`, `10` `5.66 s`, `20`
    `5.62 s`, and full `5.75 s`, with native chunks again dominated by
    `dispatch.build_plan` around `2.80-2.93 s`.
  - [x] Optimize the first native amplitude-batching bottleneck. Commit
    `a572742` changed native amplitude clones to reuse the source axon object
    while keeping each `AxonInstance` separate and mutable only through its
    stimulation. Local CPU artifact
    `benchmark/results/recruitment_amplitude_batch_local_cpu_shared_axon_20260713`
    improved warm timings to sequential `4.11 s`, `10` `4.43 s`, `20`
    `4.41 s`, and full `4.22 s`; `amplitude_batch_size=1` stays slow at
    `7.80 s` because it still builds eight separate native pools/plans. Kaggle
    P100 artifact
    `benchmark/results/kaggle/20260713_213206_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-recruitment-amplitude-batch-gpu-a572742`
    improved warm timings to sequential `3.14 s`, `10` `2.58 s`, `20`
    `2.57 s`, and full `2.56 s`; all policies matched activation counts
    `6 18 41 65 82 101 126 135`.
  - [x] Reuse spatial cohort preparation and VmRaster plans across equivalent
    groups and amplitude-only cohort refreshes. Commit `73cf2ea` separates the
    immutable spatial cache token from current axon, solver-axon, and
    stimulation rows. Kaggle P100 artifact
    `benchmark/results/kaggle/20260713_223515_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-spatial-cache-73cf2ea`
    reduced warm `simulation.run_pool` from `768.95` to `441.87 ms` for the
    sequential policy and from `1624.02` to `923.16 ms` for
    `amplitude_batch_size=1`. Corresponding `inputs.positions` totals dropped
    from `78.52` to `4.57 ms` and from `566.79` to `17.43 ms`; `observer.plan`
    dropped from `15.15` to `1.97 ms` and from `76.23` to `2.43 ms`. Every
    policy retained activation counts `6 18 41 65 82 101 126 135`.
  - [x] Reuse native amplitude work pools across equal-sized chunks. Commit
    `19c1b5f` keeps one resettable pool per chunk size, so each amplitude still
    starts from the source-row state while stable row identities let the
    dispatcher and runtime caches hit. Kaggle P100 artifact
    `benchmark/results/kaggle/20260713_224541_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-work-pool-19c1b5f`
    reduced warm `amplitude_batch_size=1` wall time from `5.72` to `2.52 s`
    versus the spatial-cache-only run and from `6.85 s` versus the original
    native baseline. `dispatch.build_plan` fell from `3096.88` to `481.06 ms`,
    `runtime.prepare` from `499.62` to `59.70 ms`, and
    `simulation.run_pool` from `923.16` to `421.70 ms`; the trace records one
    dispatch-plan miss followed by seven hits. All policies retained activation
    counts `6 18 41 65 82 101 126 135`. Remaining generic protocol cost is
    mostly the required user update work (`63.20 ms` initial pool plus
    `441.81 ms` refreshes for 1600 row-amplitude updates), while the benchmark
    harness itself spends about `0.94 s` constructing the population outside
    `recruitment_sweep`.
  - [x] Reduce true first-process recruitment cold preparation without changing
    solver paths. Commit `cb418ae` initializes structural gated/leak stack gates
    through the existing NumPy Model IR interpreter instead of triggering an
    eager JAX gate initialization; commit `f7ea77b` memoizes membrane signatures
    by object identity while one dispatch plan is built. Local CPU cold wall
    fell from `8.04 s` to `6.17 s`, with `runtime.prepare.stack_membrane`
    dropping from `912.10` to `69.20 ms` after the first change and profiled
    first-plan construction dropping from `0.77` to `0.29 s` after the second.
    Kaggle P100 artifact
    `benchmark/results/kaggle/20260713_231018_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-cold-signatures-f7ea77b`
    improved sequential cold `recruitment_sweep` from `10.38` to `9.74 s`
    despite run-to-run double-cable JIT variance, with `runtime.prepare`
    `1.79 s` to `0.95 s` and `dispatch.build_plan` `377.04` to `141.19 ms`.
    Sequential warm improved from `2.50` to `2.20 s` wall and from `1.47` to
    `1.17 s` inside `recruitment_sweep`; activation counts remained
    `6 18 41 65 82 101 126 135`. A local two-process probe also showed a JAX
    persistent-cache hit reducing cold wall from `6.53` to `4.75 s`, but keep
    persistent cache location, invalidation, and retention as a separate policy
    decision rather than enabling it implicitly.
  - [x] Make source-model runtime code generation lazy per model and runtime
    target. Source compiler v15 keeps one content-addressed model directory,
    lets JAX request only `jax_model.py`, and can add `numpy_model.py` later
    without rewriting the existing JAX artifact. Runtime program hashes no
    longer depend on which cache targets happen to be present.
  - [x] Move stateless source-model gate rates, Q10 factors, conductances, and
    reversal terms into generated `jax_model.py` functions. JAX consumes those
    functions instead of rebuilding their Model IR expression trees; the
    structural gated/leak backend now exposes model-agnostic batch capabilities
    for gate and membrane terms rather than selecting a named model family.
  - [ ] **Next important step after the current cold/warm optimization pass:**
    make each generated runtime module, starting with `jax_model.py`, the
    autonomous source of every model-specific fact required by that runtime.
    Generate parameter defaults; names and units; gate, membrane-state,
    current, observable, and diagnostic metadata; gate-update policy;
    auxiliary-state definitions and initialization; stateful prepare/finalize
    functions; recording/diagnostic contracts; runtime hashes/signatures; and
    the compact metadata needed for result labels and runtime inspection.
    After a cache hit, JAX execution must load this generated contract without
    reconstructing `JaxMembraneProgram` from Model IR or evaluating Model IR
    expressions. Keep Model IR only as a compiler artifact for source
    validation, optimization, composition, generated-code inspection, and
    NumPy/reference validation; composition or inspection may consume the IR at
    compile time, but their runtime-specific output must also be emitted into
    the generated module. Validate stateless and stateful built-ins, composite
    membranes, parameter overrides, recording labels, numerical equivalence,
    cache reuse, and cold/warm performance before removing the JAX Model IR
    fallback.
  - [x] Capture the production double-cable GPU cold JIT as separate
    trace/lower/compile/first-execution phases on Kaggle, and compare the
    generated-term plus batch-capability route against the previous recruitment
    baseline before retaining any further kernel specialization. P100 artifact
    `benchmark/results/kaggle/20260713_235628_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-model-codegen-eaa292a`
    measured `1.093/3.874/0.884/0.171 s`, respectively (`6.022 s` total), so
    lowering is 64% of the production cold JIT. The controlled row-wise artifact
    `benchmark/results/kaggle/20260714_000106_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-model-codegen-row-ad0ff51`
    measured `1.614/5.797/0.711/0.280 s` (`8.402 s` total). Keep the
    capability-based batch path: it cut measured JIT time by 28%, retained
    activation counts `6 18 41 65 82 101 126 135`, and left warm recruitment
    effectively flat (`1.238 s` batch-native versus `1.227 s` row-wise).
  - [x] Finish the retained double-cable Triton kernel pass without changing
    the runtime organization. Commits `40ccc47`, `a3ff29e`, and `4f70a5a`
    reuse scan-static linear terms, fuse forward/backward Thomas passes into
    one custom call, and reuse two outputs as internal Thomas workspaces. Dense
    P100 validation retained `1.439e-7` maximum absolute error and one StableHLO
    custom call. The retained three-repeat no-alias artifact
    `benchmark/results/kaggle/20260714_004954_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-triton-noalias-warm-4f70a5a`
    measured about `440.0 ms` warm `simulation.run_pool`, `82.8 ms`
    `kernel.dispatch_jax`, and `28.8 ms` `kernel.wait`. The official
    input/output-alias experiment in `dabf920` was rejected and reverted by
    `d40d21e`: its matching artifact increased those spans to about
    `569.3/113.5/10.0 ms`, moving work into dispatch despite the lower wait.
  - [x] Persist the supported double-cable Triton TTIR-to-PTX result under
    `.axonscope_cache/runtime/jax/triton`. Commit `cd97bfd` stores the compressed
    `TritonKernelCall` with a checksum and a content key covering source,
    shapes/dtypes, grid, metaparameters, compute capability, and JAX/JAXlib/
    jax-triton/Triton versions; unsupported jax-triton versions use the normal
    upstream lowering. The fresh-process P100 replay
    `benchmark/results/kaggle/20260714_010800_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-triton-cache-cd97bfd`
    recorded a real miss then hit for the same `[Nx=22, B=800]` call. Lowering
    fell from `3.962` to `0.063 s` (`62.5x`), instrumented cold JIT from `6.104`
    to `1.551 s` (`3.94x`), and full wall time from `12.075` to `7.186 s`.
    Recruitment counts stayed `6 18 41 65 82 101 126 135`; the independent
    dense solve passed at `1.439e-7` maximum absolute error. The reusable cache
    artifact is only `13,376` bytes for this signature.
  - [x] Test async JAX scheduling across independent dispatcher groups. Commit
    `0bf9984` added an internal enqueue/finalize scheduler and the
    `dispatcher_group_scheduling` benchmark. Kaggle P100 warm runs showed no
    gain: 256 axons were `198.7 ms` sync versus `213.8 ms` async, and 1024
    axons were `606.6 ms` sync versus `611.5 ms` async. Keep synchronous group
    execution as the default; revisit async only for workloads with several
    independent solver-heavy groups and measured device idle time.
  - [ ] Benchmark async scheduling specifically when heterogeneous workloads
    force several independent dispatch groups. Cover incompatible membrane
    models, cable formulations/Nx shapes, and genuinely different stimulus
    temporal signatures (amplitude-only scaling may remain in one factorized
    group). Sweep 2/4/8 groups and light versus solver-heavy durations on CPU
    and Kaggle GPU; compare sync/async warm and cold totals, enqueue/flush/wait,
    peak memory, device idle time, deterministic result ordering, and numerical
    equivalence. Keep async opt-in unless it gives a clear end-to-end gain for
    these forced multi-group cases without excessive pending-device memory.
- [ ] Implement Nav1.x-family and other Markov-based membrane models.
- [ ] Re-check each built-in model against the NRV implementation; some details
  may have been lost during model translation.
- [ ] Finish missing membrane models, including Gaines and Markov families.
- [ ] Before v1, make a full list of Python files and package organization.
- [ ] Before v1, inspect every function/type/module and delete anything not
  called outside tests, using `vulture` as one input.
- [ ] Before v1, check where retained code is used, what it is for, whether it
  respects contracts, and whether it duplicates another implementation. Use
  Graphify to guide the pass.
- [ ] Before v1, clean `pyproject.toml`, including optional GPU extras for
  CUDA/Triton.
- [ ] Test Apple Metal acceleration with `jax-mps`:
  https://github.com/tillahoffmann/jax-mps

### P8 - Future Bonus NumPy/SciPy Reference Solver Runtime

This is intentionally not the next implementation phase. The NumPy/SciPy
runtime remains valuable as a future reference/debug backend, but only after
the model/compiler surface and the current JAX runtime contract are clean
enough. The goal is a real reference solver runtime, not a JAX-backed
compatibility path.

- [ ] Keep `axs.runtime.numpy` reserved/non-executable until this phase reaches
  executable behavior through the same `AxonSimulation(...).run()`,
  `.estimate()`, and `.inspect()` lifecycle as JAX.
- [ ] Do not start implementation before P10 model/compiler cleanup and P11
  realistic JAX solver benchmarking/optimization are stable enough that the
  reference runtime has a clean contract to implement.
- [ ] Define the first supported scope explicitly: scalar/tiny simulations
  first, not population batching, GPU parity, or a second public workflow.
- [ ] Implement the reference solver behind the backend execution facade, using
  Model IR semantics and SciPy/NumPy numerical primitives rather than JAX
  membrane backends.
- [ ] Use the tridiagonal Crank-Nicholson solver path as the first numerical
  primitive for single-cable tiny simulations; choose SciPy banded/sparse
  helpers where they make the implementation clearer and deterministic.
- [ ] Decide and document the v1 model/input subset: single-cable first,
  intracellular current, sampled extracellular footprints, recording modes,
  observer support, and whether double-cable waits for a later slice.
- [ ] Add cross-backend validation against JAX on small deterministic cases:
  Vm traces, activation/block/latency observers, thresholds, probe recordings,
  retained membrane recordings, and model-step equivalence.
- [ ] Wire `ExecutionPolicy(runtime=axs.runtime.numpy)` only after executable
  behavior, examples, docs, estimates, inspection records, and tests exist.
- [ ] Document when to use the reference runtime: debugging tiny simulations,
  semantic validation, backend comparison, and numerical regression tests;
  document when not to use it.

## Completed Phase Summary

Detailed completed ledgers live in the archive and referenced architecture
docs. Keep only high-level state here.

- P0-P6: public API cleanup, one simulation workflow, protocols/results/views,
  examples-as-docs, inspection/runtime reports, validation policy, and
  backend/lowering cleanup are complete for the current JAX path.
- P7: class-based public membrane models, source compiler, generated JAX/NumPy
  model-step artifacts, generated-code cache/reporting, direct
  `JaxMembraneProgram` execution, and old membrane-stack deletion are complete.
- P9: cold-run micro baseline, scalar/batch span normalization, explicit
  hotpath chunk controls, and closeout decisions are recorded.
- P10: model/compiler cleanup and optimizer prep are complete enough for the
  current runtime work.
- P11: benchmark reset, JAX solver optimization, large-population Triton
  exploration, solver-engine flattening, single/double-cable cartography, and
  runtime cleanup closeout are complete for the current pass.

## Key References

- Architecture reference: `GUIDELINES.md`
- Agent guide: `AGENTS.md`
- Full pre-cleanup TODO archive:
  `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`
- P11 closeout:
  `docs/architecture/p11_closeout_2026_07_12.md`
- Solver policy cleanup:
  `docs/architecture/p11_solver_policy_cleanup_decisions_2026_07_11.md`
- P12 runtime contract:
  `docs/architecture/p12_runtime_contract_2026_07_12.md`
- P12A runtime audit:
  `docs/architecture/p12a_jax_runtime_audit_2026_07_12.md`
- P12B runtime/JAX cleanup:
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md`
- Benchmark surface map: `benchmark/README.md`
