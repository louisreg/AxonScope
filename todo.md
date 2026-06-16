# AxonScope TODO

Living operational roadmap for AxonScope documentation, API cleanup, examples,
benchmarks, solver/backend work, and Phase 8+ study APIs.

Read this file at the start of cleanup/API/performance work. Add newly
discovered mismatches here, and check items only after code, docs, examples,
and relevant tests have been verified.

## How To Use This File

- `GUIDELINES.md` is the master architecture and product-boundary reference.
  Update it only when the target philosophy, product boundary, or architecture
  changes.
- `agent.md` captures project working rules for future agents. Keep it aligned
  when the workflow or standing policy changes.
- `todo.md` is the step-by-step execution plan. It should stay actionable,
  mostly flat, and free of long benchmark prose unless the evidence changes a
  current decision.
- Source, tests, and runnable examples remain the truth for current behavior.
- AxonScope is pre-release: prefer clean breaking changes over compatibility
  shims, and delete superseded paths once replacements are in use.

## Current Snapshot

Updated on 2026-06-15.

| Area | Status | Notes |
| --- | --- | --- |
| Phases 0-7.5 | Done | Guardrails, object model, typed contracts, preparation, JAX boundary, canonical pool results, analysis layer, performance estimates, and first solver-side observers are implemented for the current public layer. |
| Phase 7.6 | In progress | Evidence and targeted hotpath cleanup before broad Phase 8 APIs. |
| Phase 7.6.4 | In progress | Experimental pseudo-double-cable / pseudo-MRG validation; physiology first, GPU speed second. |
| Phase 7.7 | Not started | Re-read `GUIDELINES.md`, then clean stimulation/placement APIs before studies. |
| Phase 7.8 | Not started | Examples learning-path cleanup after API cleanup. |
| Phase 8 | Not started | Callable studies, reuse policies, retention policies, and study results. |
| Phase 9 | Not started | Final serialization schemas and NumPy/reference backend validation. |

Latest verified non-NRV unit run:

- [x] 2026-06-15 after compact dispatch cohort cleanup:
  `314 passed, 1 skipped`.

Current benchmark/evidence headline:

- [x] Phase 7.6 initial evidence gate is complete.
- [x] Homogeneous single-cable observer-only runs avoid retained Vm traces with
  `Recording.none()` and compact `result.observations`.
- [x] GPU observer-only `n=1000`, `duration=10 ms`, `dt=0.01 ms`,
  `Nx=51` improved from `673.4 ms` to `176.8 ms` after sparse current-clamp,
  zero-field, compact-cohort, and pulse-vectorization cleanups.
- [x] Local warm path matrix at `n=100`, `duration=2 ms`, `dt=0.02 ms`,
  target `51` compartments: single-cable intracellular `89.9 ms`,
  single-cable point-source extracellular `76.0 ms`, MRG double-cable
  extracellular `176.0 ms`.

## Immediate Queue

Work should start here unless the user asks otherwise.

- [x] Phase 7.6.1: finish the benchmark evidence matrix.
- [x] Phase 7.6.2: attack memory-transfer and long-run execution.
- [ ] Phase 7.6.3: later full backend/solver-choice phase.
- [ ] Phase 7.6.4: experimental pseudo-double-cable / pseudo-MRG validation.
- [ ] Phase 7.7: re-read `GUIDELINES.md`, then clean stimulation and
  placement APIs before Phase 8.
- [ ] Phase 7.8: clean the examples learning path after the API cleanup.
- [ ] Phase 8: add callable studies, reuse policies, retention policies, and
  study result containers.
- [ ] Phase 9: finalize serialization schemas and add reference-backend
  validation.
- [ ] Keep current Phase 5-7.6 changes uncommitted until the user asks for a
  commit or an explicit checkpoint requires it.

## Phase 7.6.1 Benchmark Evidence Matrix

Goal: make intra/extra, single/double-cable, recording policy, observer policy,
cold/warm, and solver-only behavior first-class benchmark evidence before more
optimization.

- [x] Add `path_comparison_matrix` for controlled intra/extra and
  single/double evidence.
- [x] Keep MRG/double-cable measurements clearly labelled; do not mix them with
  pure formulation comparisons unless model, protocol, and output policy are
  controlled.
- [x] Add same-output-policy single-cable versus double-cable comparison rows.
- [x] Add explicit cold/warm annotations in hotpath outputs:
  `timing_mode`, `warmup_count`, and `simulation_labels`.
- [x] Extend `path_comparison_matrix` to cover center/probes/full/observer-none
  recording policies on controlled single-cable intracellular and point-source
  extracellular rows, with explicit comparison axes in workload metadata.
- [x] Add typed `ExtracellularFootprint` / `ExtracellularDrive` execution rows:
  `typed_footprint_drive_matrix` compares analytical-context lowering against
  typed-drive lowering, verifies identical dense `Vstim`, and executes the
  typed-drive lowered input through the single-cable backend.
- [x] Add solver-only or precomputed-input workloads that bypass dispatch
  planning and input materialization: `solver_only_precomputed`.
- [x] Add JAX compilation diagnostic mode for benchmark runs:
  `--jax-log-compiles` enables `jax.config.update("jax_log_compiles", True)`
  and records the setting in manifest/session metadata.
- [x] Run and analyze warm path-matrix traces at representative local sizes for
  recording full/center/probes/none and observer off/on comparisons.
- [x] Compare analytical point-source lowering against typed footprint/drive
  lowering.
- [x] Keep basic examples unchanged for pedagogical output, but preserve their
  benchmark notes: example 06 is simulation dominated, example 07 is first-call
  preparation plus repeated MRG threshold runs, and example 08 has a large
  first population compile/preparation followed by warm amplitude sweeps.

## Phase 7.6.2 Memory Transfer And Long Runs

Goal: avoid unnecessary `B x Nt x Nx` materialization and transfer, especially
for long runs and study loops.

- [x] Add compact/factorized sparse intracellular current-clamp path for
  homogeneous single-cable batch observer-only runs.
- [x] Vectorize the common one-pulse sparse current-clamp builder.
- [x] Avoid dense zero `Vstim[B,Nt,Nx]` materialization for homogeneous
  single-cable observer-only runs with no extracellular contexts.
- [x] Reduce observer-only `results.split_batch` cost by keeping compact
  population observations batched from the solver backend to the public result
  layer.
- [x] Add `double_cable_extracellular` hotpath workload and local probes.
- [x] Cache double-cable `MembraneRuntime`, `CableRuntime`, and
  `ExtracellularRuntime` by static signatures.
- [x] Add conservative analytical point-source fast path for shared homogeneous
  extracellular contexts, including combined midpoint/previous `Vstim`
  preparation.
- [x] Re-check the same bottleneck map locally for double-cable/MRG-like runs;
  Colab GPU external validation repeat was run as
  `colab_cpu_gpu_kernel_double_cable_extracellular_long_20260615_175837`. GPU
  loses at `n=100`, crosses over at `n=300`, and wins clearly at `n=600`;
  kernel throughput is the dominant difference, while dense `Vstim` lowering
  remains a Phase 8 reuse/factorized-drive target.
- [x] Re-check extracellular stimulation with typed
  `ExtracellularFootprint` / `ExtracellularDrive` lowering.
- [x] Decide solver-side factorized extracellular forcing for this phase:
  defer true in-scan `waveform[Nt] * footprint[B,Nx]` forcing until the Phase 8
  reuse/study contract can pass dynamic drive inputs cleanly. Phase 7.6.2 keeps
  typed-drive dense lowering plus explicit memory estimates instead of adding a
  half-public kernel contract.
- [x] Generalize extracellular preprocessing evidence beyond the conservative
  point-source fast path by adding the typed-drive lowering matrix; keep the
  generic multi-source fallback until a real workload shows a stable stall.
- [x] Special-case absent intracellular stimulation beyond the first
  observer-only fast paths: retained-output single-cable and double-cable batch
  runs now pass no host-materialized dense zero `Iinj[B,Nt,Nx]` into the
  backend and record skipped-shape metadata.
- [x] Add chunked long/gross simulation evidence path: hotpath runner now
  accepts `--time-chunk-steps` and applies it to public simulations and direct
  backend workloads.
- [x] Reuse prepared artifacts in benchmark loops before Phase 8 study APIs:
  `solver_only_precomputed` isolates precomputed runtime/input kernel timing,
  while `footprint_reuse_sweep` remains the public fixed-geometry sweep
  baseline.
- [x] Keep padded single-cable group collapsing as a later backend task until
  row-specific recording selectors and observer masks are supported inside
  padded kernels.

## Phase 7.6.3 Backend And Solver Choice

Status: deferred full phase. Do not treat the current adaptive-PCR cleanup as
the whole backend-choice story; use the evidence below to design a proper
later pass.

Goal: make scalar-ish threshold loops, tiny batches, and double-cable GPU
solver choices efficient without forcing JAX compile/enqueue overhead or the
wrong block solver onto every workload.

Future scope:

- [ ] Define a backend/solver decision matrix covering CPU JAX, GPU JAX,
  SciPy/NumPy reference candidates, Thomas, optimized Thomas, PCR,
  `pcr_soa`, and `pcr_adaptive`.
- [ ] Add a forced Thomas GPU benchmark after implementing an ultra-optimized
  Thomas variant, so GPU Thomas is compared fairly against PCR rather than
  dismissed based on the current scan-heavy implementation.
- [ ] Prototype the ultra-optimized GPU Thomas path for double-cable:
  reduce per-step nested scan overhead, exploit shared coefficients where
  possible, avoid tiny per-row launches/fusions, and keep the implementation
  JAX/XLA-friendly unless evidence justifies a lower-level kernel.
- [ ] Benchmark optimized GPU Thomas against `pcr`, `pcr_soa`, and
  `pcr_adaptive` on the same double-cable observer/extracellular matrix:
  `n=100/300/600/2000`, `duration=10 ms`, `dt=0.01 ms`, `51` compartments,
  trace-free and trace-captured runs.
- [ ] Decide whether `auto` should remain threshold-based, become
  benchmark-calibrated by backend/device, or expose a user-visible
  performance policy.
- [ ] Add advanced examples for solver/backend options once the public
  contract is stable: show `auto`, `thomas`, `pcr`, `pcr_soa`,
  `pcr_adaptive`, and explain when forced choices are diagnostic versus
  production-oriented.
- [ ] Keep benchmark-heavy solver comparisons under `benchmark/`; examples
  should teach options and interpretation, not become timing stress tests.

Backlog and evidence already collected for this phase:

- [ ] Prototype a SciPy backend for tiny batches and scalar-ish workloads using
  SciPy tridiagonal/banded solvers.
- [ ] Benchmark SciPy against JAX CPU/GPU for tiny `B`, short `Nt`, and
  threshold-search loops.
- [ ] Use double-cable Colab evidence as a backend-choice calibration point:
  CPU beats GPU at `n=100` (`465.1 ms` vs `824.9 ms`), GPU crosses over by
  `n=300`, and GPU wins at `n=600` (`1070.1 ms` vs `2485.8 ms`).
- [x] Run matching Colab `kernel_single_cable_extracellular_long` case
  (`point_source_extracellular`, `n=100/300/600`, `duration=10 ms`,
  `dt=0.01 ms`, `51` compartments) to separate single-cable GPU scaling from
  double-cable block-solver scaling.
- [x] Identify first double-cable GPU kernel issue after the single-cable
  comparison: current evidence points to `Nt` scan plus per-step forward/reverse
  `Nx` scans inside `solve_block_tridiagonal_2x2_scalar`, so the GPU only gets
  batch-axis parallelism and does not saturate well at `B <= 600`.
- [x] Re-run Colab `kernel_double_cable_extracellular_long` after the
  double-cable zero-Iinj and shared-coefficient kernel-input specializations to
  measure whether removing dense zero `Iinj[B,Nt,Nx]` and avoiding repeated
  `(B,Nx)` coefficient broadcasts improves GPU throughput. Result: zero-Iinj
  helped `n=600` GPU modestly; shared coefficients helped CPU strongly and
  GPU at `n=300`, but did not fix the GPU scaling ceiling.
- [x] Start backend-aware execution selection with the double-cable block
  solver: `auto` now resolves to adaptive PCR on GPU and Thomas elsewhere,
  and hotpath manifests record both requested and resolved solver choices.
- [ ] Decide whether broader runtime/device/precision planning values remain
  estimates only or start selecting execution backends beyond the double-cable
  block solver.
- [x] Prototype an
  algorithmic GPU solver change rather than more input cleanup: e.g. a batched
  or parallel block-tridiagonal solve for the per-step `2x2` system, then
  compare against the current `solve_block_tridiagonal_2x2_scalar` scan.
- [x] Re-run Colab `kernel_double_cable_extracellular_pcr_long` to compare the
  experimental PCR block solver against Thomas on GPU and CPU at
  `n=100/300/600`. Result: PCR is a strong GPU win (`243/369/688 ms` total at
  `n=100/300/600`, `0.30x/0.43x/0.70x` of the best Thomas total), but a CPU
  regression (`7.2x/11.6x/10.9x` slower than best Thomas).
- [x] Add an `auto` double-cable block-solver policy: keep Thomas for CPU/default
  execution, select adaptive PCR for GPU runs, and expose the resolved choice in
  benchmark manifests.
- [x] Re-run Colab `kernel_double_cable_extracellular_auto_long` to measure the
  realistic CPU/GPU comparison after auto selection: CPU should use Thomas,
  GPU should use the GPU PCR policy. Result: `auto` resolved correctly; GPU won
  `2.51x/2.48x/2.08x/1.68x` at `n=100/300/600/2000`.
- [x] Add homogeneous double-cable observer-only batch support for
  `Recording.none()` plus solver-side `PeakVoltage`/`Activation` observers,
  returning compact cohort observations instead of retained Vm traces.
- [x] Re-run Colab `kernel_double_cable_observer_auto_long` to compare retained
  center traces against compact observer-only output for the same double-cable
  extracellular auto policy at `n=100/300/600/2000`. Result: observer-only
  reduced GPU total by `28%/51%/59%/62%` vs retained center traces, and reduced
  `results.split_batch` by at least `96%` at every tested size.
- [x] Add solver-only/precomputed-input benchmarks for intra and extra paths so
  solver throughput is separated from preprocessing and result packaging.
- [x] Add JAX profiler trace capture to the hotpath runner and Colab notebook:
  `--jax-trace` records per-workload/per-size profiler directories in
  manifests, and `kernel_double_cable_observer_auto_trace` captures GPU
  timelines for the observer-only auto double-cable path.
- [x] Re-run Colab `kernel_double_cable_observer_auto_trace` after
  kernel-scoped tracing, incremental double-cable dispatch grouping, and
  scalarized PCR block products; compare GPU timelines against
  `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_192515`.
- [x] Analyze rerun
  `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_193544`.
  Dispatch planning dropped to `4 ms`/`9 ms` at `n=600/2000`; kernel-scoped
  traces now capture GPU events for `n=2000`; scalarized PCR removed the
  GEMM/dot hotspot and cut `n=600` device time from `554 ms` to `256 ms`.
- [x] Prototype a struct-of-arrays PCR variant that keeps the `2x2` block
  components as scalar arrays instead of stacked `(Nx, 2, 2)` matrices. The
  `193544` trace shows the remaining GPU cost is dominated by many small
  elementwise slice/add/subtract fusions and one command-buffer execution per
  time step, not memory copies.
- [x] Re-run Colab `kernel_double_cable_observer_auto_trace` and
  `kernel_double_cable_observer_auto_long` after the struct-of-arrays PCR
  rewrite. SoA is a conditional win: no-trace GPU totals improved at
  `n=100/300/600` (`148/167/230 ms`, `-20%/-12%/-22%` vs `194140`) but
  regressed at `n=2000` (`1984 ms`, `+27%`). Trace confirms that SoA removes
  the heavy slice fusions but shifts large-batch cost into `select_reduce`
  fusions (`1470 ms` at `n=2000`).
- [x] Split double-cable PCR variants after the SoA evidence gate:
  `pcr` keeps the matrix-layout scalarized PCR, `pcr_soa` exposes the
  struct-of-arrays prototype, and GPU `auto` now resolves to `pcr_adaptive`
  so small/medium batches use SoA while larger batches fall back to the
  matrix-layout PCR.
- [ ] Re-run Colab `kernel_double_cable_observer_auto_long` after
  `pcr_adaptive` to confirm the combined target: keep the SoA gains through
  `n=600` without the `n=2000` regression.
- [x] Inspect
  `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_192515`.
  The `n=600` GPU trace shows the double-cable observer kernel spends about
  `554 ms` on device, with `GEMM/dot` (`240 ms`, `43%`) plus transpose/gather/
  select/scatter fusions dominating; H2D copy is only `8.8 ms` (`1.6%`).
  The `n=2000` full-run trace hit the Python event budget before GPU events,
  so future large traces must use kernel-only scope.
- [x] Make JAX profiler traces kernel-scoped by default and keep
  `--jax-trace-scope run` only for dispatch/preparation investigations.
- [x] Optimize double-cable dispatch grouping after the trace exposed
  repeated full-group padding checks; homogeneous/compatible double-cable rows
  now maintain incremental membrane-prefix compatibility state.
- [x] Scalarize PCR `2x2` block products to avoid lowering tiny block
  multiplications as many GPU GEMM/dot kernels and transposes.

## Phase 7.6.4 Experimental Pseudo-Double-Cable / Pseudo-MRG

Status: in progress. This is an experimental validation phase, not a default
solver replacement. Exact double-cable remains the reference implementation
and the final arbiter for ambiguous or biophysically critical cases.

Goal: evaluate whether pseudo-double-cable / pseudo-MRG variants can preserve
the relevant physiological behavior of the exact double-cable model before
optimizing them for GPU throughput. Speed is useful only after threshold,
activation, propagation, and recruitment behavior are credible.

Source notes to use:

- [x] Read and keep aligned with
  `ideas/axonscope_pseudo_double_cable_gpu_implementation_plan.md`.
- [x] Read and keep aligned with `ideas/pseudo_mrg.md`.
- [x] Read and keep aligned with
  `ideas/axonscope_double_to_single_electrical_reduction_plan.md`. This is the
  most concrete guide for the next pseudo-double implementation pass:
  `series` -> `schur_local` -> `dynamic`, with exact double-cable remaining the
  reference.

Experimental guardrails:

- [x] Keep all pseudo-double modes opt-in and clearly labelled experimental.
- [x] Do not silently reinterpret `double` as pseudo-double.
- [ ] Do not make pseudo-double part of `auto` solver selection until the
  physiological validation gates below pass.
- [ ] Keep exact double-cable runs available in every validation script as the
  reference and refinement path.
- [ ] Prefer high recall over high precision when pseudo modes are used as a
  pre-filter; near-threshold or ambiguous cases must be rerun with exact
  double-cable.

Physiology-first validation harness:

- [x] Add a deterministic pseudo-double validation harness before adding broad
  GPU performance claims. It should run exact double and pseudo candidates on
  the same generated or fixture-backed workloads.
- [ ] Start with small correctness/physiology cases:
  `Nx=32/51/64/96`, low `B` for deterministic comparisons, and amplitudes
  below threshold, near threshold, and above threshold.
- [ ] Compare operational physiology metrics against exact double-cable:
  activation boolean, activation time, activation node, threshold amplitude,
  recruitment curve, conduction velocity, spike initiation location, peak Vm,
  time-to-peak, RMS/probe trace error, and subthreshold response.
- [ ] Include MRG-specific sanity checks where available: node/internode
  response, strength-duration behavior, refractory behavior, and conduction
  block or strong-gradient cases.
- [ ] Track periaxonal/auxiliary-state behavior for pseudo modes that expose
  it, but do not require pseudo-effective to reproduce full periaxonal traces.
- [x] Save validation summaries as JSON/CSV with mode, workload, backend,
  dtype, `Nx`, `B`, `Nt`, speed, memory/output size, and all error metrics.
- [x] Add optional PNG plots for pseudo-double validation runs: activation
  summary, physiology errors, thresholds, timings, and selected trace
  comparisons.

Initial acceptance gates:

- [ ] `pseudo_double_effective` is acceptable as a rough screening mode only if
  it gives useful speed and high recall for exact-double activations, even when
  trace error is imperfect.
- [ ] Target threshold relative error: ideal `<=1-3%`, acceptable screening
  `<=5%`; above `5-10%` should be labelled rough pre-screen only.
- [ ] Target activation agreement near the operating range: at least `95%`;
  ambiguous-zone recall should be near `99%` for pre-filter use.
- [ ] Recruitment and electrode/configuration ranking should preserve ordering
  well enough for screening, with rank correlation tracked explicitly.
- [ ] Do not advance a pseudo mode toward production examples until it passes a
  held-out physiology validation set, not just one calibration workload.

Implementation sequence:

- [x] Add explicit experimental mode names and plumbing only after the
  validation harness exists: `pseudo_double_effective`,
  `pseudo_double_single_myelinated_chain`, `pseudo_double_series`,
  `pseudo_double_split`, `pseudo_double_schur_local`, and optional
  `pseudo_double_modal`.
- [x] Keep not-yet-implemented pseudo mode strings raising clear
  `NotImplementedError` until their kernels exist.
- [x] Implement `pseudo_double_effective` v0 as a scalar-cable surrogate with a
  calibrated extracellular coupling multiplier. Reuse existing scalar
  tridiagonal infrastructure instead of duplicating the single-cable solver.
- [ ] Derive and implement real effective pseudo-double coefficients beyond
  scalar `vext_scale` if the v0 physiology evidence is promising enough.
- [x] Add mini calibration plumbing for `pseudo_double_effective` and include a
  same-run baseline comparison against `mrg_single_cable_surrogate`.
- [ ] Validate `pseudo_double_effective` against exact double before optimizing
  it. Decide whether it is useful for rough screening, calibrated screening,
  or not useful.
- [x] Implement `pseudo_double_split` v0 after the effective-model smoke:
  keep one local implicit auxiliary extracellular-response state and reuse the
  scalar single-cable path. This first version is field-filtered only; it does
  not yet include spatial periaxonal coupling or Vm feedback.
- [x] Add coefficient-level electrical-reduction helpers from the
  double-to-single plan: local series equivalent and diagonal-App Schur local
  v1, with synthetic tests proving Schur v1 is exact when the eliminated block
  has no spatial off-diagonal coupling.
- [x] Promote the first coefficient-derived reduction into a runnable validation
  mode: `pseudo_double_schur_local` now derives a scalar tridiagonal system
  from exact double-cable coefficients via diagonal-`App` Schur elimination.
- [x] Add a runnable `pseudo_double_series` validation mode. It uses the exact
  double-cable runtime arrays, local axolemma/myelin RC-series reduction, node
  fallback for degenerate myelin capacitance, and one scalar tridiagonal solve
  per step.
- [x] Add `pseudo_double_single_myelinated_chain` from the double-to-single
  plan as a validation-only one-voltage MRG-like chain built directly from
  AxonScope single-cable primitives. It preserves NODE/MYSA/FLUT/STIN sections,
  uses active nodes plus passive effective internodes, and supports
  segment-specific extracellular alpha without changing core `src/axonscope`.
- [ ] Decide the public vocabulary later: keep `pseudo_double_effective` /
  `pseudo_double_split` aliases, or expose plan-aligned names such as
  `pseudo_double_series`, `pseudo_double_schur_local`, and
  `pseudo_double_dynamic`.
- [ ] Tune `pseudo_double_single_myelinated_chain` beyond global
  `vext_scale`: sweep segment-specific Cm/leak/alpha and near-threshold
  amplitudes before calling the behavior physiologically credible.
- [ ] Derive a stronger `pseudo_double_split` kernel with true pointwise or
  semi-local periaxonal/myelin feedback once v0 behavior is understood.
- [ ] Validate `pseudo_double_split` on stronger double-cable regimes:
  strong extracellular gradients, short/long pulses, near-threshold
  amplitudes, heterogeneous cable properties, and long `Nt` stability.
- [ ] Consider `pseudo_double_modal` only after effective/split evidence shows
  a real fidelity gap. Add a one-step linear-system approximation test before
  full time integration.

Testing plan:

- [x] Add unit tests for mode parsing, explicit experimental status,
  pseudo-effective config/score helpers, and explicit `NotImplementedError`
  behavior for pseudo modes whose kernels have not landed.
- [x] Add unit tests for pseudo-mode shape/dtype stability at the validation
  harness level for `pseudo_double_effective`, `pseudo_double_series`, and
  `pseudo_double_split`.
- [x] Add synthetic coefficient tests for local series reduction and Schur local
  v1 exactness when `App` is diagonal.
- [x] Add unit tests for `pseudo_double_single_myelinated_chain` segment
  taxonomy, single-cable/periaxonal layout, segment alpha vector, validation
  routing, dry-run output, and scaled extracellular footprint behavior.
- [ ] Add unit tests for compact observer outputs and no dense zero `Iinj`
  materialization once pseudo kernels exist below the harness layer.
- [ ] Add loose numerical/physiology tests against exact double on small,
  deterministic cases. Avoid brittle full-trace equality; test thresholds,
  activation decisions, and bounded summary errors first.
- [ ] Add optional GPU/performance tests behind a marker only after behavior
  validation exists; normal CI should not depend on GPU timing.
- [ ] Add regression tests that exact double remains unchanged when pseudo
  modes are added.

GPU and memory work after physiology gates:

- [ ] Benchmark pseudo modes against exact double at target scale:
  `B=512/1024/2048/4096`, `Nx=32/51/64/96`, realistic `Nt`, compact observers
  and optional retained traces.
- [ ] Keep timings separated into compile, first run, steady-state run,
  host-device transfer, device execution, and postprocessing/output.
- [ ] Add JAX trace capture for pseudo modes and compare kernel structure
  against exact double-cable.
- [ ] Add factorized extracellular stimulation support
  `waveform[Nt] * footprint[B,Nx]` before claiming production-scale memory
  behavior.
- [ ] Ensure pseudo modes work with compact solver-side observers so full
  `Vm[B,Nt,Nx]` output stays opt-in.

Hybrid/refinement path:

- [ ] Add a pseudo-first / exact-refinement validation workflow: run pseudo on
  all fibers/configurations, identify clearly active, clearly inactive, and
  ambiguous cases, then rerun exact double on ambiguous cases.
- [ ] Merged outputs must mark which rows were exact-refined and report how
  much exact double-cable work was avoided.
- [ ] Later, evaluate multi-fidelity pseudo-MRG ideas from `pseudo_mrg.md`:
  pseudo-single exterior, corrected pseudo-single region, pseudo-double buffer,
  and exact full double-cable core around critical zones.
- [ ] Use activating-function and disagreement metrics for any spatial or
  dynamic fidelity promotion. Promotion should be fast; demotion should be
  conservative.

Documentation and examples:

- [ ] Document limitations prominently: pseudo modes are approximations, exact
  double remains the reference, and each new fiber/stimulation regime needs
  validation.
- [ ] Add an advanced experimental pseudo-double example only after the first
  validation gates pass. The example should demonstrate validation and
  interpretation, not only speed.
- [ ] Keep benchmark-heavy Pareto fronts under `benchmark/`; examples should
  teach safe usage and when to rerun exact double.

## Phase 7.7 Stimulation And Placement API Cleanup

Goal: make the public API match the product boundary before Phase 8 studies.

Implementation gate:

- [ ] Re-read `GUIDELINES.md` before implementing Phase 7.7 and extract the
  concrete target boundary for stimulation, placement, populations, and study
  inputs.
- [ ] Compare `GUIDELINES.md` against current source, tests, examples, and
  docs before editing public APIs; write the rename/delete checklist here or
  in a short implementation note.
- [ ] If the intended implementation differs from `GUIDELINES.md`, update
  `GUIDELINES.md` first, then align `todo.md` and `agent.md`.

- [ ] Remove remaining public `y` / `z` placement parameters from axon model
  constructors. An `Axon` describes cable, membrane, length, diameter, and
  layout only.
- [ ] Move physical placement to instance, population, or study layers where it
  is still needed.
- [ ] Remove public `intracellular_context` and `extracellular_context`
  terminology from user-facing APIs and examples.
- [ ] Replace generic context methods with explicit domain commands:
  current clamps, point-source electrodes, extracellular drives, footprints,
  stimulation collections, and study inputs.
- [ ] Keep PointSource/electrode/footprint concepts, but make them first-class
  stimulation objects rather than hidden context plumbing.
- [ ] Decide which lower-level internal objects may keep `context` as an
  implementation detail; keep them out of the public facade and examples.
- [ ] Update tests, docs, examples, benchmark workload builders, public API
  tests, `tests/unit/test_examples.py`, and `CHANGELOG.md` in the same pass.
- [ ] Add or update guardrails so removed names do not return as compatibility
  aliases.

## Phase 7.8 Examples Learning Path

Goal: keep examples didactic, runnable, plot-rich where helpful, and aligned
with the cleaned public API.

- [ ] Update examples after the stimulation/placement API cleanup.
- [ ] Add an advanced solver-options example after Phase 7.6.3 stabilizes the
  public contract: demonstrate `BatchOptions` / hotpath CLI choices for
  `auto`, `thomas`, `pcr`, `pcr_soa`, and `pcr_adaptive`, with clear guidance
  that forced solver choices are mainly diagnostic unless selected by `auto`.
- [ ] Add a separate advanced pseudo-double / pseudo-MRG experimental example
  only after Phase 7.6.4 has a validated physiology harness; keep it framed as
  validation-first approximation work, not a default solver tutorial.
- [ ] Keep examples verbose, line-by-line, and commented near the code being
  taught.
- [ ] Add useful plots for signals, metrics, activation, recruitment, velocity,
  observer outputs, dispatch layouts, or memory comparisons.
- [ ] Avoid turning examples into stress tests; benchmark-heavy evidence stays
  under `benchmark/`.
- [ ] Re-run `tests/unit/test_examples.py` after example edits.

## Phase 8 Studies

Goal: add the public workflow for sweeps, thresholds, recruitment, reuse, and
compact study outputs.

- [ ] Define callable update contract:
  `update(base_simulation: AxonSimulation, condition: Condition) -> AxonSimulation`.
- [ ] Require updates to avoid mutating the base simulation and to make the
  condition explicit.
- [ ] Add sweep API.
- [ ] Add threshold-search API.
- [ ] Add recruitment-sweep API.
- [ ] Add reuse policies for prepared cohorts, compiled kernels, footprints,
  and stimulus-only updates:
  `AUTO`, `REQUIRE`, `NONE`.
- [ ] Add retention policies so threshold/recruitment studies do not retain
  every trace by default.
- [ ] Add study result containers with compact per-row/per-condition outputs
  and optional retained traces.
- [ ] Document callable reproducibility limits; do not claim arbitrary lambdas
  are serializable.
- [ ] Add didactic advanced examples for each new public study concept.

## Phase 9 Serialization And Reference Backend

Goal: stabilize schemas only after object/result/analysis/study models settle.

- [ ] Define final schemas for simulations, results, and study results.
- [ ] Serialize typed values, identifiers, recording manifests, analysis
  definitions, backend/device/precision, and environment metadata.
- [ ] Do not add readers for prototype formats.
- [ ] Add NumPy reference backend validation for small deterministic cases.
- [ ] Add cross-backend validation before treating serialization as stable.
- [ ] Add final docs only after schemas and reference validation are stable.

## Open Architecture Decisions

- [ ] Decide whether scalar public `simulate(...)` eventually returns
  `AxonSimulationResult` instead of `SimResult`.
- [ ] Replace direct `Recording.to_batch_options()` solver coupling with:
  `Recording -> RecordingPlan -> validation -> backend lowering`.
- [ ] Add backend-neutral axon structure descriptors and cable capability
  descriptors.
- [ ] Extend semantic signals beyond Vm/gates/currents/conductances/states:
  intracellular potential, periaxonal potential, ionic current, and
  cable/role-aware signal availability.
- [ ] Decide whether latency/block-style analyses become direct solver
  observers or thin views over activation observer state.
- [ ] Decide logging policy: Python logging, benchmark traces, warnings,
  result diagnostics, and user-facing summaries.
- [ ] Decide print/Rich/progress policy: defaults, opt-in progress, notebook
  behavior, and CI degradation.
- [ ] Decide when Sphinx docs are stable enough to generate after the docs/code
  audit.

## Documentation Audit

Gate:

- [ ] Audit `/docs` against current code before writing Sphinx docs.
- [x] Mark each doc page as current, partially stale, proposal-only, or
  redundant.
- [x] Move confirmed mismatches into this TODO with concrete file paths.

Current page snapshot:

| Page | Status | Next action |
| --- | --- | --- |
| `docs/axon_model_organization.md` | Partially current | Re-check examples against `src/axonscope/axons/`. |
| `docs/solver_organization.md` | Likely current | Light verification before Sphinx. |
| `docs/membranes.md` | Mostly current | Verify `Composite`, `SectionLayout`, and examples. |
| `docs/stimulation.md` | Mostly current | Re-check after Phase 7.7 API cleanup. |
| `docs/pool_dispatch.md` | Mostly current | Review for overlap with README and API drift. |
| `docs/results_recording_analysis.md` | Partially stale | Refresh for Phase 7.5 `Recording.none()` and solver-side observations. |
| `docs/recorders_observers_activation_strategy.md` | Proposal | Refresh status now that Phase 7.5 observer-only execution exists. |
| `docs/api_public_draft.md` | Proposal-only | Later split implemented API from proposal if it remains user-facing. |
| `docs/validation.md` | Mostly current | Add dated NRV result only after a fresh NRV-ready run. |

Open documentation tasks:

- [ ] Split `docs/api_public_draft.md` into implemented API versus proposal if
  it remains part of user-facing docs.
- [ ] Refresh `docs/results_recording_analysis.md` for solver-side observer
  execution and trace-free `Recording.none()` results.
- [ ] Re-run NRV validation only in an NRV-ready environment; record dated
  validation notes after a fresh run.
- [ ] Keep proposal/roadmap docs clearly labelled so users do not run future
  API snippets as current API.
- [ ] Provide extensive public docstrings before generating API docs.
- [ ] Decide what belongs in Sphinx pages versus README versus examples.

## Benchmark Workstream

Canonical lightweight evidence loop:

- `benchmark/hotpaths/`
- `benchmark/hotpaths/run.py --list`
- `benchmark/hotpaths/README.md`
- `benchmark/hotpaths/COLAB.md`
- `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`

Open benchmark-agent tasks:

- [x] Implement opt-in JAX profiler traces for hotpath runs; traces are
  recorded under `jax_traces/<workload>_n<size>/` and linked from manifests.
- [ ] Add explicit cold-start/first-call signature labels.
- [ ] Audit scalar `simulate(...)` instrumentation against pool
  instrumentation and ensure both expose consistent root spans.
- [ ] Decide whether `level="minimal"` and `level="detailed"` are worth
  implementing, or keep only `level="hotpaths"` and document that choice.
- [ ] Improve benchmark summaries with percentages of root time, median/p95
  columns, parent names, and enough dimensions to compare runs without
  reopening every `events.jsonl`.
- [ ] Add or refresh docs for asynchronous GPU timing, `kernel.enqueue`,
  `kernel.wait`, first-call classification, output files, and JAX trace
  limitations.
- [ ] Add skipped GPU integration tests that verify device metadata and
  `kernel.wait` behavior when a GPU is available.
- [ ] Separate correctness validation from performance benchmarking in docs and
  scripts.
- [ ] Clean legacy benchmark assets only after benchmark-agent and bottleneck
  leftovers are closed.

Legacy benchmark areas to classify later:

- `benchmark/runtime/`
- `benchmark/results/runtime/`
- `benchmark/reports/runtime/`
- old standalone runtime scripts
- generated caches
- dated exploratory output files

## API And Units Backlog

- [ ] Stabilize the public API before adding Sphinx.
- [ ] Prefer clean user-facing interfaces over backward compatibility shims.
- [ ] Remove temporary compatibility aliases and old argument names once
  examples/tests/docs use the clean API.
- [ ] When removing or renaming public API, update affected examples in the
  same change and keep `tests/unit/test_examples.py` aligned.
- [ ] Pass units everywhere at public boundaries, including membranes, axons,
  stimulation, protocols, recording, and solver wrappers.
- [ ] Audit membrane constructors for plain-number ambiguity and decide where
  units should be required versus normalized.
- [ ] Preserve `SimResult.Vm` as a stable notebook-friendly convenience, with
  explicit errors when observer-only runs do not carry Vm.

Locked compatibility decisions:

- [x] `axs.analysis` is a real package; old forwarding aliases are removed.
- [x] `axs.visualization` remains absent; plotting stays under
  `axs.results.visualization`.
- [x] `axs.run_batch(...)` is removed; use `simulate_pool(...)`.
- [x] Public wrappers use `duration` / `dt` with Pint quantities.
- [x] Direct solvers use solver-level `tsim` / `dt` only.
- [x] Public recording uses `sample_dt` and `positions` with units.
- [x] Public point-source electrodes use quantity-oriented `x`, `y`, `z`, and
  `min_distance`.
- [x] Activation protocol currents require current units.
- [x] Explicit `Stimulus` time inputs require units.
- [x] `clear_extracellular_context(...)` is singular.

## Recording, Observables, And Analysis Backlog

- [ ] Verify single-axon and pool recording behavior for Vm, gates, currents,
  conductances, and state variables across supported groups.
- [ ] Keep post-hoc `ActivationCriterion` semantics aligned with future
  solver-side observers.
- [ ] Add plotting helpers for batch groups and retained recording layouts.
- [x] Lock current public `Recording` contract with tests: scalar runs require
  Vm and may include observable groups; pool runs support Vm spatial modes
  only; unsupported position/temporal/pool-observable filters raise explicit
  errors.
- [x] Make batch diagnostics discoverable from per-axon public result views.

## Evidence Ledger

Use this section for compact dated evidence that affects current decisions.
Keep long narrative in benchmark artifacts, not here.

| Date | Evidence | Result / Decision |
| --- | --- | --- |
| 2026-06-14 | Phase 2 final unit + NRV | Unit `263 passed, 1 skipped`; NRV `116 passed, 516 warnings`. |
| 2026-06-14 | Phase 4 final validation | Compileall passed; targeted backend/guardrail run `70 passed`; full unit `286 passed, 1 skipped`; full NRV `116 passed, 516 warnings`; hotpath smoke passed. |
| 2026-06-14 | Phase 5 final unit | `291 passed, 1 skipped`. |
| 2026-06-14 | Phase 6 final unit | `300 passed, 1 skipped`. |
| 2026-06-14 | Phase 7 final unit | `306 passed, 1 skipped`. |
| 2026-06-15 | Phase 7.5 final unit | `308 passed, 1 skipped`. |
| 2026-06-15 | Compact dispatch cohort cleanup | Full unit `314 passed, 1 skipped`. |
| 2026-06-15 | `colab_cpu_gpu_20260615_102221` | `realistic_mixed_population_n500` GPU improved from `11364.8 ms` to `556.3 ms`; CPU from `3698.0 ms` to `238.4 ms`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_realistic_long_20260615_103306` | First realistic trace where GPU wins: `744.6 ms` GPU vs `1253.8 ms` CPU. |
| 2026-06-15 | `colab_cpu_gpu_kernel_observer_long_20260615_104356` | Observer-only retained Vm eliminated; GPU speedup around `5.8x` at `n=500/1000`; dense `Iinj` was the next pressure point. |
| 2026-06-15 | `colab_cpu_gpu_kernel_observer_long_20260615_114221` | Sparse current-clamp path improved GPU `n=1000` total from `673.4 ms` to `378.0 ms`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_observer_long_20260615_120457` | Zero-field path eliminated dense zero `Vstim`; CPU improved more than GPU. |
| 2026-06-15 | `colab_cpu_gpu_kernel_observer_long_20260615_122541` | Compact cohort path reduced GPU `results.split_batch` at `n=1000` from `140.6 ms` to `0.76 ms`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_observer_long_20260615_132920` | Pulse-vectorized sparse input path reduced GPU `inputs.intracellular` at `n=1000` from `150.2 ms` to `23.8 ms`; total GPU `176.8 ms`. |
| 2026-06-15 | `local_double_cable_extracellular_after_runtime_cache_n50` | Double-cable `runtime.prepare` dropped from `45.3 ms` to `2.55 ms`; total `108.8 ms` to `55.6 ms`. |
| 2026-06-15 | `local_double_cable_extracellular_after_combined_vstim_builder_n50` | Double-cable `inputs.extracellular` dropped from `5.93 ms` to `2.81 ms`; next meaningful decision needs Colab GPU. |
| 2026-06-15 | Local path matrix at `n=100` | Single intra `89.9 ms`, single point-source extra `76.0 ms`, MRG double extra `176.0 ms`; double-cable/MRG jump is the main current difference. |
| 2026-06-15 | `phase7_6_path_matrix_extended_smoke` | `path_comparison_matrix` now builds 10 labeled scenarios, records explicit comparison axes for single intra, single point-source extra, and double MRG point-source rows, and manifests include the `jax_log_compiles` setting. |
| 2026-06-15 | `phase7_6_path_matrix_warm_local` | Warm local `path_comparison_matrix` at `n=5/20` covers center/probes/full/observer-none rows; totals were `86.1 ms` and `91.6 ms`, with kernel enqueue around `44-45 ms`. |
| 2026-06-15 | `phase7_6_solver_only_warm_local` | Direct precomputed-input workload isolates solver timing: warm `n=5` kernel enqueue `11.6 ms`, wait `0.06 ms`, bypassing dispatch/input materialization. |
| 2026-06-15 | `phase7_6_typed_drive_warm_local` | Typed-drive lowering matched analytical-context lowering exactly (`max_abs_delta_mV=0.0`); warm `n=5` context lowering `4.07 ms`, typed-drive lowering `3.24 ms`, kernel enqueue `4.78 ms`. |
| 2026-06-15 | `phase7_6_double_cable_warm_local` | Local MRG double-cable extracellular `n=5` warm total `19.2 ms`; zero-Iinj path reduced `inputs.intracellular` to `0.043 ms`, with `kernel.enqueue 13.0 ms`. |
| 2026-06-15 | `phase7_6_observer_chunked_warm_local` | Runner-level `--time-chunk-steps 5` verified on observer-only `n=20`, `duration=1 ms`; total `16.9 ms`, sparse `inputs.intracellular 1.33 ms`, zero-field `inputs.extracellular 0.025 ms`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_extracellular_long_20260615_175837` | Double-cable extracellular long run: GPU loses at `n=100` (`824.9 ms` vs CPU `465.1 ms`), crosses over at `n=300` (`954.8 ms` vs `1165.8 ms`), and wins `2.32x` at `n=600` (`1070.1 ms` vs `2485.8 ms`). Runtime now skips dense zero `Iinj` (`108 MB` skipped at `n=600`); static memory estimates were aligned afterward. |
| 2026-06-15 | `colab_cpu_gpu_kernel_single_cable_extracellular_long_20260615_181029` | Matching single-cable extracellular long run scales much better on GPU: `5.43x`, `6.81x`, and `9.78x` CPU/GPU at `n=100/300/600`; double-cable slowdown is kernel-specific, not generic extracellular preprocessing. |
| 2026-06-15 | `phase7_6_double_cable_zero_iinj_kernel_smoke` | Local smoke passed after keeping absent double-cable intracellular input as `None` through the kernel path instead of passing a dense device zero `Iinj[B,Nt,Nx]`; Colab double-cable rerun needed for GPU impact. |
| 2026-06-15 | `phase7_6_double_cable_shared_constants_smoke` | Local smoke passed after keeping shared double-cable coefficients unbatched through the stateful `vmap` path instead of broadcasting every cable/extracellular coefficient to `(B,Nx)` or `(B,Nx-1)`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_extracellular_long_20260615_181846` | Zero-Iinj kernel-input specialization: GPU total improved at `n=100` (`809.7 ms`, `-1.8%`) and `n=600` (`984.9 ms`, `-8.0%`) vs baseline, but regressed at `n=300` (`990.9 ms`, `+3.8%`); CPU improved `8-12%`. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_extracellular_long_20260615_182156` | Shared-coefficient specialization: GPU improved at `n=300` (`859.4 ms`, `-10.0%`) and slightly at `n=600` (`1031.8 ms`, `-3.6%`) vs baseline, but regressed at `n=100` (`851.9 ms`, `+3.3%`); CPU improved most (`-30.7%` at `n=600`). |
| 2026-06-15 | `phase7_6_double_cable_pcr_batch_smoke` | Experimental PCR block solver landed behind `BatchOptions(double_cable_block_solver="pcr")` / `--double-cable-block-solver pcr`; local batch smoke passed at `n=2`, and numerical tests match Thomas. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_extracellular_pcr_long_20260615_183818` | PCR block solver is the first strong double-cable GPU win: GPU totals `243.1/369.2/687.8 ms` at `n=100/300/600`, or `3.3x/2.3x/1.4x` faster than the best Thomas run. CPU regressed badly (`3092/10262/18699 ms`, `7.2x/11.6x/10.9x` slower), so the next comparison should use `auto`: PCR on GPU, Thomas on CPU. |
| 2026-06-15 | `phase7_6_double_cable_auto_solver_policy` | Added `double_cable_block_solver="auto"` as the default batch option and hotpath CLI default. It resolves to PCR on GPU and Thomas elsewhere, while benchmark manifests record both requested and resolved solver choices. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_extracellular_auto_long_20260615_185532` | Auto double-cable long run resolved GPU to PCR and CPU to Thomas. GPU totals were `336/768/1670/7523 ms` vs CPU `845/1907/3471/12619 ms` at `n=100/300/600/2000`, for `2.51x/2.48x/2.08x/1.68x` CPU/GPU speedups. |
| 2026-06-15 | `phase7_6_double_cable_observer_batch_smoke` | Homogeneous double-cable observer-only batch path now returns compact cohort observations with `Vm=None`; local smoke for `double_cable_observer n=2` produced `recording_policy=none`, `vm_shapes=[]`, and `peak_voltage`/`activation` observations. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_observer_auto_long_20260615_190721` | Double-cable observer-only auto run kept `vm_shapes=[]`, GPU=PCR, CPU=Thomas. GPU totals were `243/378/682/2841 ms` at `n=100/300/600/2000`, `28%/51%/59%/62%` faster than retained center traces from `185532`; GPU `results.split_batch` fell to `0.10/0.20/0.55/1.15 ms` (`>=96%` lower). |
| 2026-06-15 | `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_192515` | JAX trace captured `n=600` GPU kernel details: device time was mostly PCR/XLA compute (`GEMM/dot` `43%`, transpose fusions `19%`, select/reduce `14%`), not transfers (`MemcpyH2D` `1.6%`). The `n=2000` full-run trace filled the profiler event budget with Python dispatch frames, motivating kernel-scoped tracing and incremental double-cable dispatch grouping. |
| 2026-06-15 | `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_193544` | Kernel-scoped trace after dispatch and scalar-PCR changes: dispatch planning fell to `4/9 ms` at `n=600/2000`; `n=600` GPU device time fell from `554 ms` to `256 ms` and GEMM/dot disappeared; `n=2000` now captures GPU device time (`1200 ms`) with remaining cost dominated by small elementwise fusions and `1000` command-buffer/CUDA-graph executions. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_observer_auto_long_20260615_194140` | No-trace reference after scalar-PCR and dispatch cleanup. GPU totals were `185/190/296/1557 ms` at `n=100/300/600/2000`, vs CPU `475/817/2041/5229 ms`, for CPU/GPU speedups `2.57x/4.30x/6.89x/3.36x`. Compared with `190721`, GPU total improved by `24%/50%/57%/45%`; `n=600` kernel fell from `532 ms` to `204 ms`. Dense `Vstim` remains visible (`16-27%` of GPU total at `n>=300`). |
| 2026-06-15 | `phase7_6_double_cable_pcr_soa_smoke` | Rewrote PCR internals to keep `2x2` block components as separate arrays. Local numerical tests match Thomas; the JAX/HLO audit for `Nx=45` reports `0` `dot`, `dot_general`, `scatter`, or `transpose` operations, leaving neighbor `gather`s as the expected PCR data movement. |
| 2026-06-15 | `colab_gpu_only_kernel_double_cable_observer_auto_trace_20260615_195401` | Kernel-scoped trace after SoA PCR: GPU device time improved at `n=600` (`282 -> 223 ms`, `-21%`) but regressed at `n=2000` (`1236 -> 1627 ms`, `+32%`) vs `193544`. SoA removed slice fusions (`64/675 ms -> ~0`) but shifted cost to `select_reduce` (`152/1470 ms`), while memcpy remained negligible. |
| 2026-06-15 | `colab_cpu_gpu_kernel_double_cable_observer_auto_long_20260615_195524` | No-trace SoA PCR run: GPU totals `148/167/230/1984 ms` vs CPU `441/840/1994/4804 ms` at `n=100/300/600/2000`, giving CPU/GPU speedups `2.98x/5.04x/8.67x/2.42x`. Versus `194140`, SoA improves GPU totals through `n=600` (`-20%/-12%/-22%`) but regresses `n=2000` (`+27%`). |
| 2026-06-15 | `phase7_6_double_cable_pcr_adaptive_policy` | Kept both PCR layouts: `pcr` is the matrix-layout scalarized solver, `pcr_soa` is the struct-of-arrays solver, and GPU `auto` resolves to `pcr_adaptive`, selecting SoA for batches up to `1024` and matrix-layout PCR beyond that. Targeted solver/batch tests and a local `double_cable_observer n=2` hotpath smoke pass. |
| 2026-06-15 | `phase7_6_4_pseudo_double_validation_harness` | Added experimental pseudo-double validation harness under `benchmark/pseudo_double/`: exact double-cable reference versus `mrg_single_cable_surrogate` on matched MRG point-source workloads, JSON/CSV physiology metrics, explicit `NotImplementedError` for planned pseudo modes, and local smoke `size=1`, `nodes=3`, amplitudes `20/60 uA`. Unit tests cover mode metadata, formulation routing, threshold summaries, dry-run, and output writing. |
| 2026-06-15 | `phase7_6_4_pseudo_double_effective_v0_smoke` | Implemented `pseudo_double_effective` v0 as an experimental MRG single-cable surrogate with calibratable extracellular coupling (`--pseudo-vext-scale`, `--calibrate-vext-scales`) and automatic baseline comparison. Local smoke `size=1`, `nodes=3`, `duration=0.5 ms`, amplitudes `20/60/100/140 uA`, scales `8/10/12/16` selected `vext_scale=10`: activation agreement `1.0`, threshold relative error `0.0`, and no false negatives on this tiny workload, while baseline single-cable missed the exact activation threshold. Peak/RMS trace errors remain large (`peak_abs_error_mean_mV` up to `57.2`, `rms_vm_error_mean_mV` up to `143.8`), so this is not yet physiology-accepted beyond rough activation screening. |
| 2026-06-15 | `phase7_6_4_pseudo_double_split_v0_smoke` | Implemented `pseudo_double_split` v0 in the validation harness as a scalar-cable surrogate with one implicit local auxiliary extracellular-response state (`--split-aux-tau-ms`, `--split-direct-scale`, `--split-aux-scale`, `--split-aux-alpha`) plus `--calibrate-vext-scales`. Local smoke `size=1`, `nodes=3`, `duration=0.5 ms`, amplitudes `20/60/100/140 uA`, `aux_tau_ms=0.05`, scales `4/6/8/10` selected `vext_scale=6`: activation agreement `1.0`, threshold relative error `0.0`, no false negatives, and baseline single-cable still missed the exact activation threshold. Trace fidelity remains rough (`peak_abs_error_mean_mV` up to `56.1`, `rms_vm_error_mean_mV` up to `161.7`), so split v0 is not yet physiology-accepted; next work is stronger periaxonal feedback/spatial coupling and broader validation. |
| 2026-06-15 | `phase7_6_4_double_to_single_reduction_helpers` | Incorporated `ideas/axonscope_double_to_single_electrical_reduction_plan.md` into Phase 7.6.4 and added experimental coefficient helpers under `benchmark/pseudo_double/reductions.py`: `series_equivalent`, exact-solver-sign-convention block coefficient assembly, and `schur_local_v1`. Synthetic tests prove series equivalents are bounded and Schur local v1 matches the exact 2x2 block solve for `Vi` when the eliminated `App` block is diagonal. This prepares real `pseudo_double_series` / `pseudo_double_schur_local` validation modes instead of only stimulus-scale surrogates. |
| 2026-06-15 | `phase7_6_4_pseudo_double_schur_local_v1_smoke` | Added runnable `pseudo_double_schur_local` validation mode using the exact double-cable coefficient assembly and diagonal-`App` Schur elimination, returning normal `AxonSimulationResult` rows from a custom validation-only scalar tridiagonal runner. Local smoke `size=1`, `nodes=3`, `duration=0.5 ms`, amplitudes `20/60/100/140 uA`, scales `4/6/8/10` selected `vext_scale=8`: activation agreement `1.0`, threshold relative error `0.0`, and no false negatives; baseline single-cable still missed the exact activation threshold. Trace errors remain too large for acceptance (`peak_abs_error_mean_mV` up to `51.3`, `rms_vm_error_mean_mV` up to `219.0`), but this is now the first coefficient-derived runnable surrogate rather than a pure stimulus-scale probe. |
| 2026-06-15 | `phase7_6_4_pseudo_double_series_v1_smoke` | Added runnable `pseudo_double_series` validation mode using exact double-cable runtime arrays, local axolemma/myelin RC-series reduction, a node fallback when myelin capacitance is degenerate, and one scalar tridiagonal solve per step. Local smoke `size=1`, `nodes=3`, `duration=0.5 ms`, amplitudes `20/60/100/140 uA`, scales `1/2/4/8`, `capacitance_floor_fraction=0.02` selected `vext_scale=1`: activation agreement `1.0`, threshold relative error `0.0`, no false negatives, peak error `<=2.7 mV`; RMS trace error still rose to `66.2 mV`. Broader local smoke `size=2`, amplitudes `20/40/60/80/100/140 uA`, scales `0.5/1/2/4` selected `vext_scale=2`: activation agreement `1.0`, threshold relative error `0.0`, no false negatives, peak error was small in the active mid-range (`0.7-2.9 mV` at `40-100 uA`) but worse at subthreshold/strong extremes (`19.2 mV` at `20 uA`, `39.8 mV` at `140 uA`) and RMS stayed high (`83.0 mV` max). This is the most promising physiology candidate so far, but the current runner is validation-only and not a GPU performance result. |
| 2026-06-15 | `phase7_6_4_pseudo_double_validation_plots` | Added optional validation plotting via `--plots`, writing `activation_summary.png`, `error_summary.png`, `thresholds.png`, `timings.png`, and selected `trace_amp_*_row_*.png` files under `OUT_DIR/plots`. Trace samples are limited by `--plot-trace-rows` and `--plot-trace-amplitudes-uA` so larger runs do not silently dump every `Vm` trace. Smoke with `pseudo_double_series`, `size=1`, `nodes=3`, amplitudes `20/60/100/140 uA`, trace amplitudes `60/140 uA`, wrote 6 PNGs successfully under `/tmp/axonscope-pseudo-double-series-plots-smoke/plots`. |
| 2026-06-15 | `phase7_6_4_pseudo_double_single_myelinated_chain_smoke` | Added `pseudo_double_single_myelinated_chain`, a validation-only one-voltage NODE/MYSA/FLUT/STIN chain built from AxonScope single-cable primitives, with active nodes, passive effective internodes, series Cm/leak options, and per-segment extracellular alpha scaling. Local smoke `size=1`, amplitudes `20/60/100/140 uA`, scales `0.25/0.5/0.75/1.0` selected `vext_scale=0.75`: activation agreement `1.0`, threshold relative error `0.0`, no false negatives, peak error `3.7 mV` subthreshold and `9.7-10.9 mV` active. Broader `size=2`, amplitudes `20/40/60/80/100/140 uA`, scales down to `0.05` still false-positive activated at `20 uA` while exact threshold was `40 uA`, so global Vext scale alone cannot fix excitability. A no-series-Cm diagnostic removed the `20 uA` false positive but produced false negatives at `40-140 uA`, confirming Cm/alpha must be tuned by segment rather than toggled globally. |

## Completed Roadmap Archive

Keep this as a compact map of what has landed. Detailed history lives in git,
tests, examples, and benchmark result folders.

- [x] Phase 0: guardrails, public API cleanup checks, import-boundary checks,
  obsolete benchmark inventory, and non-NRV baseline.
- [x] Phase 1: direct `AxonSimulation` -> `AxonInstance` rename, root
  `AxonSimulation`, `AxonPopulation`, one/population lifecycle, and public
  diameter inspection.
- [x] Phase 2: typed recording signals, typed position selectors,
  `CableFormulation`, opaque identifiers, `ExtracellularFootprint`,
  `ExtracellularDrive`, `ExtracellularStimulation`, and analytical footprint
  builders.
- [x] Phase 2.5: opt-in benchmark spans, hotpath workload catalog, and Colab
  GPU workflow.
- [x] Phase 3: preparation signatures, reusable prepared cohorts, lower
  planning/input overhead, and footprint-oriented preparation path.
- [x] Phase 4: JAX batch and scalar execution enter through
  `axonscope.backends.jax`; public/descriptive layers are guarded against
  direct JAX imports.
- [x] Phase 5: `CohortResult`, `AxonSimulationResult`, `AxonResultView`,
  extensible `Signal` descriptors, `SignalId`, `RecordingManifest`, and
  `RecordedSignal`; no public `list[SimResult]` pool result.
- [x] Phase 6: real `axs.analysis` package, analysis definitions, low-level
  post-hoc helpers, structured requirements/statuses, population denominators,
  `AnalysisReport`, `result.analyze(...)`, `result.report(...)`, and online Vm
  observers.
- [x] Phase 7: `axs.performance`, simulation memory estimates, runtime/device/
  precision planning values, hotpath memory metadata, and
  `footprint_reuse_sweep`.
- [x] Phase 7.5: public `axs.analysis.PeakVoltage` and
  `axs.analysis.Activation` definitions lower to compact solver observer
  state; scalar kernels and homogeneous single-cable batch kernels update that
  state at every `dt`; `Recording.none()` returns trace-free observations.

Didactic examples landed with phases:

- Phase 1:
  `examples/advanced/example_08_root_axon_simulation.py`,
  `examples/advanced/example_09_axon_population.py`
- Phase 2:
  `examples/advanced/example_10_typed_recording_signals.py`,
  `examples/advanced/example_11_typed_position_selectors.py`,
  `examples/advanced/example_12_cable_formulation.py`,
  `examples/advanced/example_13_extracellular_footprint_drive.py`
- Phase 2.5/7:
  `examples/advanced/example_14_hotpath_benchmarking.py`
- Phase 3:
  `examples/advanced/example_15_preparation_signatures.py`
- Phase 5:
  `examples/advanced/example_16_canonical_pool_results.py`
- Phase 6:
  `examples/advanced/example_17_analysis_layer.py`
- Phase 7.5:
  `examples/advanced/example_18_solver_side_observers.py`

## Cleanup And Sync

- [ ] Do a general cleanup pass after docs, examples, recordings, observers,
  and benchmarks are aligned.
- [ ] Remove stale aliases, removed file references, duplicate docs, and dead
  benchmark/example paths.
- [ ] Keep `agent.md` and `todo.md` synchronized after each cleanup step.
- [ ] Keep this TODO flat: when a section starts accumulating long narrative,
  move details into docs, benchmark manifests, or a compact evidence-ledger row.
