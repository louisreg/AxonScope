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
| Phase 7.7 | Not started | Stimulation/placement API cleanup before studies. |
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
- [ ] Phase 7.6.3: add backend choices for tiny workloads.
- [ ] Phase 7.7: clean stimulation and placement APIs before Phase 8.
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

## Phase 7.6.3 Tiny Workloads And Backend Choice

Goal: make scalar-ish threshold loops and tiny batches efficient without forcing
JAX compile/enqueue overhead onto every workload.

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
- [ ] Decide whether runtime/device/precision planning values remain estimates
  only or start selecting execution backends.
- [x] Prototype an
  algorithmic GPU solver change rather than more input cleanup: e.g. a batched
  or parallel block-tridiagonal solve for the per-step `2x2` system, then
  compare against the current `solve_block_tridiagonal_2x2_scalar` scan.
- [ ] Re-run Colab `kernel_double_cable_extracellular_pcr_long` to compare the
  experimental PCR block solver against Thomas on GPU and CPU at
  `n=100/300/600`.
- [x] Add solver-only/precomputed-input benchmarks for intra and extra paths so
  solver throughput is separated from preprocessing and result packaging.

## Phase 7.7 Stimulation And Placement API Cleanup

Goal: make the public API match the product boundary before Phase 8 studies.

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

- [ ] Implement or intentionally drop `jax_trace=True`; it currently raises
  `NotImplementedError`.
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
