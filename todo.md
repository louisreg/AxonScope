# AxonScope TODO

Living operational roadmap for AxonScope documentation, API cleanup, examples,
benchmarks, solver/backend work, and study APIs.

Read this file at the start of cleanup/API/performance work. Keep it actionable,
chronological, and free of long benchmark prose. Detailed evidence belongs in
dedicated reports under `benchmark/reports/` or focused roadmap files under
`ideas/`.

## How To Use This File

- `GUIDELINES.md` is the master architecture and product-boundary reference.
- `agent.md` captures project working rules for future agents.
- `todo.md` is the current execution plan, not the full historical log.
- Source, tests, and runnable examples remain the truth for current behavior.
- AxonScope is pre-release: prefer clean breaking changes over compatibility
  shims, and delete superseded paths once replacements are in use.

## Current Snapshot

Updated on 2026-06-20 after the full/center/VmRaster CPU+GPU recording-mode
comparison showed that the current GPU bottleneck is the execution envelope
around the solver, not the solver kernel itself.

| Area | Status | Notes |
| --- | --- | --- |
| Phases 0-7.5 | Done | Guardrails, object model, typed contracts, JAX boundary, pool results, analysis layer, performance estimates, and solver-side observers are implemented for the current public layer. |
| Phase 7.6.1 | Done | Benchmark evidence matrix exists under `benchmark/hotpaths/`. |
| Phase 7.6.2 | Done | Memory-transfer and long-run cleanup landed for current hotpaths. |
| Phase 7.6.3 | Closed | Exact double-cable GPU solver optimization pass is complete. No new public solver route; see `benchmark/reports/double_cable_solver_optimization_2026_06.md`. |
| Phase 7.6.4 | Standby | Pseudo-double/pseudo-MRG remains validation-only under `benchmark/pseudo_double/`; not public, not `auto`. |
| Phase 7.6.5 | In progress | Execution-envelope optimization: runtime/input reuse, dispatch/probe-plan reuse, Vext/stimulus materialization, launch/enqueue overhead, and result packaging. |
| Phase 7.6.6 | Planned | GPU dispatch scheduling after reuse work: memory-aware bucket/coalesce first, optional async enqueue second. |
| Phase 7.6.7 | In progress | VmRaster observer redesign: observer-only now lowers to one strict packed membrane-voltage threshold raster. CPU/GPU P100 validation passed; remaining work is decoder breadth and larger memory-focused stress. |
| Phase 7.7 | Next | Stimulation and placement API cleanup against `GUIDELINES.md`. |
| Phase 7.8 | Later | Examples learning-path cleanup after API and Vext work. |
| Phase 8 | Later | Callable studies, reuse policies, retention policies, and study results. |
| Phase 9 | Later | Serialization schemas and reference backend validation. |

Current solver surface:

- `BatchOptions.double_cable_block_solver` accepts exactly `auto`, `thomas`,
  `pcr`, `pcr_soa`, and `pcr_adaptive`.
- `auto` resolves on CPU/default backends to `thomas`; GPU-like backends use
  `pcr_adaptive`.
- `pcr_adaptive` uses `pcr_soa` for `B <= 4096`, then matrix-layout `pcr`.
- Pallas, Triton, JAX-Triton, CUDA FFI, split iterative, associative-transfer,
  and pseudo-double candidates are archived/standby evidence, not active solver
  routes.

## Immediate Queue

Work should start here unless the user asks otherwise.

Attack plan from the 2026-06-20 CPU/GPU recording-mode comparison:

1. Stabilize the benchmark target.
   - Keep the six-way full/center/VmRaster CPU+GPU comparison as the regression
     harness.
   - Preserve plots under
     `benchmark/results/realistic_examples/recording_mode_compare/plots_to_review_20260620`
     as the current reference.
   - Track `runtime.prepare`, `dispatch.build_plan`, `kernel.enqueue`,
     `kernel.wait`, `results.split_batch`, RSS, and device memory estimates.

2. Reduce repeated stable preparation.
   - Identify what `runtime.prepare` rebuilds on every amplitude in
     `example08_recruitment`.
   - Split stable cohort/runtime/probe/footprint preparation from dynamic
     stimulus/amplitude values.
   - Add cache-hit/miss metadata and reuse-failure explanations.

3. Reuse dispatch and observer plans.
   - Cache dispatch groups, padding signatures, recording width, and VmRaster
     row-aware probe tables when static shapes match.
   - Confirm result order and padded-row masks remain identical.

4. Reuse Vext/stimulus inputs.
   - Keep spatial footprints stable and update only temporal stimulus/amplitude
     buffers for sweeps.
   - Avoid dense zero `Iinj` and avoid full dense `Vext` rematerialization when
     a factorized representation is available.

5. Re-benchmark before scheduler work.
   - Re-run the six-way comparison and regenerate all plots.
   - If `runtime.prepare + dispatch.build_plan` falls substantially and group
     count/enqueue remains a bottleneck, start Phase 7.6.6 coalescing.
   - Only test async groups after bucket/coalesce and memory-budget checks.

- [x] Close Phase 7.6.3 solver optimization campaign.
- [x] Clean active solver package and move non-retained custom-kernel tests/code
  to benchmark/archive locations.
- [x] Add a clean solver-campaign summary report with a small speedup plot.
- [x] Add workflow-level benchmark based on basic examples 06/07/08:
  `benchmark/realistic_examples/bench_basic_examples.py`.
- [x] Run Kaggle P100 `realistic_stress` CPU vs GPU with solver/workflow
  profiling; summarized in
  `benchmark/reports/double_cable_solver_optimization_2026_06.md`.
- [ ] Keep the bounded `standard` matrix as an optional cheaper regression run
  once the next Vext/runtime changes land.
- [x] Phase 7.6.5 first pass: reuse batch-safe solver runtimes, cache shared
  point-source footprints, and expose per-group memory estimates in realistic
  profile CSV/events.
- [x] Phase 7.6.5 validation: Kaggle P100 `realistic_stress` CPU vs GPU
  completed at
  `benchmark/results/kaggle/20260619_195351_realistic_stress_NvidiaTeslaP100`;
  warm totals improved from `72.58 -> 60.31 s` CPU and `43.04 -> 36.99 s`
  GPU versus the previous stress run.
- [ ] Propagate optional memory/cache metadata into
  `realistic_examples_profile_cpu_vs_gpu.csv`; current run writes them in the
  per-platform profile CSVs and raw events.
- [x] Phase 7.6.5 observer-only protocol pass: route activation/recruitment
  sweeps through solver-side `Activation` observers when `Recording.none()` is
  used, so recruitment can return compact bool arrays instead of moving full
  `Vm` traces back to host.
- [x] Validate VmRaster observer-only on CPU stress. The old generic observer
  path hit CPU LLVM compile-memory errors, but the VmRaster route completed on
  Kaggle CPU-only at
  `benchmark/results/kaggle/20260620_155134_realistic_stress_observer_cpu_cpu`.
- [x] Run Kaggle `realistic_stress_single_vm` to compare CPU vs GPU with
  example 08 retaining one center Vm column instead of full spatial `Vm`.
  Completed at
  `benchmark/results/kaggle/20260619_232746_realistic_stress_single_vm_NvidiaTeslaP100`.
- [x] Try Kaggle `realistic_stress_observer_gpu` separately after excluding CPU
  observer-only. Decision: inconclusive, not rejected; the run was stopped
  during `example08_recruitment` before a full timing row was produced. The
  `426.7s` Kaggle UI value was a log timestamp, not a case duration.
- [x] Preserve the double-cable batch-native `pcr_soa` fast path for
  observer-only output at realistic batch sizes. Observer-only now uses packed
  `VmRaster` state rather than the removed generic observer state.
- [x] Fix the observer-only route threshold for realistic recruitment batches.
  The first post-fix Kaggle run at `926a8ce` showed `runs=50` taking about
  19 min before `runs=100` because observer-only reused the retained-Vm
  `B >= 2048` threshold and therefore did not activate the new batch-native
  path for `B=50/100`. The batch-native `pcr_soa` route now uses one shared
  low batch threshold for full, center, and observer-only recording modes.
- [x] Add progress diagnostics for long realistic observer GPU runs. The
  `d64fb77` Kaggle run was cancelled while `example08_recruitment runs=50` was
  still silent, so `realistic_stress_observer_gpu` now prints solver route,
  amplitude/dispatch progress, and per-case completion timing/RSS.
- [x] Fix realistic mixed-pool route diagnostics and threshold. The `35693e5`
  Kaggle log confirmed `runs=50` is split into `single-cable B=25` and
  `double-cable B=25`, so the previous `B >= 32` shared threshold still missed
  the double-cable subgroup. The shared route threshold is now `B >= 16`, and
  realistic benchmark logs print solver routing per dispatch group.
- [x] Analyze the completed observer-only Kaggle artifacts against
  `realistic_stress` and `realistic_stress_single_vm`. Report:
  `benchmark/reports/compact_activation_observer_2026_06_20.md`. Decision:
  threshold/probes workflows are normal, but the generic double-cable observer
  path was not retained for the hot path exposed by `example08`: `B=100`
  warm mean is `503.184 s` versus `5.903 s` full Vm and `6.308 s` center Vm on
  P100.
- [x] Validate the packed VmRaster replacement on Kaggle P100
  `realistic_stress_observer_gpu`. Run:
  `benchmark/results/kaggle/20260620_144714_realistic_stress_observer_gpu_NvidiaTeslaP100`.
  Result: `example08` observer-only warm mean is `3.532 s` at `B=50` and
  `6.338 s` at `B=100`, versus `244.105 s` and `503.184 s` for the rejected
  observer path. Whole stress warm total is `38.59 s`, close to center Vm
  `39.09 s`; process peak is reduced from `20,906 MiB` to `5,325 MiB` at
  `B=100`.
- [x] Phase 7.6.7A implementation pass: add internal VmRaster lowering for
  threshold-style observer requests. The user concept stays simple:
  `observers=[Activation/Latency/ConductionBlock(...)]`; there are no public
  implementation-specific observer modes.
- [x] Phase 7.6.7B implementation pass: replace the active solver-side observer
  state with packed VmRaster state, `words[B, R, P, W] uint32`, where
  `W=ceil(Nt/32)`. The solver only writes threshold bits
  (`Vm >= threshold`) and returns `observations["vm_raster"]`; activation,
  latency, velocity, threshold-search, and recruitment summaries are
  post-processing. `PeakVoltage` remains post-hoc on recorded Vm.
- [x] Phase 7.6.7C instrumentation pass: VmRaster observer execution now exposes
  `runtime.prepare`, input materialization, `kernel.enqueue`, `kernel.wait`,
  and result finalization in the realistic profile CSV. The P100 validation run
  shows warm `B=100` is dominated by `runtime.prepare` (`3.585 s`) and
  `dispatch.build_plan` (`1.061 s`), not `kernel.wait` (`0.107 s`).
- [x] Phase 7.6.7D padded-group probe pass: VmRaster observers use row-aware
  static probe indices/masks and `-1` padded original indices, so heterogeneous
  padded groups can stay trace-free.
- [ ] Phase 7.6.7E post-processing pass: finish CPU-side decoders for
  `VmRasterResult`. Activation/recruitment decoding is now wired for the
  realistic observer route; remaining decoder breadth is first crossing/latency,
  conduction velocity, threshold-search updates, and public summary helpers
  without changing the solver contract.
- [x] Phase 7.6.7 validation, first P100 pass: compare full Vm, single-probe Vm,
  and VmRaster observer outputs on the realistic `example08` stress workload.
  VmRaster is retained: it is within center-Vm timing noise and removes the old
  observer slowdown.
- [ ] Phase 7.6.7 validation, broader pass: compare VmRaster decoders against
  full Vm for example 06 velocity and example 07 threshold, then run larger
  `Naxon`/diameter sweeps where output memory should matter most.
- [x] Run full/center/VmRaster recording-mode comparison on CPU and GPU.
  Artifacts:
  `benchmark/results/realistic_examples/recording_mode_compare/plots_to_review_20260620`.
  Warm totals: GPU `observer 37.08 s`, `center 38.44 s`, `full 38.49 s`;
  CPU `full 68.83 s`, `observer 69.19 s`, `center 71.61 s`. Conclusion:
  GPU `kernel.wait` is small; `runtime.prepare`, `dispatch.build_plan`,
  `kernel.enqueue`, and result/input packaging dominate.
- [x] Phase 7.6.5A first execution-reuse pass: cache stable dispatch plans,
  batch solver runtimes, prepared cohorts, and VmRaster probe plans for repeated
  runs over the same `AxonInstance` pool. The cache is intentionally tied to
  stable simulation objects so stimulus amplitudes can change without freezing
  the dynamic drive.
- [x] Phase 7.6.5A Kaggle validation on P100 observer stress:
  `benchmark/results/kaggle/20260620_183215_realistic_stress_observer_gpu_NvidiaTeslaP100`.
  Against the previous observer GPU stress run
  `20260620_144714_realistic_stress_observer_gpu_NvidiaTeslaP100`, example 08
  warm time dropped from `9.87 s` to `3.64 s` total (`2.71x`). `B=50` improved
  `3.53 s -> 1.35 s`; `B=100` improved `6.34 s -> 2.29 s`.
  Warm profile totals for example 08: `runtime.prepare`
  `16.51 s -> 2.33 s`, `dispatch.build_plan` `4.95 s -> 0.43 s`,
  `simulation.pool.total` `27.56 s -> 8.85 s`. Comparison plots:
  `benchmark/results/realistic_examples/execution_reuse_compare_20260620`.
- [x] Phase 7.6.5A Vext first factorization pass: shared point-source
  single-cable observer-only groups can pass `current_mid_A[Nt]` plus
  `footprint_mV_per_A[B,Nx]` into the VmRaster kernel instead of materializing
  dense `Vstim[B,Nt,Nx]`. The specialized sparse-Iinj VmRaster kernel applies
  the cable forcing operator once to the spatial footprint and scans only the
  temporal current. Local validation: dense and factorized builders match;
  dense and factorized VmRaster outputs match; dispatcher smoke confirms
  `inputs.extracellular input_format=factorized_point_source`.
- [ ] Phase 7.6.5A follow-up: add explicit input/Vext reuse for repeated
  amplitude or stimulus-only updates. Entry target: `example08_recruitment`
  should not rebuild spatial footprints or dense/factorized `Vext` structures
  for every amplitude when static shapes are unchanged.
- [ ] Phase 7.6.5B: split stable versus dynamic preparation in profile spans.
  Required visibility: planning, runtime construction/cache hit, footprint
  materialization/cache hit, stimulus sampling, dense/factorized `Vext`
  materialization, device transfer, enqueue, wait, and result packaging.
- [ ] Phase 7.6.5C: implement a stimulus/amplitude-only execution cache for
  point-source/extracellular sweeps. Reuse prepared spatial footprints and
  compiled executable; update only temporal stimulus/amplitude buffers where
  shapes match.
- [ ] Phase 7.6.5D: benchmark reuse using full/center/VmRaster output modes on
  CPU and GPU, then regenerate the recording-mode comparison plots. Success:
  reduce `runtime.prepare + dispatch.build_plan` by at least 30% on
  `example08` warm repeats without changing solver outputs.
- [ ] Phase 7.6.6: evaluate GPU dispatch scheduling only after Phase 7.6.5
  reuse work. Use `ideas/axonscope_dispatch_scheduling_gpu_note.md`: first test
  memory-aware bucket/coalesce of compatible groups, then optional async
  enqueue/wait. Do not rely on async for core speedups and do not enable it by
  default without memory-budget checks.
- [ ] Phase 7.7: clean stimulation and placement APIs after the first Vext pass.

## Phase 7.6.5 Execution-Envelope And Vext Plan

Goal: reduce complete workflow time now that solver-only custom-kernel work is
closed and VmRaster is validated. The working conclusion from the 2026-06-20
recording-mode comparison is that the GPU solver kernel is not the current
dominant cost. The dominant costs are the execution envelope around it:
`runtime.prepare`, dispatch/probe-plan rebuilds, `kernel.enqueue`, Vext/stimulus
materialization, result splitting, and public packaging.

Evidence:

```text
benchmark/results/realistic_examples/recording_mode_compare/plots_to_review_20260620
benchmark/results/realistic_examples/recording_mode_compare/kaggle_p100_cpu_gpu_recording_modes_20260620_allplots_*
```

Current benchmark diagnosis:

- GPU total stress warm time is about `37-38 s` across full, center, and
  VmRaster output modes.
- CPU total stress warm time is about `69-72 s`.
- For GPU, aggregate `kernel.wait` is less than `1 s`; `kernel.enqueue`,
  `runtime.prepare`, and `dispatch.build_plan` are much larger.
- For `example08 B=100`, GPU VmRaster spends about `3.44 s` in
  `runtime.prepare`, `1.00 s` in `dispatch.build_plan`, `0.61 s` in
  `kernel.enqueue`, and only `0.11 s` in `kernel.wait`.
- Therefore, optimize reuse and preparation before reopening solver kernels.

1. Baseline realistic workflows.
   - Run example 06 velocity, example 07 threshold, and example 08 recruitment
     with `benchmark/realistic_examples/bench_basic_examples.py`.
   - Compare CPU vs GPU by workflow, fiber type, run count, and population size.
   - Record build time, first run, warm run, backend, and devices.
   - Current Kaggle P100 stress evidence:
     `benchmark/results/kaggle/20260619_093205_realistic_stress_NvidiaTeslaP100`.

2. Add execution-envelope and `Vext` timing visibility.
   - Separate public object construction, extracellular footprint evaluation,
     dense `Vext` array materialization, host-to-device movement, solver time,
     and result packaging.
   - Keep measurements available in CSV/JSON, not only profiler traces.
   - Treat `runtime.prepare`, `dispatch.build_plan`, `kernel.enqueue`, result
     splitting, and GPU/CPU `kernel.wait` as first-class timings too; the
     recording-mode stress profile shows these dominate recruitment before raw
     GPU solver time does.
   - First pass implemented:
     - reuse the `solver_axon` already built by dispatch planning when preparing
       batch runtimes;
     - cache whole solver runtimes only for batch-safe paths where stimulation
       callables/precomputed drive tensors are not embedded in the runtime;
     - cache stable dispatch plans across repeated runs of the same
       `AxonInstance` pool;
     - cache prepared cohorts and VmRaster probe plans across amplitude sweeps,
       while keeping context/stimulus objects live;
     - cache shared point-source spatial footprints while keeping stimulus
       amplitudes live;
     - add `memory_estimate_*` and footprint-cache columns to realistic profile
       CSVs.
   - 2026-06-20 second pass implemented:
     - `dispatch.build_plan` now records `dispatch_plan_cache=hit|miss`;
     - `runtime.prepare` now records `batch_runtime_cache=hit|miss`;
     - `inputs.positions` now records `prepared_cohort_cache=hit|miss`;
     - new `observer.plan` span records `vm_raster_plan_cache=hit|miss`.
   - 2026-06-20 Kaggle P100 validation:
     - observer GPU stress run:
       `benchmark/results/kaggle/20260620_183215_realistic_stress_observer_gpu_NvidiaTeslaP100`;
     - example 08 warm total: `9.87 s -> 3.64 s` versus the previous observer
       GPU run (`2.71x`);
     - example 08 `runtime.prepare`: `16.51 s -> 2.33 s`;
     - example 08 `dispatch.build_plan`: `4.95 s -> 0.43 s`;
     - generated comparison CSV/plots:
       `benchmark/results/realistic_examples/execution_reuse_compare_20260620`.
   - 2026-06-20 Vext factorization local pass:
     - shared point-source single-cable observer-only batches now avoid dense
       `Vstim[B,Nt,Nx]` and pass `current_mid_A[Nt]` plus
       `footprint_mV_per_A[B,Nx]`;
     - the VmRaster sparse-Iinj kernel uses a precomputed spatial forcing
       footprint, then multiplies by the temporal current inside the scan;
     - local dispatcher smoke recorded
       `inputs.extracellular input_format=factorized_point_source`;
     - remaining work: reuse dynamic current buffers across amplitude sweeps and
       extend the same idea to double-cable RHS construction.
   - Local validation:
     `benchmark/results/realistic_examples/local_runtime_cache_smoke_local_smoke_profile.csv`.

3. Split stable and dynamic state.
   - Static across amplitude/stimulus-only updates:
     - axon cohort/grouping;
     - cable/membrane runtime;
     - dispatch group and padding plan;
     - VmRaster probe tables;
     - spatial extracellular footprints;
     - compiled executable for the same static shapes.
   - Dynamic across amplitude/stimulus-only updates:
     - temporal stimulus samples;
     - scalar amplitude/current values;
     - any dense/factorized input buffer whose values change but shape does not.
   - Make cache hits/misses explicit in benchmark metadata.
   - Add a reuse-failure explanation when a condition forces reprepare/recompile.

4. Reduce avoidable dense inputs.
   - Preserve the current public API while testing internal representations for
     shared point-source/electrode drives.
   - Avoid materializing dense zero `Iinj`.
   - Use factorized point-source `Vext` on hot observer-only paths when the
     spatial footprint is static and only the temporal current changes.
   - Reuse or cache `Vext` when protocols sweep only current amplitude.
   - Prefer observer-only outputs for activation/recruitment protocols when the
     user only needs compact decisions, reducing GPU-to-CPU movement and
     avoiding per-row `Vm` materialization.
   - Explore on-device/lazy `Vext` generation for analytical point sources.

5. Validate behavior.
   - Re-run unit tests for stimulation, dispatcher, protocols, and solvers.
   - Re-run relevant NRV comparisons if `Vext` semantics change.
   - Keep `pcr_adaptive` as the GPU solver baseline during Vext work.
   - Re-run the six-way full/center/VmRaster CPU+GPU comparison and compare
     `runtime.prepare`, `dispatch.build_plan`, `kernel.enqueue`, `kernel.wait`,
     and memory columns.

6. Decide next branch.
   - If runtime/dispatch/Vext reuse improves the envelope, continue toward
     study-level reuse policies.
   - If dispatch group count or launch overhead remains high, move to Phase
     7.6.6 bucket/coalesce scheduling.
   - If solver time becomes dominant again, reopen custom kernels only with a
     clear validation gate and a target device that supports the required stack.

## Phase 7.6.6 GPU Dispatch Scheduling

Reference note: `ideas/axonscope_dispatch_scheduling_gpu_note.md`.

Goal: improve GPU throughput by launching fewer, larger compatible JAX calls
and, only after that, testing optional async enqueue/wait scheduling for
remaining independent groups. This is a dispatch/planning phase, not a solver
replacement.

Latest 2026-06-20 interpretation:

- Do not start with async scheduling. JAX async enqueue may avoid host-side
  bubbles, but the GPU may still serialize work and async keeps more inputs and
  outputs alive.
- Start with better reuse and, after that, conservative bucket/coalesce
  scheduling.
- Treat hardware memory capacity versus estimated simulation/output bytes as an
  entry condition before coalescing or async pending groups.
- Full Vm output should use stricter pending-memory limits than VmRaster
  observer-only output.

Entry gate:

- [x] Use `realistic_examples_*_profile.csv` and hotpath traces to confirm that
  dispatch group count, repeated `kernel.wait`, input preparation, or
  `results.split_batch` are meaningful bottlenecks. Current finding:
  input/runtime preparation, dispatch plan rebuilds, and enqueue dominate GPU
  more than raw `kernel.wait`; CPU `kernel.wait` still matters.
- [ ] Compare available hardware capacity, especially GPU memory, against the
  estimated memory cost of each simulation/bucket before enabling coalescing or
  async scheduling.
- [ ] Re-check this gate on larger heterogeneous pools after Phase 7.6.5 reuse:
  current P100 `realistic_stress` evidence mostly has one dispatch group per
  simulation call, with mixed recruitment at two groups, so scheduling is not
  yet the immediate bottleneck.
- [x] Keep Phase 7.6.5 execution-envelope profiling as the immediate source of
  truth before changing dispatch architecture.

Implementation plan:

1. Add conservative execution bucket keys.
   - Start with `mode`, resolved solver/backend, `Nx` bucket, dtype,
     recording mode, and geometry compatibility.
   - Keep scalar fallback groups out of bucket coalescing.

2. Add `Nx` bucketing and padding as an explicit scheduling policy.
   - Target buckets: `32`, `64`, `128`, then `256` only if profiling demands it.
   - Consider a `96` bucket if padding from `65 -> 128` is too expensive.
   - Slice padded outputs back to original rows and keep observers blind to
     padded compartments.

3. Coalesce compatible groups before considering concurrency.
   - Prefer one larger JAX call over many small calls when safety rules match.
   - Track original group count, bucket count, coalesced group count, effective
     batch size, available device memory, estimated simulation memory, and
     estimated output bytes.

4. Prototype optional async group scheduling behind an explicit option.
   - Split batch execution into prepare/enqueue/wait/finalize steps.
   - Add `PendingGroup` plus memory-pressure flushing.
   - Track `max_pending_groups`, `max_pending_output_bytes`,
     `async_flush_count`, and `async_pending_max`.
   - Do not enable async by default until benchmarks show stable wins.

5. Add a dedicated scheduler benchmark.
   - Create `benchmark/dispatcher/bench_group_scheduling.py`.
   - Compare `sync_current`, `async_groups`, `coalesce_buckets`, and
     `coalesce_buckets_async`.
   - Cover many small compatible groups, semi-compatible `Nx` groups, mixed
     single/double groups, full recording, center recording, and observer-only
     output.
   - Report memory budget versus estimated memory cost per bucket, including
     inputs, outputs, padded rows, retained traces, and pending async groups.

Success criteria:

- [ ] Keep bucket coalescing if it improves total wall time by at least about
  20% or materially reduces JIT call count without memory regressions.
- [ ] Keep async scheduling only if it improves total wall time by at least
  about 10% on relevant GPU workloads and peak memory remains acceptable.
- [ ] Reject or downshift any scheduling policy when estimated memory pressure is
  too close to the available hardware budget.
- [ ] Keep all changes internal to dispatch/runtime options until the public
  API story is clear.

## Phase 7.6.7 VmRaster Observer Redesign

Goal: keep the public observer/analysis story simple while making the first
solver-side observer implementation deliberately optimized for the workflows
AxonScope needs most in examples 06/07/08: velocity, threshold, activation, and
recruitment-style sweeps over many axons.

Evidence gate:

- [x] Kaggle P100
  `20260620_111038_realistic_stress_observer_gpu_NvidiaTeslaP100` confirms the
  generic double-cable observer path is unsuitable for the hot path exposed by
  the recruitment-style stress case. The mixed `B=100` case spends about
  `62.7 s` per double-cable subgroup amplitude, while full/center Vm output
  spends about `0.5 s` for the same double-cable subgroup shape.
- [x] The slowdown is isolated to the double-cable observer path exercised by
  observer-only `example08`. The single-cable observer subgroup stays around
  `50 ms` per amplitude, and example 06/07 timings remain comparable to earlier
  GPU stress runs.
- [x] Treat process `peak_rss` as a high-water diagnostic, not a direct retained
  output size. The observer run reached `20,906 MiB` process peak at `B=100`,
  but case-local deltas and profiler memory estimates point toward XLA/cache or
  hidden compile/execution pressure rather than raw `Vm` output arrays.
- [x] Kaggle P100
  `20260620_144714_realistic_stress_observer_gpu_NvidiaTeslaP100` validates the
  packed VmRaster replacement. `example08` observer-only warm mean is
  `6.338 s` at `B=100`, close to center Vm `6.308 s` and far below the rejected
  observer path `503.184 s`. The double-cable subgroup warm mean is `561.5 ms`,
  about `1.1x` full Vm, and process peak falls to `5,325 MiB`.

User-facing rule:

- [x] Keep one analysis-oriented concept for solver-side observations, centered
  on scientific requests such as activation/first-crossing time, plus trace-free
  simulation output when the user does not request Vm traces.
- [x] Do not introduce public implementation-specific observer modes while the
  API is still pre-release. A user should ask for the scientific result, not
  choose an internal kernel family.
- [x] Document that the first hot-path observer is intentionally strict:
  membrane-voltage threshold events only, with fixed target positions/probes.

Implementation direction:

1. Define VmRaster lowering.
   - Done: lower simple membrane-voltage threshold definitions to an internal
     VmRaster plan.
   - Supported first shape: one or more threshold/probe definitions, scalar
     thresholds, fixed target indices/probes, and static raster/probe slots.
   - Done: padded/heterogeneous groups lower to row-aware static probe tables
     and padded original indices are masked with `-1`.
   - Done: broader `PeakVoltage` solver-side observer support was removed.
     Peak voltage remains available as post-hoc analysis on recorded Vm.

2. Define the packed raster state.
   - Done: minimal state is `words[B, R, P, W] uint32` for static raster count
     `R`, probe count `P`, and `W=ceil(Nt/32)`.
   - Done: one bit is written per solver step/probe when `Vm >= threshold`.
   - Started: activation/recruitment can decode from `VmRasterResult`.
   - Remaining: derive latency, velocity, threshold-search, and public summary
     helpers from `VmRasterResult` in CPU post-processing.
   - Per-step input: `Vm[B, Nx]`.
   - Output: `observations["vm_raster"]`, not `Vm[Nt, Nx, Naxon]`.

3. Keep reductions fixed and JAX-friendly.
   - Center/probe mode: set bit from `Vm[:, idx] >= threshold`.
   - Small fixed probe set: set bits independently so velocity can decode
     per-probe first crossing times afterward.
   - Whole-axon activation should be represented by explicit fixed probes for
     now; do not add broad reductions until benchmarked.
   - Avoid arbitrary Python callbacks, dynamic output tables, or rich metadata
     in the solver loop.

4. Preserve solver fast paths.
   - Done locally: the active observer update no longer uses the old generic
     observer state or the short-lived compact event state.
   - It must be independent of recording mode selection in dispatch routing.
   - Done locally: it works with `pcr_soa` batch-native paths for mixed groups
     such as `B=25`; validate next on Kaggle.
   - Done locally: it avoids the previous conservative center/probe padded-group behavior
     where `_kernel_batch_options(...)` forces `BatchRecording.full()` before
     the solver kernel and only slices back to the requested probe afterward.
   - Done for the first P100 stress validation: VmRaster observer-only warm time
     is within center-Vm noise for `example08`, and double-cable subgroup timing
     is about `1.1x` retained-Vm timing.

5. Benchmark as a memory feature first.
   - Compare full Vm, single-probe Vm, and VmRaster observer on the
     realistic examples.
   - Track real process RSS deltas, device memory estimates, output bytes, and
     solver/profile timings.
   - Stress large axon counts and diameter sweeps, because memory savings should
     matter most there.
   - First P100 result: retained simulation arrays are estimated at `0.94 MiB`
     for `B=50` and `1.88 MiB` for `B=100`; process RSS is now dominated by
     runtime/JAX/cache overhead rather than the packed raster itself.

6. Migrate the public learning surface in the same phase.
   - Done locally: observer runtime, solver, dispatcher, public facade,
     performance, hotpath catalog, and analysis tests were updated.
   - Done locally: `example_14_hotpath_benchmarking.py` and
     `example_18_solver_side_observers.py` now use VmRaster observer-only output.
   - Done locally: the old generic solver-side observer runtime was deleted
     from active code; do not reintroduce it as a fallback.
   - Remaining: update any longer-form benchmark docs after the next Kaggle
     validation run.

Success criteria:

- [ ] Observer-only execution is clearly lighter than full Vm output for large
  `Naxon` workloads that only need event outputs.
- [ ] Observer-only execution does not materially slow the retained-Vm fast
  path for the same solver route.
- [ ] The public API remains one clean analysis concept with documented
  constraints.
- [ ] All observer-related tests, examples, benchmark docs, and roadmap notes
  are updated in the same change set as the new observer contract.
- [ ] Any broader observer system stays out of the hot path until there is
  benchmark evidence and a concrete user need.

## Solver Campaign References

- Summary report: `benchmark/reports/double_cable_solver_optimization_2026_06.md`
- Plot: `benchmark/reports/double_cable_solver_optimization_2026_06_speedups.svg`
- Compact Vm-event observer report:
  `benchmark/reports/compact_activation_observer_2026_06_20.md`
- Compact Vm-event observer plot:
  `benchmark/reports/compact_activation_observer_2026_06_20.svg`
- Active solver README: `benchmark/solvers/README.md`
- Kaggle runner README: `benchmark/kaggle/README.md`
- Solver roadmap archive: `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`
- Dispatch scheduling note: `ideas/axonscope_dispatch_scheduling_gpu_note.md`

Archived experiment locations:

- `benchmark/archived_solver_spikes/`
- `benchmark/triton_solver/`
- `benchmark/jax_triton_solver/`
- `benchmark/cuda_ffi_solver/`
- `tests/archive/solver_spikes/`

## Phase 7.7 Stimulation And Placement API Cleanup

Goal: make the public API match the product boundary before Phase 8 studies.

- [ ] Re-read `GUIDELINES.md` before implementation.
- [ ] Audit public stimulation/context API names after Vext work clarifies the
  internal representation.
- [ ] Keep user-facing examples simple: clamps, point-source electrodes,
  extracellular drives, footprints, stimulation protocols, and populations.
- [ ] Avoid exposing solver/backend implementation details in public examples.

## Phase 7.8 Examples Learning Path

- [ ] Update basic examples after Vext/API cleanup.
- [ ] Add a solver-options example only for retained public options:
  `auto`, `thomas`, `pcr`, `pcr_soa`, `pcr_adaptive`.
- [ ] Do not add pseudo-double or custom-kernel examples unless a candidate
  leaves standby and becomes public.

## Phase 8 Studies

- [ ] Add callable study objects for threshold curves, recruitment curves,
  conduction validation, and parameter sweeps.
- [ ] Define reuse policies for prepared populations and stimulation contexts.
- [ ] Define retention policies for recordings and derived analysis outputs.

## Phase 9 Serialization And Reference Backend

- [ ] Finalize serialization schemas for public objects.
- [ ] Add NumPy/reference backend validation where it improves trust in JAX
  lowering or custom kernels.

## Recent Verification

- 2026-06-15: non-NRV unit run after dispatch cleanup: `314 passed, 1 skipped`.
- 2026-06-18: solver optimization campaign closed; active solver surface cleaned.

Update this section only with high-signal final checks, not every exploratory
benchmark run.
