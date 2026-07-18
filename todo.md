# AxonScope TODO

Living execution plan for AxonScope. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh benchmark
artifacts own current behavior. Historical detail before the 2026-07-12
cleanup remains in `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.

## Snapshot

Updated on 2026-07-17 after validating compact activation at P14E scale and
implementing compact first-crossing latency.

- P7, P11, P12, the VmRaster part of P13, and the P14 performance gate are
  closed. The remaining P14B architecture items are retained as non-blocking
  convergence work rather than silently discarded.
- The production runtime is JAX. CPU double-cable uses Thomas; GPU
  double-cable uses the Triton tiled-Thomas route; single-cable uses the JAX
  tridiagonal route.
- The strict solver-side threshold route now has three retention policies:
  compact activation keeps one boolean per row and definition, compact latency
  keeps one first-crossing `int32`, and VmRaster remains the temporal reference
  for analyses that need crossing history.
- Recruitment now keeps one source population and a native numerical amplitude
  axis; it no longer expands `Namplitude x Naxon` Python objects. Large sweeps
  use compact activation when their criterion is activation-only. P15 still
  needs bounded crossing, spike, and propagation event states.
- Work proceeds in dependency order:
  1. P15 compact activation, spike, and propagation observers.
  2. P16 low-level JAX temporal solver and dispatch optimization.
  3. P17 autonomous generated membrane runtime contracts.
  4. P18 membrane-model completion and validation.
  5. P19 pre-v1 cleanup and public-surface convergence.

Latest fast validation recorded by the compact-activation checkpoint:

```text
python -m compileall -q src tests/unit
pytest -q tests/unit --tb=short
705 passed, 1 skipped
```

## Non-Negotiables

- AxonScope is pre-release with one active user. Prefer direct convergence over
  compatibility shims, aliases, deprecated wrappers, or parallel old/new
  paths.
- Keep one concept, one public name, one execution path, and one canonical
  public result model.
- Promote optimizations by replacing the existing production path, not by
  adding parallel `optimized`/`legacy` routes, hidden slow fallbacks, duplicate
  kernels, or extra public switches. Benchmark-only prototypes may coexist
  temporarily until a decision is made; remove the rejected/replaced route
  when the optimized path is promoted.
- Internal representations may change freely when public behavior is
  preserved. If an optimization requires a material change to canonical
  results, serialized output, user-visible progress, or the public workflow/UI,
  discuss and approve that product change before implementation.
- Public orchestration enters JAX through `axonscope.runtime.execution`.
- Public examples must not import runtime or solver internals.
- External packages own nerve geometry, trajectories, world coordinates,
  electrode CAD, and FEM solves. AxonScope owns intrinsic axon coordinates,
  sampled-footprint stimulation, membrane/cable execution, recording, and
  analysis.
- Every public feature must be documented in a runnable example or removed
  from the public surface.
- Do not remove an unfinished task unless it is completed, rejected, or moved
  to a named tracking document.

## Active Roadmap

### P14 - Population, Sweep, And Preparation Scalability

Primary objective: remove work that scales with Python object count before
optimizing the temporal solver. A recruitment sweep must keep one source
population plus a native amplitude dimension; it must not represent every
`amplitude x axon` pair as a cloned `AxonInstance` and stimulus graph.

#### P14A - Reproducible realistic baseline

- [ ] Profile the apparent idle startup before `examples/basic/08` begins
  visible recruitment work, especially with `fibers_per_family=1000`. Split
  template/model construction, population expansion, analytical footprint
  sampling, runnable/sweep-plan construction, first dispatch/signatures,
  runtime and membrane lowering, JIT, and first solve. Add global wall spans
  where the current benchmark is blind, then validate locally and on Kaggle
  CPU/GPU before deciding which work moves behind the lazy runner boundary.
  - [x] Add the standalone `benchmark/examples/basic_08_startup.py` probe;
    keep the public example unchanged. A warm-module-cache local CPU startup
    run with 1000 fibers per family records `16.389 s` before the sweep:
    `3.399 s` importing modules and `12.990 s` constructing the workload. The
    latter contains `9.046 s` constructing 1000 MRG axons, `1.425 s`
    constructing 1000 Rattay-Aberham axons, and `1.257 s` sampling the 2000
    analytical footprints. Artifacts:
    `benchmark/results/basic_08_startup_local_20260715_f1000_timing` and
    `benchmark/results/basic_08_startup_local_20260715_f1000_profile`.
  - [ ] Run the dedicated probe through `first-amplitude` and `full` at
    realistic scale to attribute plan construction, runtime preparation, JIT,
    first solve, and warm amplitudes; repeat on Kaggle CPU/GPU.
    Local `first-amplitude` validation now covers 1000 fibers per family:
    startup `2.797 s`, `recruitment_sweep=4.529 s`,
    `simulation.run_pool=3.069 s`, and 43/2000 activated rows at 5 uA. Full
    multi-amplitude and Kaggle CPU/GPU evidence remain open.

- [x] Add the `p14_realistic` recruitment workload: 21 amplitudes, `3 ms`,
  `dt=0.001 ms`, single-cable Rattay-Aberham with `Nx=200`, and double-cable
  MRG with padded `Nx=74`.
- [x] Run Kaggle P100 baselines for single/double cable at 196, 1024, and 4096
  axons with amplitude policies `1`, `2`, and `full` where memory allows.
- [x] Analyze the 1024/4096 runs using non-overlapping timing spans. Warm
  policy-1 `simulation.run_pool` is `24.80/91.60 s` for single 1024/4096 and
  `14.59/47.06 s` for double 1024/4096. `kernel.dispatch_jax` is about 82% of
  `run_pool`, but still includes deferred GPU execution and VmRaster updates.
- [x] Identify the large-batch memory failure. With `Nt=3000`, full VmRaster
  state is `6.02 GiB` for single 4096 x 21 amplitudes and `2.23 GiB` for
  double. The single executable exposes about `14.03 GB` of input/output
  arguments and XLA constant-folding over `[86016, 200]`.
- [x] Record the baseline artifacts:
  `benchmark/results/kaggle/20260715_161913_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14-realistic-single-1024-cbb0ae4`,
  `benchmark/results/kaggle/20260715_161913_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14-realistic-double-1024-cbb0ae4`,
  `benchmark/results/kaggle/20260715_162614_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14-realistic-single-4096-cbb0ae4`,
  and
  `benchmark/results/kaggle/20260715_162614_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14-realistic-double-4096-cbb0ae4`.

#### P14B - Shared immutable axon templates

Architecture contract: Python objects are the immutable scientific description;
execution uses one canonical description-to-array lowering. Materialize compact
backend-neutral NumPy tables first, then let the selected runtime place/lower
those arrays to JAX or another backend. Optimizations must be selected from
typed capabilities and structural/parameter signatures, never concrete axon or
membrane class names.

- [x] Define and enforce the ownership contract: `Axon` is an immutable,
  reusable structural template; `AxonInstance` owns per-fiber stimulation and
  mutable simulation state. Audit axon/layout/membrane fields, solver-axon
  construction, dispatch signatures, and caches for hidden mutation. `Axon`
  and `Layout` now reject post-construction reassignment, section and membrane
  descriptions were already frozen, flattened arrays are read-only, and
  instance-level stimulation/overrides remain separate.
- [ ] Split the population-construction profile into axon geometry/layout,
  membrane source/model construction, diameter/section parameter derivation,
  solver-axon lowering, signatures/code generation, and row stacking. Report
  object counts and unique structural hashes as well as elapsed time.
- [ ] Define a model-agnostic population materialization contract containing
  padded or bucketed geometry/cable arrays, section or structure codes, unique
  membrane parameter rows, row/compartment indices, masks, and stable
  signatures. It must represent every current `Axon`/`Layout`, not only MRG.
- [ ] Replace per-row/per-section Python materialization with one canonical
  NumPy lowering over unique descriptions. Repeated population rows should
  gather from numerical tables rather than instantiate duplicate `Section`,
  `LayoutElement`, `FlattenedLayout`, or `SolverAxon` object graphs.
- [ ] Keep backend ownership strict: backend-neutral preparation may use NumPy,
  but only `runtime/jax` creates JAX arrays, transfers to device, compiles, or
  launches kernels. Do not use JAX merely to build host-side descriptions.
- [ ] Make work outside execution lazy by default. Construction should retain
  only lightweight immutable Python descriptions; numerical materialization,
  signatures, code generation, and backend preparation should be triggered by
  `run()` or by an explicit `estimate()`/`inspect()` request, then cached from
  the same canonical lowering. Do not eagerly perform execution-only work while
  assembling an axon, population, stimulation, or protocol.
  - Canonicalize and validate unit-bearing structural parameters once per
    unique scientific template before numerical population expansion. Rows
    should reference canonical unitless values/template indices; they must not
    repeat Pint conversion, diameter quantization, membrane construction, or
    layout construction. This lowering remains model agnostic and feeds the
    existing `MaterializedAxonRows` execution route.
- [ ] Evaluate a unified runnable-plan architecture: axons, populations,
  recruitment curves, sweeps, and studies should compose immutable simulation
  plans; one runner should execute one or many plans and own materialization,
  grouping, scheduling, runtime selection, compilation, and caches. Everything
  outside the runner remains lazy. Before adopting it, map the concept onto the
  canonical `AxonSimulation(...).run()` workflow and discuss any public API,
  result, progress UI, or output change rather than introducing a parallel path.
- [ ] Preserve public inspection and result semantics. If eager descriptive
  layouts become expensive, materialize their public view lazily from the same
  canonical numerical representation; discuss any observable API/UI/output
  change before implementation.
- [ ] Add architecture guards forbidding concrete axon/model family checks in
  generic materialization and runtime modules. Validate at least uniform
  single-cable, heterogeneous custom layout, shifted myelinated double-cable,
  and stateful membrane cases through the same lowering contract.
- [x] Make the shared membrane contract explicit. A 4096-MRG population should
  compile/load each distinct membrane equation model once; diameter- and
  section-specific values should be parameter data over that shared model, not
  thousands of reconstructed `MembraneModel`, Model IR, or `JaxMembraneProgram`
  objects. The existing model-agnostic gated/leak stack now receives shared
  solver templates: at 4096 rows it reports three unique encoded rows, 4093
  cache hits, six host-side leak signatures, and one compiled JAX gated model.
- [ ] Represent repeated membrane parameterizations as unique parameter rows
  plus row/compartment indices where this reduces construction and stacking
  cost. Keep this capability model agnostic: derive uniqueness from the
  membrane contract and parameters, never from an `MRG` name check. The
  membrane compiler and generated runtime modules are optimization surfaces:
  change code generation when useful so `jax_model.py` can emit batch-aware
  initialization, parameter tables, or lowering helpers directly instead of
  making the runtime reconstruct model-specific facts.
- [x] Add focused tests proving that many `AxonInstance` rows may share one
  axon template without state leakage, changed stimulation, changed results,
  or invalid cache reuse. Shared and distinct double-cable populations now
  produce equivalent results while preserving per-instance stimulation.
- [ ] Add a controlled 4096-MRG A/B benchmark: current construction of 4096
  equivalent MRG objects versus three shared templates for diameters
  `7.3/10.0/12.8 um`. Measure population construction, peak host memory,
  membrane/model object counts, unique parameter rows, `dispatch.build_plan`,
  first `runtime.prepare`, warm `run_pool`, and exact recruitment equivalence.
- [x] Promote template sharing into the realistic workload and applicable
  public population/integration paths. Keep transverse placement and sampled
  footprints row-specific. The P14 workload shares exact MRG templates, and
  `population_from_nrv` uses a local complete-constructor-key cache while
  retaining row-specific NRV metadata and footprints.
- [x] Keep full `Axon` objects explicit, but reuse immutable derived layouts
  only after defining a complete key and bounded lifecycle. The canonical MRG
  layout route now uses a 256-entry cache keyed by quantized diameter, length,
  nodes, compartment layout, intrinsic shift, temperature, morphology mode,
  and every physical geometry/myelin parameter. Custom membrane assignments
  bypass it. Uniform unmyelinated layouts and immutable membrane descriptors
  use corresponding bounded model-agnostic caches, while generic public
  `Unmyelinated(membrane=...)` construction preserves the exact user-provided
  membrane object.
- [x] Reduce the 4096 double-cable population-construction bottleneck that was
  about `68.5 s` in the previous Kaggle run. The first local shared-template
  pass builds 4096 rows in `4.56 s` and `dispatch.build_plan` in `0.143 s`, with
  three axon templates and three `SolverAxon` objects. Confirm the ratio on the
  same Kaggle GPU as part of the full P14 acceptance run.

Local construction evidence on 2026-07-15: at 196 MRG rows, exact-template
sharing reduced population construction from `4.67 s` to `0.292 s` and
`dispatch.build_plan` from `0.200 s` to `0.0048 s`. Shared construction measured
`1.58 s` at 1024 rows and `4.56 s` at 4096 rows. `Layout` now owns one canonical
read-only flattened representation, so shared templates no longer repeat
per-compartment layout lowering for footprint sampling or solver preparation.
The 4096-row gated/leak membrane stack takes `0.077 s` after its first compile
and materializes `8.09 MiB` of initial gate/parameter state. It already
deduplicates preparation by membrane signatures; P14B's remaining unique-row
task is specifically about whether carrying a parameter table plus row indices
into the kernel beats materializing the per-fiber initial state.

Basic-08 construction evidence after bounded template/model reuse is recorded
under `benchmark/results/basic_08_startup_local_20260715_f1000_final_r1` and
`benchmark/results/basic_08_first_amplitude_local_20260715_f1000_final`. At
1000 fibers per family, pre-sweep workload construction dropped from
`12.990 s` to `2.813 s` (4.62x) and total startup from `16.389 s` to `6.054 s`
(2.71x). MRG construction dropped from `9.046 s` to `0.385 s` (23.5x), while
Rattay-Aberham construction dropped from `1.425 s` to `0.677 s` (2.10x).

The lazy-template A/B probe is recorded under
`benchmark/results/basic_08_startup_local_20260715_f1000_shared_templates` and
`benchmark/results/basic_08_first_amplitude_local_20260715_f1000_{distinct,shared}_ab`.
Vectorized unit validation and diameter quantization cost `0.139 ms` and reduce
2000 eagerly distinct axons to 63 Rattay-Aberham plus three MRG templates while
preserving `43/2000` activations at 5 uA. Pre-sweep construction drops from
`2.835 s` to `1.765 s`; on the cold first-amplitude run,
`dispatch.build_plan` drops from `467.0 ms` to `173.7 ms`, `runtime.prepare`
from `2046.0 ms` to `278.9 ms`, and `simulation.run_pool` from `4.241 s` to
`2.348 s`. Promote this as lazy, model-agnostic description-to-row lowering;
do not retain the benchmark's family-specific template dictionaries as a
second production path.

The first production convergence slice now gives `AxonPopulation` the same
canonical descriptive shape: `axon_templates` in first-occurrence order plus
`row_template_indices`, while retaining distinct row instances and
stimulations. `examples/basic/08` canonicalizes its two diameter arrays once,
constructs 63 Rattay-Aberham and three MRG templates, and computes intrinsic
positions once per template. The production-backed startup probe at
`benchmark/results/basic_08_startup_local_20260715_f1000_canonical_templates`
measures `0.132 ms` for vectorized unit/diameter canonicalization, `0.756 ms`
for population template indexing, and `1.445 s` total pre-sweep construction;
`1.201 s` of that remainder is the intentionally row-specific analytical
footprint test workload. A full local public-example run retained recruitment
counts `43 141 403 652 843 983 1245 1319`. The next slice must pass this
template table through protocol/dispatch preparation directly and move
execution-only numerical derivation behind the runner; do not add a separate
population execution path.

Shifted-template scaling on local CPU is recorded under
`benchmark/results/p14_mrg_template_scaling_local_20260715`. At 4096 axons,
increasing exact `(diameter, x_shift)` templates from `3/11/32/128/512/1024`
raises population construction from `4.49/4.57/4.98/7.09/17.54/36.78 s`.
`dispatch.build_plan` remains `0.15-0.32 s`, membrane stacking remains
`0.20-0.43 s`, and 1024 cable templates collapse to only 57 unique membrane
rows. Optimize shifted MRG geometry/layout construction before membrane codegen.

First canonical NumPy-row lowering on 2026-07-15: `PreparedCohort` now builds
one read-only `MaterializedAxonRows` table inside `runtime.prepare`; positions,
single/double-cable coefficients, capacitance rows, membrane area, and
extracellular cable arrays consume that table. Cable and extracellular values
are computed vectorially over unique templates, gathered into population order,
then transferred once by `runtime/jax`; the former per-`SolverAxon` stacking
loops were removed. A local 1024-axon/128-shifted-template double-cable smoke
under `benchmark/results/p14_materialized_rows_local_20260715_v2` measured cold
`materialize_axons=6.47 ms`, `stack_cable=4.04 ms`, and
`stack_extracellular=6.46 ms`; warm materialization was `3.09 ms`. The same cold
run still spent `309.07 ms` in representative base runtime and `247.78 ms` in
membrane stacking, while population construction outside the runner took
`6.49 s` on the noisy local laptop. The next convergence step is membrane
parameter-row lowering and lazy plan construction, not another cable path.

The first membrane-row lowering pass now builds a backend-neutral
`MembraneRowPlan` with population-to-parameter indices, unique descriptive
rows, seven unique model signatures, and per-compartment model indices. On the
same 1024-axon/128-template workload it found 45 membrane rows and 979 reuse
hits. The JAX gated/leak stack encodes only those rows, gathers dynamic initial
state once, reuses compiled representative executables, and no longer retries
the deleted `from_runtime_rows` specialization. Local no-memory-trace evidence
under `benchmark/results/p14_membrane_host_leak_final_local_20260715` reduced
`stack_membrane` from `247.78 ms` to `25.29 ms` (`9.8x`); combined axon/membrane
materialization fell from `254.25 ms` to `32.30 ms` (`7.9x`) with identical
activation output. Remaining measured membrane-stack work is six generic host
leak lowerings (`16.54 ms`); prefer emitting their scalar conductance terms in
the generated runtime contract over adding a model-family shortcut.

#### P14C - Generic native numeric execution axis

- [x] Replace `_build_native_amplitude_pool` and
  `_refresh_native_amplitude_pool` as the execution representation. Keep a
  typed plan containing `source_pool[Naxon]`, `values[Namplitude]`, update or
  scale semantics, ordering, and chunk boundaries without constructing
  `Namplitude x Naxon` Python objects.
- [x] Keep `recruitment_sweep` as a small user-facing protocol. It defines the
  values, criterion, progress, and result contract; dispatcher/runtime code
  owns grouping, lowering, device execution, and scheduling.
- [x] Lower amplitude-only point-source changes as temporal current scales or
  indices over one prepared footprint cohort. Do not recreate
  `Stimulus`, `Drive`, `ExtracellularStimulation`, or `AxonInstance` objects
  for every value.
  - `ExtracellularWaveformUpdate` evaluates one complete waveform factory per
    value, never per axon. The runtime samples those waveforms into
    `current_mid_A[Namplitude, Nt]`, maps amplitude-major logical rows through
    `current_row_indices`, and reuses factorized spatial footprints. It never
    assumes the value is a global scale, so independently varying phases and
    timings remain distinct.
- [x] Promote the current amplitude/waveform mechanism into one generic,
  planning/runtime-owned numeric execution axis. A protocol supplies source
  simulations, ordered axis values, typed dynamic inputs, and result indices;
  planning, dispatch, lowering, and execution consume that contract without
  depending on recruitment result types or recruitment-specific orchestration.
  `recruitment_sweep` must remain only one user-facing client of this path.
- [ ] Generalize and validate that axis beyond MRG and recruitment. Keep it
  membrane-model agnostic and support both cable formulations, variable
  diameters, variable footprints, independently varying waveform phases and
  timings, and every stimulus family whose dynamic inputs can preserve one
  static execution contract. Reject static-contract changes explicitly.
  - [x] Support a selected waveform axis in multi-drive extracellular rows.
    Preparation records each source row's immutable drive waveforms and the
    selected `drive_id`; lowering replaces only that numerical waveform while
    retaining every other drive. Single- and double-cable compact factorized
    kernels consume the resulting `[axis, source, drive, time]` values without
    constructing axis-by-source simulation objects or falling back to dense
    `Vext`.
  - [x] Compress repeated multi-drive temporal rows as unique current patterns
    `current_mid_A[U, S, Nt]` plus `current_row_indices[B]`. The canonical
    factorized payload, dense reference materializer, and CPU/GPU kernel input
    contract now support the same indexed rank-S representation; arbitrary
    complete waveforms remain supported, including independently changing
    positive/negative phases and timings.
  - [x] Validate the multi-drive route on Kaggle P100, additional membrane
    models, heterogeneous diameters/footprints, and longer chunks. CPU tests
    cover exact waveform-table materialization, dispatcher subgroup slicing,
    single- and double-cable recruitment, and compact factorized versus dense
    double-cable VmRaster equivalence. The P100 artifact ending in
    `axs-p14c-multidrive-p100-400cf49` covers 16 Rattay-Aberham and 16 MRG axons,
    two distinct footprints, four amplitudes, and chunk sizes `1/full`. Every
    cold/warm policy returned `0 18 20 21`; route guards confirmed factorized
    single/double inputs, two retained drives, and `jax_triton_loop_xb` for
    double cable. The independent dense Triton check passed at `1.439e-7`
    maximum absolute error. Warm `simulation.run_pool` was `1.910 s` for four
    size-1 chunks and `0.563 s` full (`3.39x`), while full warm `kernel.wait`
    was `14.0 ms` (`2.49%` of run-pool time).
  - [x] Profile the matching asynchronous P100 route to distinguish device work
    from host dispatch. The artifact ending in
    `axs-p14c-dispatch-profile-p100-400cf49` records a warm full
    `kernel.dispatch_jax=531.8 ms`, final `kernel.wait=0.12 ms`, and `330.9 ms`
    of events on the serial GPU compute stream. Thus dispatch includes solver
    execution through queue backpressure; it is not pure launch overhead. The
    stream contains 67,675 events, led by 3,072 fused double-cable Triton solves
    (`136.4 ms`) and 3,000 single-cable PCR loop solves (`25.1 ms`) plus first
    passes (`9.4 ms`). Use the reusable
    `benchmark/analysis/jax_perfetto_summary.py` reader for future traces. Check
    whether kernel fusion or fewer launches remains valuable at realistic large
    populations; never optimize `kernel.wait` in isolation from the device
    timeline.
    - The 1024-axon follow-up ending in
      `axs-p14c-run-pool-profile-1024-p100-9725f34-v2` profiles only the first
      canonical `simulation.run_pool`, avoiding the one-million-event host
      saturation seen when profiling the whole sweep. Warm run-pool is
      `2.757 s`; the GPU compute stream contains 63,811 events totaling
      `898.7 ms`, while `kernel.wait` is only `68.7 ms`. Host attribution is
      `runtime.prepare=1.047 s`, numeric-axis lowering about `0.339 s`,
      extracellular preparation `0.266 s`, and `kernel.enqueue=1.013 s`.
      Runtime signature construction accounts for about `0.971 s` of prepare,
      including `0.850 s` in repeated `repr`, directly confirming P14D's
      trusted-signature task as the next high-value host optimization.
  - [ ] Add typed numeric-axis inputs for other stimulus families only where
    their dynamic fields preserve one static execution contract. Changing the
    number of drives remains an explicit rejection for one prepared axis.
- [x] Reuse the same generic axis executor from other protocols/studies that
  vary compatible numerical stimulus inputs. Do not add a second
  recruitment-only executor, protocol-specific dispatcher route, or hidden
  expanded-object fallback.
- [ ] Preserve explicit amplitude chunking during optimization, including `1`,
  small bounded groups, and `full`. Pad or mask the final group where that
  avoids a second compiled shape without changing result ordering.
- [ ] Only after P15 compact observers are implemented and the generic numeric
  axis is optimized and validated, define user-facing chunk and memory-budget
  policies from measured CPU/GPU costs. Do not freeze an automatic heuristic
  while preparation, dispatch, observer, and solver memory costs are changing.
- [x] Define explicit fallback/rejection behavior when an update changes model,
  geometry, `Nx`, stimulus timing, or another static execution contract. Never
  silently expand through a slower legacy route.
- [x] Remove the current `protocol.sweep.build_amplitude_pool` and
  `protocol.sweep.refresh_amplitude_pool` scaling costs. At 4096 axons they
  currently consume about `28 s` per 21-value sweep.

The compact plan now reaches the canonical dispatcher and JAX runtime. Dispatch
items repeat only lightweight references and virtual result indices; source
`AxonInstance`, stimulation, membrane, cable, and footprint descriptions remain
shared. The factorized runtime consumes numerical waveform tables and one
solver call per chunk for both cable formulations. A selected drive may vary in
multi-drive rows while the other source waveforms remain fixed. Opaque
callbacks and static-contract changes are rejected instead of silently taking
a slower route.

The execution contract is now `NumericAxisInput`, owned by dispatcher planning;
`ExtracellularWaveformAxisInput` is its first typed dynamic-input family.
`AxonSimulation._run_numeric_axis`, dispatch expansion, and JAX lowering no
longer depend on recruitment types. Both `recruitment_sweep` and generic
`pool_sweep` use this same path. The replaced mutable `Stimulus` runtime handle,
its global shape revision, and the protocol-only `apply_pool` route were deleted.

Small real CPU cohorts preserve exact activation outputs across chunk sizes and
show the expected fixed-cost amortization. For three amplitudes, mixed `N=2`
warm chunk time fell from `136.05 ms` at size 1 to `32.16 ms` full; single
`N=4` fell from `42.37 ms` to `25.10 ms`, and double `N=4` from `41.41 ms` to
`27.99 ms`. Artifacts are `benchmark/results/p14c_numeric_axis_debug`,
`benchmark/results/p14c_numeric_axis_single4_debug`, and
`benchmark/results/p14c_numeric_axis_double4_debug`.

The realistic single-cable CPU validation under
`benchmark/results/p14c_numeric_axis_realistic_single196_cpu_20260715`
preserves the same 21-point curve for chunk sizes `1/2/full`, ending at
`89/196`. The laptop was thermally unstable: the useful warm comparison is
chunk 2 `simulation.run_pool=105.76 s` versus full `96.32 s`; the anomalous
chunk-1 warm run took `475.39 s`. Re-run the same workload on Kaggle P100 before
making a GPU speedup claim. The priority remaining P14C work is extracting the
numeric axis as a protocol-independent planning/runtime contract, validating
broader stimulus families and consumers, then bounded/padded final chunks.
Define the automatic chunk/memory policy only after P15 observer costs and this
execution path have stabilized.

Basic-08 typed-waveform evidence is recorded under
`benchmark/results/basic_08_full_local_20260715_{callback,typed}_ab`. At 2000
fibers and eight amplitudes, both routes return activation counts
`43 141 403 652 843 983 1245 1319`. On the same noisy local sequence, the
callback sweep took `31.02 s` and the typed sweep `11.26 s`. The attributable
host work is clearer than the thermally variable solver wall time: callback
`protocol.sweep.value` self time was `15.07 s`; the typed waveform updates cost
`4.12 ms` total, and dispatch planning fell from eight calls/`753.9 ms` to one
call/`159.1 ms`. That sequential checkpoint predates the numeric waveform-axis
lowering above and remains the before-baseline for the next P100 run.

The matching public basic-08 P100 checkpoint is retained under
`benchmark/results/kaggle/20260715_225850_basic_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-basic-08-p14-gpu-timing-8e113bf`.
For 100 single-cable plus 100 double-cable fibers and eight amplitudes, the
fresh cold example took `12.50 s`; the second complete example took `719.8 ms`.
Inside the second sweep, its first newly constructed-population amplitude took
`110.9 ms` and the following amplitudes averaged about `38.3 ms`, including
protocol work (`simulation.run_pool` averaged `31.73 ms`). The curve was stable
across both passes at `6 18 41 65 82 101 126 135`. Relative to the pre-P14 P100
artifact from 2026-07-14, the warm sweep fell from `1.802 s` to `392.5 ms` and
typical hot amplitudes from roughly `162-187 ms` to `38.3 ms`. The older curve
used unquantized continuous unmyelinated diameters and is therefore not an
exact same-population numerical A/B.

The exact matching local CPU artifact is
`benchmark/results/basic_08_cpu_local_p14_compare_20260715`. CPU and P100 both
returned `6 18 41 65 82 101 126 135`. Post-first-amplitude
`simulation.run_pool` averaged `107.74 ms` on the local CPU versus `31.73 ms`
on P100 (`3.40x` GPU speedup); the non-overlapping enqueue-plus-wait kernel
pipeline averaged `102.62 ms` versus `23.66 ms` (`4.34x`). The fresh cold
example favored CPU (`5.29 s` versus `12.50 s`) because the GPU paid its JIT;
the complete already-compiled example favored P100 (`0.720 s` versus `1.627 s`).

Do not use device-memory tracing for GPU timing claims. The matching diagnostic
artifact ending in `axonscope-basic-08-p14-gpu-8e113bf` enabled per-span device
memory sampling and inflated the warm example from `0.72 s` to `30.84 s`.
Its roughly `12.5 GiB` `nvidia-smi` footprint mostly reflects JAX's default GPU
preallocation rather than live basic-08 arrays; keep memory diagnostics and
timing runs separate.

The same typed path was validated on the full public
`with_nrv/01_synthetic_fascicle_geometry.py` workflow under
`benchmark/results/with_nrv_01_realistic_cpu_20260715_typed_waveform_r1`:
193 NRV-derived axons, sampled LIFE/FEM footprints, 3 ms at 1 us, and all 21
amplitudes from 0 to 300 uA. The recruitment curve progressed from `0/193` to
`120/193`; `dispatch.build_plan` ran once (`86.87 ms`), all waveform updates
cost `8.05 ms`, and post-cold `runtime.prepare` stayed around `0.4-0.7 ms` per
amplitude. The CPU sweep took `241.81 s`, dominated by `kernel.wait`
(`193.02 s`, about 80% of warm `simulation.run_pool`); NRV population and FEM
footprint construction remained outside that sweep at `3.03 s` and `35.70 s`.

#### P14D - Reusable preparation and signatures

- [x] Stop recomputing a deep O(`Naxon`) runtime digest before a cache hit when
  an unchanged source cohort already has a trusted structural signature.
  Introduce a versioned typed signature owned by the prepared cohort/plan and
  invalidate it on relevant structural mutation.
  - `DispatchGroupStructure` is now constructed once by the canonical dispatch
    plan. Numeric-axis repetition and backend-only last-row padding derive it
    compositionally; runtime and prepared-cohort cache lookup consume it
    directly. The old runtime-owned weak identity signature caches and their
    duplicate deep builders were removed.
  - Equivalent reconstructed signatures use compact pickle payloads plus
    BLAKE2 instead of allocating about `63.3 MB` of `repr` text for the 1024-row
    mixed workload. Local source-plan construction fell from `1.15 s` to
    `0.244 s`; expanded runtime+spatial lookup is about `2.8 us` for both
    groups.
  - The strict unprofiled P100 A/B artifacts end in
    `axs-p14d-signature-baseline-1024-p100-9725f34` and
    `axs-p14d-trusted-signatures-1024-p100-95b1327`. Both return 758 activated
    axons. Warm `runtime.prepare` falls from `914.1` to `67.8 ms` (`13.5x`),
    `simulation.run_pool` from `2.190` to `1.422 s` (`1.54x`), and the sweep
    from `2.473` to `1.773 s` (`1.40x`). Enqueue/dispatch/wait remain stable.
- [x] Compute each source row's solver and temporal signatures once per
  dispatch-plan build, then reuse that prepared description for both the
  content cache key and `DispatchItem` normalization. This replaces the
  duplicate production work without adding another cache or dispatch path.
  On the N=1024 mixed/two-drive/five-amplitude P100 R3 comparison,
  median warm `dispatch.build_plan` falls from `397.3` to `216.4 ms`
  (`-45.5%`). Median `simulation.run_pool` remains stable at `4.12/4.11 s`,
  and every repeat returns `0 565 637 707 758`. The retained optimized artifact
  ends in `axs-p14d-build-plan-reuse-r3-1024-p100-fae6750`; the matching pinned
  baseline ends in `axs-p14d-build-plan-r3-pin-1024-p100-0c060eb`.
- [ ] Preserve safe content-based reuse for separately reconstructed but
  equivalent populations without hashing large `repr(...)` values repeatedly.
  Benchmark cache miss, identity hit, structural hit, and invalidation.
  Equivalent reconstruction is covered and no longer uses the large repr path;
  retain this item for the explicit four-mode cache/invalidation benchmark. A
  P100 prototype that content-hashed complete footprint arrays was rejected:
  it replaced about `258 ms` of footprint work with `379 ms` of hashing and
  raised warm `inputs.extracellular` from `438` to `531 ms`.
- [ ] Reuse dispatch groups, prepared spatial rows, observer probe plans,
  membrane/cable runtime rows, and device arrays across amplitude values.
  Current first-amplitude `runtime.prepare` is about `7.24 s` for single 4096
  despite the batch runtime itself being found in cache.
- [ ] Keep one prepared factorized extracellular plan per source cohort and
  vary only temporal scales/indices. Eliminate repeated row scans and
  footprint-key reconstruction caused solely by replacing stimulus objects.
- [x] Sample factorized spatial footprints once per numeric-axis source row,
  then expand the numerical array according to the trusted amplitude-major
  dispatch shape and final-row padding. The N=1024 mixed/two-drive P100 run
  ending in `axs-p14d-compact-spatial-1024-p100-5f1e37c` samples `512` source
  rows for each `2560`-row cable group and preserves activation counts
  `0 565 637 707 758`. Versus the deferred-current run, warm footprint compute
  falls from `291.4` to `53.5 ms`, footprint-key construction from `37.8` to
  `6.5 ms`, `inputs.extracellular` from `380.1` to `99.1 ms`, and
  `simulation.run_pool` from `4.417` to `4.099 s`. This removes repeated work
  inside one run; persistent source-plan reuse across calls remains open above.
- [x] When a typed numeric axis will replace extracellular currents, defer the
  base current materialization and sample only the final axis waveforms. The
  retained N=1024 mixed/two-drive P100 run ending in
  `axs-p14d-deferred-current-1024-p100-31bd6d6` returns the same
  `0 565 637 707 758` activations. The discarded
  `current_scaled_shared_waveform` work was `132.7 ms` in the baseline;
  `inputs.extracellular` falls from `438.1` to `380.1 ms` and
  `simulation.run_pool` from `4.440` to `4.417 s`, while dispatch/wait remain
  stable. The larger sweep span in this single run comes from unrelated
  `dispatch.build_plan` variance and is not treated as a regression claim.
- [ ] Consume indexed rank-S current patterns without a device-side
  `current_mid_A[B, S, Nt]` gather only if this can preserve the canonical
  solver executable and cold compilation cost. The first prototype fused the
  gather into the double-cable JIT: warm 1024-axon five-amplitude run-pool
  improved slightly, but cold double-cable `kernel.dispatch_jax` rose from
  `7.02 s` to `10.09 s`, so that form was rejected. Keep the compact host
  payload and external gather until a scan-level indexing/fusion design has a
  better cold/warm tradeoff.
- [ ] Retain the compact factorized Vext representation. For single 4096 its
  current device payload is about `6.6 MB` versus a `9.83 GB` dense equivalent;
  optimize host discovery rather than rematerializing dense Vstim.
- [ ] Keep preparation caches bounded, diagnosable, explicitly clearable, and
  independent of Python object-id reuse after garbage collection.

The indexed multi-drive P100 A/B uses the baseline artifact ending in
`axs-p14d-indexed-baseline-1024-p100-95b1327` and retained artifact ending in
`axs-p14d-compact-host-1024-p100-73aadd8`. Both return activation counts
`0 565 637 707 758` for 1024 mixed axons, five amplitudes, and two drives.
Per cable group, temporal host payload falls from `61.44 MB` of repeated
currents to `120 KB` of five unique patterns plus about `10 KB` of indices.
Warm `simulation.run_pool` falls from `4.545 s` to `4.440 s` (`1.02x`), while
`kernel.dispatch_jax` remains effectively stable (`3.531 s` to `3.512 s`).
The dedicated `inputs.numeric_axis` spans are `46.0/54.6 ms` for
single/double cable. This is primarily a preparation/memory convergence, not a
solver speedup.

#### P14E - Acceptance gate

- [x] Re-run local CPU and Kaggle GPU at `Naxon={196,1024,4096}` for both cable
  formulations. Report population construction, amplitude-plan construction,
  first and later amplitude times, `run_pool`, preparation, dispatch, wait,
  host/device memory, and numerical outputs.
  - The corrected five-amplitude P100 matrix is recorded in the six artifacts
    ending in `axs-p14e-source-reuse-{single,double}-{196,1024,4096}-p100-f46bbec`.
    One source population is built per policy and reused unchanged by cold and
    warm phases. Warm `simulation.run_pool` is `0.988/4.774/18.706 s` for
    single cable and `0.649/2.666/9.971 s` for double cable at
    `Naxon=196/1024/4096`; the non-overlapping `kernel.enqueue + kernel.wait`
    share is `95.8-97.4%`. Full native amplitude batching intentionally has no
    separate Python call or timing for each amplitude.
  - Reusable P100 source construction is `0.303/1.221/5.648 s` single and
    `0.376/1.708/6.912 s` double. Cold phase-only run-pool is
    `3.784/8.786/26.832 s` single and `8.225/10.731/18.188 s` double;
    `one_shot_wall_ms` separately preserves source-plus-cold user cost.
  - Separate N=4096 diagnostics report actual peak JAX device bytes of
    `5.84 GB` single and `2.63 GB` double, plus peak host RSS of
    `1.87/1.93 GiB`. The `12475 MiB` shown by `nvidia-smi` is JAX allocator
    preallocation, not live workload state. `memory-trace=all` is rejected at
    this scale because Python tracemalloc makes the run impractically slow;
    device and RSS traces are collected independently.
  - CPU double cable uses Thomas, P100 double cable uses the guarded
    `jax_triton_loop_xb` route, and both cable formulations retain factorized
    extracellular inputs. The corrected local single-N=1024 boundary run
    returns `0 85 197 325 433`, versus `0 85 197 326 433` on P100. This one
    near-threshold fiber at 225 uA is within the documented cross-backend
    tolerance; all same-backend cold/warm outputs match exactly.
- [x] Require exact source-pool immutability and matching activation curves.
  Cross-backend near-threshold comparisons may use the already documented
  one-amplitude-step tolerance, but same-backend A/B results must match.
- [x] Exit P14 only when Python population/preparation work no longer scales as
  `Namplitude x Naxon` and the first-amplitude structural preparation is no
  longer a material fraction of a warm sweep. The native numeric axis now
  carries amplitudes without cloned Python rows. At N=4096, warm
  `runtime.prepare` is `171 ms` single and `29 ms` double, while reusable
  build-plan work is `52/226 ms`; together they remain below 3% of warm
  `simulation.run_pool` in both formulations.

### P15 - Compact Activation And Spike Observers (complete)

Primary objective: use bounded solver-side event state when a protocol needs
events rather than a temporal voltage raster. Keep VmRaster as the strict
raster route and add explicit typed fast paths; do not restore a broad generic
solver-side observer fallback.

#### P15A - Typed compact event contracts (complete)

- [x] Define three explicit solver-output families without a generic observer
  fallback:
  - temporally downsampled VmRaster for visualization or retained threshold
    state, with a typed step stride and documented window semantics;
  - spatial VmRaster probes using the existing `PositionSelector` and
    row-aware probe tables, optionally combined with temporal downsampling;
  - bounded spike events storing timestep indices rather than a temporal
    raster. For population sweeps, use
    `[Namplitude, Naxon, Nprobe, K]` `int32` event indices plus per-probe
    `count` and `overflow`; `Nprobe=Nx` is the explicit all-compartment case.
  Keep full VmRaster as the lossless reference route. Downsampled raster output
  must not silently back exact activation/spike analyses when crossings may be
  missed; use event-preserving window reduction or a compact event plan for
  those analyses.
- [x] Define canonical runtime output plans and result contracts for compact
  activation, first crossing, spike count, and bounded spike times. Specify
  shapes, dtypes, units, blanking, threshold crossing,
  hysteresis/refractory behavior, invalid states, and CPU/GPU semantics.
- [x] Implement minimal activation as one boolean per amplitude/axon, updated
  during the temporal scan and returned as `[Namplitude, Naxon]` without
  allocating or decoding VmRaster. Apply `Activation.blanking` in the scan so
  threshold hits are accepted only for samples at or after `tmin`.
- [x] Implement first-crossing/latency state as `int32` timestep sentinels per
  selected probe group, converting to physical time only during finalization.
- [x] Implement constant-memory first/last spike time and spike count. Detect
  rising threshold crossings with blanking plus hysteresis/refractory semantics
  so one broad spike is not counted at every timestep. Store integer timesteps
  during the scan and convert to physical timestamps only during result
  finalization.
- [x] Add a bounded `K`-event representation with explicit overflow only for
  workflows that need individual timestamps.
- [x] Support existing `PositionSelector` semantics and selectors containing
  several positions, reducing each probe group online.
- [x] Extend `estimate()`/`inspect()` with output-state and transfer estimates
  for full raster, downsampled raster, probes, and bounded events. Make the
  `Nprobe x K` cost visible before execution and require an explicit policy for
  expensive all-compartment event capture. At 21 amplitudes, 4096 axons,
  200 compartments, and `K=4`, event indices alone are about `263 MiB`; sparse
  probes remain the preferred route when complete spatial timestamps are not
  scientifically required.

#### P15B - Runtime integration and validation (complete)

- [x] Integrate compact plans through `axonscope.runtime.execution`, output
  contracts, dispatcher caching, result assembly, and protocol results. Keep
  identical public semantics for single/double cable and crash on an
  unintended VmRaster fallback.
- [x] Switch `recruitment_sweep` to compact activation only after exact
  equivalence. Finalize all amplitudes together and avoid creating one public
  `AxonSimulationResult` per amplitude.
- [x] Test no spike, one/repeated spikes, blanking/chunk boundaries, overflow,
  and multi-position probes.
- [x] Compare compact plans with retained VmRaster references on local CPU and
  Kaggle GPU at `Naxon={196,1024,4096}`. Record cold/warm timing, peak memory,
  transfers, output size, and exact activation/recruitment equivalence.
- [x] Require the 4096-axon full recruitment workload to run without a
  multi-GiB observer state. The target activation state is roughly `86 KB` for
  21 x 4096 booleans rather than `6.02 GiB` of single-cable VmRaster.
- [x] Add a didactic compact-activation example showing solver-side blanking,
  bounded retained shape, and exact equivalence with post-hoc recorded Vm.
- [x] Add a didactic compact-latency example showing first-crossing retention,
  bounded retained shape, and exact equivalence with post-hoc recorded Vm.
- [x] Add a didactic compact-spike example showing dense Vm, its threshold
  raster, and exact count/first/last equivalence with constant-memory output.

### P16 - JAX Temporal Solver And Dispatch Optimization (complete)

Primary objective: optimize the actual temporal program only after P14/P15
remove host-pool expansion and full-raster contamination. Treat
`kernel.dispatch_jax + kernel.wait` as one solver interval; moving work between
the spans is not a gain.

- [x] Establish a fresh clean baseline from the P14 realistic workload using
  compact activation. Cover both cable formulations, `Naxon={196,1024,4096}`,
  first-call cold execution, subsequent hot amplitudes, representative
  amplitude chunk sizes, throughput, memory, executable identity, and outputs.
- [x] Decompose the jitted program with optimized HLO and device traces.
  Measure membrane terms, factorized stimulation, system assembly,
  tridiagonal/Triton solve, compact observer update, copies/materializations,
  and kernel/custom-call count.
  - Optimized-HLO and warm Perfetto capture now cover both production routes.
    At N=1024, single cable is about 51% cuSPARSE and otherwise mostly gate,
    membrane, and system assembly. Double cable is only about 19% Triton and is
    dominated by gate reconstruction, batch/node layout changes, assembly, and
    copies. Artifacts end in
    `axs-p16-hlo-{single,double}-4096-p100-7648bbc` and
    `axs-p16-warmtrace-{single,double}-1024-p100-7648bbc`. A separate double
    N=4096 device-memory run validates the production Triton kernel to
    `8.82e-7` max absolute error and reports about `781 MiB` peak JAX bytes in
    use; the roughly `12.5 GiB` NVIDIA figure is JAX's allocator reservation.
    Perfetto plus HLO answered the kernel-boundary questions, so a redundant
    Nsight pass was not retained as a closure requirement.
- [x] Re-evaluate time chunking after compact observers. The current realistic
  VmRaster run launches six dependent JAX calls of 512 steps per cable group and
  amplitude. A local 58-axon CPU double-cable A/B retained under
  `benchmark/results/p14_enqueue_cpu_double_58_chunk{512,1024}_r3_20260715`
  reduced mean warm `simulation.run_pool` from `2.109 s` to `1.991 s` per
  amplitude at 1024 steps (about 5.6%), but mostly moved asynchronous work from
  `kernel.enqueue` into `kernel.wait`. Do not change the global default from
  this narrow CPU case; repeat across single/double cable, population sizes,
  GPU, memory, and the future compact-state route. The final compact matrix is
  recorded under `p16_time_chunk_compact_196_cpu_local_20260718_v3` and P100
  artifacts ending in `axs-p16-timechunk-realistic-1024-p100-e0bde84` and
  `axs-p16-timechunk-single-4096-clean-p100-e0bde84`. At P100 N=4096 single
  cable, the current 512 default is fastest (`3.393 s`) versus unchunked
  (`3.569 s`), 128 (`3.625 s`), and 1024 (`3.635 s`); the N=1024 unchunked
  gain therefore does not generalize, and no adaptive specialization is
  retained.
- [x] Hoist HLO-confirmed run-invariant work: prepared cable terms,
  area/background arrays, `cx_plus_gx`, `cx_over_dt`, current rows, scan
  layouts, and static observer tables. Keep the representation internal,
  typed, reusable, and membrane-model agnostic. Existing preparation caches
  retain the host/runtime invariants; node-first state and the fused physical
  assembly call remove the measured double-cable device materializations.
  The remaining single-cable assembly is time-step dependent.
- [x] Optimize the existing single-cable program before adding a new route.
  Compare `vmap(scan(step))` with `scan(vmap(step))`, inspect
  `jax.lax.linalg.tridiagonal_solve`, and remove measured materializations or
  kernel boundaries. A production `scan(vmap(step))` A/B was numerically exact
  but regressed median warm runtime by `3.1%` at N=1024 and `2.9%` at N=4096,
  so commit `7483737` restores `vmap(scan(step))`. HLO/Perfetto show the
  retained route is already dominated by cuSPARSE; no second scalar solver was
  added without the required end-to-end evidence.
- [x] Optimize double cable in stages: first fuse system assembly with the
  tiled-Thomas custom call so it consumes compact physical/runtime inputs.
  Temporal blocking crosses membrane/observer step boundaries and therefore
  moves to P17's generated-contract work instead of becoming another
  hand-written P16 specialization.
  - [x] Keep the compact double-cable scan state node-first when the membrane
    backend advertises model-agnostic node-first batch support. On P100 at
    N=1024, median warm `simulation.run_pool` falls from about `1.870 s` to
    `1.508 s` (`-19.4%`) with exact activation counts. At N=4096 it falls from
    `6.703 s` to `5.077 s` (`-24.3%`), while cold run-pool time falls from
    `18.101 s` to `14.678 s` (`-18.9%`). Warm Perfetto device
    time falls from about `1.76 s` to `1.39 s` (`-21%`): the five measured
    per-step layout kernels costing about `639 ms` disappear, while the
    remaining assembly fusion grows from `252` to `430 ms`. Artifacts end in
    `axs-p16-node-first-double-1024-p100-ac542e4`,
    `axs-p16-nftrace-d1024-p100-ac542e4`, and
    `axs-p16-nf-d4096-p100-ac542e4`.
  - [x] Replace the production coefficient-materialization boundary with one
    physical-term Triton custom call that assembles each 2x2 block in registers
    before Thomas elimination. Combined with node-first state, median warm
    `simulation.run_pool` improves from the P16 baseline by `26.3%` at N=1024
    (`1.870 -> 1.378 s`) and `26.2%` at N=4096 (`6.703 -> 4.945 s`). The
    incremental solver-interval gain over node-first alone is about `8.9%` and
    `6.8%`; first-miss Triton compilation grows from `3.70` to `4.98 s`, so
    cold remains a cache target. Independent P100 dense-solve validation passes
    with max absolute/scaled errors `8.82e-7/1.96e-7`. Artifacts end in
    `axs-p16-fusedasm-d{1024,4096}-p100-ff80df8`,
    `axs-p16-fusedtrace-d1024-p100-ff80df8`, and
    `axs-p16-fusedvalidate-p100-ff80df8`.
  - Keep one coherent state layout across membrane evaluation and the
    node-first Triton solve. A direct `dynamic_update_slice` gate-carry
    experiment was rejected: it changed optimized gate layout to
    `{gate,batch,node}`, added a full per-step transpose, and regressed N=1024
    median warm runtime from about `1.87 s` to `2.62 s` despite exact outputs.
- [x] Revisit Triton input/output aliases only with XLA buffer-assignment
  evidence. The retained warm medians are `437.5/82.7/28.7 ms` for
  `run_pool/dispatch/wait` without aliases and `565.5/113.5/9.75 ms` with
  aliases. The lower `kernel.wait` did not offset higher dispatch/run-pool
  time, so the alias candidate was rejected.
- [x] Keep P16 membrane work backend-contract based. Profiles may justify a
  future fused membrane kernel, but generating model-specific JAX/Triton
  operations from the compiled membrane contract belongs to P17; never
  hard-code MRG or another built-in model into a solver.
  - The retained fused-assembly profile leaves about `337 ms` in generated JAX
    membrane/gate reconstruction at N=1024. Moving that work into Triton is a
    P17 generated-contract experiment, not another hand-written P16 solver
    specialization.
- [x] Precompile and persist reusable solver executable families instead of
  merely caching each complete simulation after first use. Define a structural
  compilation signature from runtime/backend and device capability, precision,
  cable formulation, generated membrane contract, observer contract, and
  bucketed batch/Nx/time-chunk shapes. Keep amplitudes, footprints, waveform
  values, membrane parameter rows, and other same-shape simulation data as
  dynamic operands so changing them does not create another executable.
  - Use JAX's persistent compilation cache for non-Triton executable replay
    under `.axonscope_cache/runtime/jax/xla`, together with the existing Triton
    TTIR-to-PTX cache. Determine whether executable families can be lowered and
    compiled eagerly when a runnable plan is finalized, before the first
    measured simulation amplitude. Follow JAX's official persistent-cache
    contract:
    <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>.
  - Benchmark true cross-process miss/hit and same-signature reuse while
    changing amplitudes, footprints, waveform values, and parameter rows.
    Report trace/lower/compile/first execution separately, executable identity,
    the exact reason for every specialization/cache miss, cache size/LRU,
    clean/disable behavior, trusted sharing, and shape-bucket cardinality.
    Explicitly test `jax_persistent_cache_min_compile_time_secs`,
    `jax_persistent_cache_min_entry_size_bytes`, the relevant
    `jax_persistent_cache_enable_xla_caches` GPU modes, and
    `jax_explain_cache_misses`. Treat reuse as exact cache-key reuse over
    non-optimized HLO, jaxlib/XLA flags, device topology, and compression; do
    not describe structurally similar but distinct HLO programs as cache hits.
  - The retained process-wide policy uses JAX's native cache, bounded by a
    configurable maximum size, with explicit disable, minimum compile-time,
    minimum entry-size, and GPU XLA-cache controls. On P100 double cable,
    exact replay and changed-amplitude/value replay produce identical
    StableHLO, add zero XLA files, and reduce the captured cold phase from
    `6.034 s` to `0.518/0.520 s` (`11.6x`). Triton lowering changes from
    `3.906 s` to `0.122 s`; the isolated caches occupy about `720 KiB` XLA and
    `24 KiB` Triton. Local CPU single cable likewise reuses the same StableHLO
    with changed values and reduces captured cold work from `1.007 s` to
    `0.641 s` (`1.57x`). Artifacts end in
    `axs-p16-dynamic-cache-double-p100-f43d7d1` and
    `p16_compilation_cache_single_196_cpu_local_20260718_dynamic`.
    Eager compilation before a plan has concrete structural shapes was not
    retained: it only moves first-use work and cannot produce JAX's exact cache
    key. The persistent family is populated once a concrete runnable signature
    exists, then reused across processes and dynamic simulation values.
- [x] Keep synchronous scheduling as the P16 default. Prior homogeneous
  256/1024 runs showed no async benefit, and P16 does not have a representative
  forced-heterogeneous workload. The 2/4/8-group device-idle, memory, ordering,
  and end-to-end matrix remains tracked under future HPC integration rather
  than blocking temporal-solver closure.
- [x] Validate every promoted optimization against CPU references and the prior
  GPU route, including stateful/stateless membranes, both cable formulations,
  factorized stimulation, compact observers, and retained VmRaster. Focused
  runtime/protocol/dispatcher and benchmark suites pass locally, and guarded
  P100 dense-solve plus activation-equivalence checks pass for the retained
  double and single routes.
- [x] Require a repeatable 10-15% end-to-end or solver-interval gain before
  retaining materially more complex kernel code. Node-first plus fused double
  assembly clears the threshold at N=1024/4096; scan-order, aliases, async, and
  additional double chunk specialization do not and were rejected or deferred.

### P17 - Autonomous Generated Membrane Runtime Contracts

Primary objective: after compilation, runtime-specific generated modules are
the only source of model-specific runtime facts. Model IR remains a compiler,
validation, composition, inspection, and reference artifact rather than a
runtime reconstruction path.

- [x] Generate runtime artifacts lazily per model and target with
  content-addressed cache directories; JAX currently requests `jax_model.py`.
- [x] Move stateless gate rates, Q10 factors, conductances, and reversal terms
  into generated model-agnostic JAX functions.
- [x] Introduce a versioned, typed JAX runtime contract in `jax_model.py` for
  canonical parameter metadata, names/groups, states, gate policies,
  diagnostics, step-hook declarations, and structural identity. Keep generated
  cache contents canonical across per-instance parameter overrides.
- [x] Generate parameter defaults; typed input/state/current/observable/
  diagnostic metadata; gate update policy; auxiliary-state initialization;
  stateful prepare/finalize and current terms; diagnostics; and callable
  signatures into the v2 `jax_model.py` contract. Single-source generated
  routes fail on incomplete entrypoints instead of silently interpreting the
  missing operation, and generated signatures avoid unrequested current-matrix
  construction in state hooks.
- [x] Audit recording and compact result metadata against the generated
  contract. All currently public solver-recorded groups (gates, aggregated
  currents/conductances, membrane states, and diagnostics) now load from
  `jax_model.py`; generated loading validates display names, raw-column
  partitions, observables, state/diagnostic outputs, and callable signatures.
  Generic model observables remain compiler/inspection outputs until a public
  typed `Recording` policy is deliberately introduced, rather than being
  exposed implicitly as a P17 side effect.
- [x] Generate one content-addressed JAX/NumPy artifact for stateless composite
  models. Cache hits derive identity from component source keys and public
  labels, preserve parameter overrides as runtime values, and load without
  deserializing/recomposing Model IR or entering `JaxModelIRLowering`. The local
  P17 probe measures `46.47 ms` for first generation and `3.71 ms` for a warm
  Rattay-Aberham plus Passive composite build. Direct composite loading is
  `1.96x` faster than the earlier `7.27 ms` graph-recomposition cache hit; the
  remainder is small generated-contract and JAX-program construction overhead.
- [x] Load single-source JAX plus NumPy host-support artifacts directly after a
  cache hit without parsing the source, deserializing `optimized_graph.json`,
  rebuilding `MembraneProgram`, or evaluating Model IR expressions. Preserve
  per-instance parameters in the runtime static signature. A local 25-repeat
  check reduced HH cache loading from `2.04 ms` to `0.79 ms` and program build
  from `2.62 ms` to `1.09 ms`; Schild97 moved from `30.06 ms` to `1.04 ms` and
  `27.23 ms` to `4.65 ms`, respectively. The reproducible probe lives in
  `benchmark/membrane_runtime_cache.py`.
- [x] Keep invariant compartment layout values outside the evolving GPU scan
  state for capability-stacked gated/leak membranes. The canonical backend now
  carries only generated dynamic gates through the node-first double-cable
  scan and restores the full internal gate array once per chunk. On the exact
  P100 Naxon=1024, five-amplitude, 3000-step profile, this removes the
  per-step `f32[74,5120,7]` gate reconstruction: device time falls from
  `1267.3` to `945.1 ms` (`-25.4%`) and profiled `simulation.run_pool` from
  `1740.2` to `1431.5 ms` (`-17.7%`), with unchanged `0 1024 1024 1024 1024`
  activation counts. The Thomas kernel itself remains `404.4 ms`; artifacts
  end in `axs-p17-membrane-trace-d1024-p100-d8ccff1` and
  `axs-p17-static-gates-d1024-p100-cf8adcd`. Repeated unprofiled P100 runs
  confirm median warm `simulation.run_pool` improvements against the final P16
  reference from `1.378` to `1.111 s` at Naxon=1024 (`-19.4%`) and from
  `4.945` to `3.599 s` at Naxon=4096 (`-27.2%`). Median complete recruitment
  sweeps are `1.218` and `4.119 s`, respectively; artifacts end in
  `axs-p17-sg-time-d{1024,4096}-5623f0a`.
- [ ] Apply and benchmark the same dynamic-gate/invariant-layout separation on
  the canonical single-cable GPU observer path. Replace the existing carry in
  the shared-rank1, factorized, and zero-stimulation scans through one common
  capability-based mechanism; do not add parallel scan variants. Profile
  Naxon={1024,4096} on P100, report cuSPARSE versus membrane/layout/observer
  time, cold and warm end-to-end timing, memory, and exact activation/result
  equivalence. Retain it only if the complete single-cable workload improves.
- [ ] Emit equivalent target-specific metadata/functions for future runtimes
  rather than making them consume JAX artifacts.
- [ ] If P16 leaves material membrane/gate cost after layout and generic
  assembly optimization, let the generated contract optionally emit
  model-agnostic Triton membrane operations that can be fused with temporal
  execution. Keep equations and parameters generated from the membrane source,
  not hand-written in a cable solver.
- [ ] Once generated membrane and observer operations can share a temporal
  program, benchmark temporal blocking `K={2,4,8,16}` by total runtime,
  registers, memory, compile cost, and numerical equivalence. Do not add a
  solver-specific blocking path before that contract exists.
- [x] Validate built-ins, stateful models, composition, parameter overrides,
  recording labels/groups, diagnostics, numerical equivalence against the
  Model IR oracle, cache invalidation, generated-code inspection, and cold/warm
  performance. The final focused/full local passes complete with `115 passed`
  and `724 passed, 1 skipped`; P17 cache evidence is recorded under
  `benchmark/results/p17_composite_generated_runtime_local_20260718/`.
- [x] Remove the JAX Model IR production fallback. Cache misses and hits for
  single-source stateless/stateful membranes and stateless composites now
  construct `JaxMembraneProgram` only through autonomous generated modules;
  missing JAX/NumPy targets fail explicitly and production programs retain no
  Model IR graph. `JaxModelIRLowering` remains an internal numerical oracle for
  compiler tests, not an `AxonSimulation` execution route.

### P18 - Membrane Model Completion And Validation

- [ ] Implement Nav1.x-family and other Markov-based membrane models through
  the canonical membrane-source and generated-runtime contracts.
- [ ] Re-check every built-in membrane model against its NRV implementation;
  explicitly audit formulas, defaults, states, temperature behavior, and
  recording semantics that may have been lost during translation.
- [ ] Finish missing Gaines and Markov model families.
- [ ] Add focused numerical references and runnable advanced examples for each
  retained public model.

### P19 - Pre-V1 Cleanup And Public Surface

- [ ] Reorganize `src/`, especially Python modules still at package root, only
  after P14-P18 settle their ownership boundaries.
- [ ] Inventory every Python module, function, type, and public export.
- [ ] Use Graphify, `vulture`, source call sites, examples, and tests to remove
  code used only by tests, legacy paths, replaced slow routes, and duplicated
  implementations.
- [ ] Verify retained code ownership, contracts, runtime boundaries, and lack
  of duplicate public concepts.
- [ ] Remove public API not documented by basic or advanced runnable examples.
- [ ] Run a final Graphify-guided whole-package cleanup after CPU/GPU
  optimization, with `runtime/jax` treated as the already completed first
  slice.
- [ ] Clean `pyproject.toml`, including explicit optional CUDA/Triton GPU
  dependencies and removal of obsolete extras.
- [ ] Revisit artifact caching globally: build only requested runtimes, define
  first-call versus install-time built-ins, document clean/disable/retention,
  and keep `.axonscope_cache` deterministic and inspectable. Evaluate shape
  buckets for final partial time chunks only with observer masking that makes
  padded steps numerically inert.

### P3 - Documentation And Examples

- [x] Rewrite README after post-P7 stabilization.
- [x] Audit public examples after benchmark flattening.
- [ ] Write the indexed notebook mini-course under `examples/tutorials/`.
- [ ] Prepare proper Sphinx documentation.
- [ ] Complete/update all public docstrings.

### P13 Remainder - Dense Vm Recording

- [ ] Benchmark dense/full Vm recording separately from VmRaster. It may need
  a distinct chunk policy, but it must not complicate the established
  `DEFAULT_OBSERVER_TIME_CHUNK_STEPS = 512` observer default without evidence.

## Future Product Phases

### Propagation And Conduction-Block Validation

The former `ConductionBlock` definition was removed in P15 because distal
non-activation alone cannot establish a block. Do not restore it as an inverted
activation criterion. Reintroduce a public block result only after a dedicated
scientific validation campaign.

- [ ] Add canonical propagation analysis with one source probe and one or more
  targets. Classify at least `propagated`, `blocked`, `not_initiated`, and
  `ambiguous`.
- [ ] Require target activation after source activation, optionally inside a
  valid delay window, so direct or reverse activation is not mistaken for
  expected propagation.
- [ ] Support source/proximal/distal probes and distinguish bidirectional,
  proximal-only, distal-only, local-only, no-initiation, and target-only/direct
  activation.
- [ ] Define repeated-spike propagation separately. Evaluate counters and
  last-event matching for summaries, with a small bounded FIFO only when
  source-to-target event pairing is required.
- [ ] Run a dedicated CPU/GPU validation campaign for propagation and true
  conduction block before exposing a public analysis or block-threshold study.
- [ ] Add a didactic KES/block example distinguishing local activation, failed
  initiation, propagation, and true conduction block, including the required
  filtering workflow.

### Studies And Persistence

- [ ] Implement callable threshold curves, block-threshold curves, recruitment
  curves, conduction validation, parameter sweeps, reuse/retention policies,
  and canonical study results.
- [ ] Define final schemas, typed serialization, and persistence strategy.

### External Integration

- [ ] Continue NRV hardening only where the package contract is stable. Keep
  geometry in `examples/with_nrv` or benchmarks and avoid duplicating the
  canonical sampled-footprint path in `axonscope.integrations.nrv`.
- [ ] Work on HPC integration, including cache sharing, scheduling, artifact
  retention, and reproducible benchmark execution.
- [ ] Benchmark async JAX scheduling for real forced heterogeneous groups
  (incompatible membrane contracts, cable/Nx shapes, or temporal stimulus
  signatures). Sweep 2/4/8 groups and require device-idle evidence, bounded
  pending memory, deterministic ordering, and an end-to-end gain.
- [ ] Implement the CPU/NRV FEM-footprint path described in
  `ideas/fem_axon_gpu_coupling_design.md` before GPU FEM. Split FEM solve,
  first footprint, cached sampling, and AxonScope solve; cache reusable field
  bases; introduce axon embedding/projection to avoid repeated point location;
  then select full footprints, chunked projection, or future fused paths by
  memory budget.
- [ ] Test Apple Metal acceleration with `jax-mps`:
  https://github.com/tillahoffmann/jax-mps

### Deferred NumPy/SciPy Reference Runtime

This remains a future debugging/reference backend, not a JAX compatibility
wrapper and not the next implementation phase.

- [ ] Keep `axs.runtime.numpy` reserved until it executes through the same
  `AxonSimulation(...).run()`, `.estimate()`, and `.inspect()` lifecycle.
- [ ] Define a tiny deterministic v1 subset: single cable first, intracellular
  current, sampled footprints, recording, observers, and selected membranes.
- [ ] Implement tridiagonal Crank-Nicolson with clear NumPy/SciPy primitives,
  favoring readability and numerical reference value over population speed.
- [ ] Consume the canonical membrane/compiler contract through a NumPy-specific
  generated target or clear reference representation; never call into the JAX
  runtime or preserve a second user-facing membrane-authoring path.
- [ ] Add cross-backend JAX comparisons for Vm, activation, block, latency,
  thresholds, probes, retained membrane values, and model-step equivalence.
- [ ] Add runtime policy, examples, docs, estimates, and inspection records only
  after executable behavior exists.
- [ ] Document when to use this runtime for tiny deterministic debugging and
  numerical regression, and when not to use it for population performance or
  GPU-parity expectations.

## Completed Evidence Summary

- P12 closed runtime/JAX cleanup, warm/cold host preparation, strict route
  guards, dead-code cleanup, factorized Vext, generated JAX membrane terms,
  Triton fusion, and TTIR-to-PTX persistence. Detailed decisions remain in
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md` and the benchmark
  artifacts referenced by `benchmark/README.md`.
- P13 retained observer/VmRaster chunk size `512`; the measured effect across
  single/double cable and `Naxon={1,64,1024}` was small enough that adaptive
  policy was rejected for now.
- P16 retained node-first double-cable state and fused physical-term Triton
  assembly, cutting median warm P100 run-pool time by about `26%` at
  Naxon=1024/4096. It rejected single scan-order inversion, Triton aliases,
  homogeneous async scheduling, and adaptive time chunks; added exact
  cross-process JAX/XLA plus Triton cache replay; and moved generated membrane
  fusion and temporal blocking to P17.
- Basic examples 06/07/08 and `with_nrv/01` were validated on local/Kaggle CPU
  and P100 GPU before the P14 work. Current reference artifacts include
  `benchmark/results/kaggle/20260714_021129_basic_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-basic-06-07-08-post-p12-gpu`
  and
  `benchmark/results/kaggle/20260714_115613_with_nrv_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-with-nrv-01-post-p12-fullpop-gpu`.
- Native amplitude batching became numerically correct, added configurable
  chunk sizes, shared source axons across amplitude clones, reused spatial and
  runtime plans, removed the double-cable dense-current fallback, handled zero
  waveform scaling, and validated exact recruitment curves. Those changes are
  retained as evidence and stepping stones, but P14 replaces the remaining
  expanded-object execution representation.
- GPU async scheduling across already compatible homogeneous groups was tested
  and rejected as a default: 256 axons measured `198.7 ms` sync versus
  `213.8 ms` async, and 1024 measured `606.6` versus `611.5 ms`.
- The supported double-cable Triton cache reduced TTIR-to-PTX lowering from
  `3.962` to `0.063 s` and instrumented cold JIT from `6.104` to `1.551 s` for
  its validated signature, with unchanged recruitment results.

## Key References

- Product and architecture: `GUIDELINES.md`
- Working guide: `AGENTS.md`
- Benchmark surfaces: `benchmark/README.md`
- Validation policy: `docs/validation.md`
- Examples map: `examples/README.md`
- P11 closeout: `docs/architecture/p11_closeout_2026_07_12.md`
- P12 runtime cleanup:
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md`
- Pre-cleanup TODO archive:
  `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`
