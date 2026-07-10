# AxonScope TODO

Living execution plan for AxonScope cleanup, benchmark evidence, validation,
documentation, and the next solver/runtime phases.

`GUIDELINES.md` is the architecture reference. `AGENTS.md` is the agent working
guide. The source tree, tests, runnable examples, and benchmark reports remain
the implementation truth for current behavior.

This file should stay short enough to plan from. Detailed completed ledgers
belong in architecture docs, benchmark reports, changelog entries, or commit
history. Do not remove unfinished work from this file unless it is explicitly
completed, rejected, or moved to a named tracking document.

## Snapshot

Updated on 2026-07-09 during the P11C Triton workflow benchmark gate.

Current state:

- P7 is closed: public membrane authoring is class-based through
  `axs.membranes.Model`; built-ins live as standalone model classes under
  `src/axonscope/membranes/models/`.
- The historical `channel_models`, `icm`, `model_ir/models`, and
  `model_ir/builtins.py` paths are removed from the active package and must
  stay absent.
- Model IR remains internal compiler/runtime vocabulary. Users write membrane
  models, equations, parameters, gates, currents, and observables.
- `axs.runtime.numpy` remains reserved for a future real NumPy/SciPy reference
  runtime, but it is now a bonus/future phase. It must not become a JAX-backed
  compatibility path.
- Solver-side observer-only execution is the strict VmRaster path under
  `observations["vm_raster"]`; activation, latency, velocity, threshold, and
  recruitment summaries are post-processing.
- `PeakVoltage` remains post-hoc on recorded Vm unless a dedicated benchmarked
  solver-side design is accepted.
- P3 is paused after current-docs cleanup. P9 is closed with a short cold-run
  baseline, scalar/batch span coverage normalization, explicit hotpath
  `time_chunk_steps` controls, and documented closeout decisions before any
  larger runtime project.
- The next priority is not the NumPy solver. First flatten the public
  model/compiler surface, then build realistic benchmark evidence and optimize
  the current JAX solver path.

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
- Public examples must not import solver/backend internals.
- Every public feature, option, workflow, analysis, runtime mode, inspection
  view, or advanced concept must be documented in runnable examples or removed
  from the public surface.
- World/anatomical coordinates, trajectories, nerve geometry, electrode CAD,
  surgical placement, and FEM solving stay outside AxonScope core. AxonScope
  consumes intrinsic positions and sampled footprints.

## Active Plan

### P0 - Post-P7 Consistency

- [x] Rewrite `CHANGELOG.md` Unreleased so it reflects the actual post-P7
  state. Remove or rewrite entries that still present deleted/rejected paths as
  current additions, including `ModelIRMembrane`, `CompositeICM`,
  `ExtracellularContext`, `axonscope.icm`, broad solver-side
  `PeakVoltageObserver`, and obsolete example names.
- [x] Decide the public status of `PeakVoltageObserver`:
  - privatize/remove it from `axs` and `axs.analysis`, then update tests and
    docs.
- [x] Clean public docs that still present `MembraneModel` as a user-facing
  concept. Public wording should prefer "membrane model" and
  `axs.membranes.Model`; `MembraneModel` is an internal normalized descriptor.
- [x] Audit `docs/api_public_draft.md`: either mark it more explicitly as
  historical/proposal material or trim/update sections that conflict with the
  current public API.
- [x] Keep proposal docs clearly labeled when they show future APIs.
- [x] Update guardrails after the `PeakVoltageObserver` decision so tests
  encode the accepted public surface.

### P0.5 - Benchmarking And Memory Profiling Observability

Goal: before flattening P1 benchmark commands, make sure AxonScope can explain
where time and memory go at each simulation stage. A useful benchmark report
must map both timings and measured memory pressure across planning, dispatch,
preparation, input lowering, kernel enqueue/wait, and result assembly.

Audit on 2026-07-02:

- [x] Inventory existing benchmark/profiling tools:
  `src/axonscope/benchmarking/hotpaths.py` records nested timing events to
  `events.jsonl`, `summary.csv`, and `metadata.json`; public execution already
  emits spans for `dispatch.build_plan`, `runtime.prepare`,
  `inputs.positions`, `observer.plan`, `inputs.intracellular`,
  `inputs.extracellular`, `kernel.enqueue`, `kernel.wait`,
  `results.split_batch`, and `results.to_public`.
- [x] Inventory current JAX profiling support:
  `benchmark/runtime/benchmark_solver.py` can wrap a run with
  `jax.profiler.start_trace`; `benchmark/hotpaths/run.py` can capture JAX
  traces around the whole run or kernel spans; `trace_annotation(...)` labels
  benchmark calls.
- [x] Inventory current memory evidence:
  `AxonSimulation.estimate()`, `record_group_memory_estimate(...)`, and
  `benchmark/runtime/pool_memory.py` estimate tensor sizes; benchmark metadata
  includes array shape/dtype/device/nbytes, JAX device `memory_stats()` when
  exposed, process/host metadata, and `nvidia-smi` GPU/VRAM snapshots when
  available.
- [x] Record the main gap: current benchmark events do not measure per-span
  Python/RSS allocation deltas, `tracemalloc` peaks, NumPy host allocation
  pressure, or JAX device-memory snapshots/profiles. Estimated tensor bytes are
  useful for planning, but they are not measured peak memory.
- [x] Add opt-in measured memory tracing to `BenchmarkSession`:
  per-span start/end RSS via `psutil`, `tracemalloc` current/peak deltas for
  Python/NumPy-visible allocations, and optional top allocation frames with a
  small configurable `--memory-top-n`.
- [x] Add optional device-memory tracing:
  best-effort JAX device `memory_stats()` snapshots, `nvidia-smi` snapshots
  for CUDA machines, and optional JAX device memory profiles using
  `jax.profiler.save_device_memory_profile(...)` after `block_until_ready()`.
  Keep `.prof` artifacts under the benchmark result directory and record the
  pprof/XProf instructions in metadata/docs. Official reference:
  https://docs.jax.dev/en/latest/device_memory_profiling.html
- [x] Add benchmark CLI flags across active runners, starting with
  `benchmark/hotpaths/run.py`:
  `--memory-trace {off,rss,tracemalloc,device,all}`,
  `--memory-top-n`, `--jax-device-memory-profile`, and a way to restrict
  device-memory profile capture to selected stages such as `kernel.wait` or
  `simulation.pool.total`.
- [x] Extend benchmark outputs:
  keep `events.jsonl` backward-compatible while adding memory fields to event
  metadata; write a `memory_summary.csv` grouped by span name with timing,
  RSS delta/peak, `tracemalloc` delta/peak, device-memory before/after when
  available, estimated tensor bytes, and retained output bytes.
- [x] Add a smoke benchmark that produces a full time+memory map for one small
  public `AxonSimulation` population and one observer-only run; compare
  measured memory with `AxonSimulation.estimate()` and flag large unexplained
  gaps rather than treating estimates as truth.
- [x] Add unit tests for memory tracing with synthetic allocations, disabled
  tracing, missing `psutil`/device stats fallbacks, JSON/CSV schema stability,
  and JAX profile metadata when JAX exposes the profiler.
- [x] Update `benchmark/README.md` after implementation so users know which
  tool answers each question: timing spans, host Python/NumPy allocation
  tracing, RSS/process memory, JAX/XLA device memory profiles, and remote GPU
  VRAM snapshots.

### P1 - Benchmark Surface Flattening

Goal: make benchmark commands and outputs clear enough that new model/compiler
performance claims are reproducible and not mixed with old solver-spike
evidence. Do this after P0.5 has defined the time+memory observability contract.

- [x] Re-audit every active benchmark entry point after P7:
  `benchmark/runtime`, `benchmark/hotpaths`, `benchmark/nrv_performance`,
  `benchmark/realistic_examples`, `benchmark/kaggle`, and
  `benchmark/solvers`.
- [x] For each benchmark command, label it as one of: public-runtime,
  hotpath-diagnostic, model-codegen, validation-only, external-comparison,
  remote-GPU, archive, or generated-output.
- [x] Update `benchmark/registry.py` and `benchmark/README.md` so the active
  benchmark surface is flat, current, and names the retained command for each
  use case.
- [x] Remove stale benchmark wording that mentions deleted public paths,
  broad solver-side observer designs, or pre-P7 membrane implementations as
  current behavior.
- [x] Define a standard benchmark result directory and metadata contract for
  model-codegen runs: git state, Python/JAX versions, backend/device,
  `AXONSCOPE_MODEL_CODEGEN_CACHE`, cache hit/miss state, source hash, cache key,
  model kind, model class, runtime route, recording policy, and cold/warm
  timing labels.
- [x] Keep `benchmark/results/` ignored and out of architecture decisions;
  summarize retained evidence in tracked docs or changelog notes only after
  fresh runs.

### P2 - Benchmarks For New Class-Based Membrane Models

Goal: measure the P7 class-based source compiler, generated-code cache, and
JAX execution path across the built-in model families before making
performance claims.

- [x] Add or adapt a model-codegen benchmark suite covering:
  `Passive`, `HodgkinHuxley`, `RattayAberham`, `Sundt`, `AxNode`, `Tigerholm`,
  `Schild94`, and `Schild97`.
- [x] Measure cold compile/cache miss, warm cache hit, generated files, source
  hash, cache key, and cache reason for built-in and custom model classes.
- [x] Add first-run and warm `AxonSimulation` timings for representative
  model-template runs separately from pure codegen/cache timings.
- [x] Include small deterministic model-step benchmarks for generated NumPy,
  generated JAX, NumPy interpreter, and JAX runtime lowering where practical.
- [x] Include short public `AxonSimulation` runs for representative axon
  templates using the new model classes:
  HH, Rattay-Aberham, Sundt, Tigerholm, Schild94/Schild97, and MRG/AxNode.
- [x] Include at least one custom user-defined `axs.membranes.Model` benchmark
  from `examples/advanced/axon_models/05_custom_membrane_authoring.py`.
- [x] Compare cold versus warm behavior with an empty codegen cache and a
  populated codegen cache. Record cache status and cache reason in metadata.
- [x] Verify benchmark correctness with targeted model-step equivalence and
  small Vm/activation comparisons before using timings.
- [x] Decide which benchmark command becomes the smoke command for P7 model
  performance regressions.
- [x] Add benchmark docs that explain when to use model-codegen benchmarks
  versus runtime/hotpath/NRV benchmarks.

### P3 - Documentation And Examples

- [x] Refaire `README.md` après la stabilisation post-P7: présenter l'API
  actuelle, le workflow `AxonSimulation`, les membranes class-based, la
  stratégie benchmark, et pointer vers les exemples/docs à jour.
- [x] Faire une mise à plat manuelle de `docs/`, `GUIDELINES.md` et
  `AGENTS.md`: relire les textes comme un utilisateur, supprimer le bruit
  historique, garder les règles opérationnelles utiles, et vérifier que les
  pages roadmap/proposal ne ressemblent pas à de la documentation courante.
- [ ] Write real notebook tutorials under `examples/tutorials/` following the
  indexed mini-course sequence.
- [ ] Add a didactic basic example for high-frequency block after block
  detection exists, so the example distinguishes propagation, activation
  failure, and true conduction block.
- [ ] Prepare proper Sphinx documentation.
- [ ] Do/update all public docstrings.
- [x] Audit public examples after benchmark flattening so examples remain
  public-API-only and benchmark/profiling material stays under `benchmark/`.

### P4 - Runtime Boundary For Inspection And Estimates

- [x] Decide whether `inspection.py` and `performance.py` may import
  `axonscope.runtime.jax.*` directly, or whether input/recording lowering
  summaries should move behind a backend inspection/estimate facade.
- [x] If the current inspection remains JAX-specific, label that explicitly in
  docs and inspection records. Decision: not needed for the public modules;
  JAX-specific details now live behind the backend benchmark facade.
- [x] If a facade is preferred, route planning, estimate, and inspection through
  the same runtime boundary used by execution without forcing device transfers.
  The JAX backend exposes host-side benchmark support in
  `src/axonscope/runtime/jax/benchmark.py`, delegated through
  `axonscope.runtime.execution`.

### P5 - Validation Policy

- [x] Run NRV validation only for numerical behavior changes.
- [x] Re-run hotpath/realistic benchmarks only when making performance claims.
- [x] Preserve the fast local acceptance loop: compileall, focused unit tests,
  architecture guardrails, and example smoke/import tests.

## Next Major Phases

### P8 - Future Bonus NumPy/SciPy Reference Solver Runtime

This is intentionally no longer the next implementation phase. The NumPy/SciPy
runtime remains valuable as a future reference/debug backend, but only after
the model/compiler surface is flat and the current JAX solver has realistic
benchmark evidence. The goal is a real reference solver runtime, not a
JAX-backed compatibility path.

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

### P9 - Cold-Run And Runtime Performance

- [x] Add a short local P9 baseline command:
  `python benchmark/hotpaths/run.py --workload cold_run_micro --sizes 1 --duration 1.0 --dt 0.02 --warmups 0 --memory-trace rss --prefix cold_run_micro`.
- [x] Record the first CPU baseline in
  `docs/benchmarks/cold_run_micro_baseline_2026_07_03.md`. Current evidence:
  model-codegen correctness `16/16 ok`; `cold_run_micro` covers retained Vm,
  observer-only VmRaster, and point-source extracellular paths in about 5.3 s
  total on local CPU.
- [x] Normalize hotpath span coverage across scalar retained-Vm and
  batch/observer routes so preparation, input lowering, kernel enqueue/wait,
  and result assembly are comparable in cold-run reports.
- [x] Add explicit hotpath CLI chunk controls: `--time-chunk-steps default`
  keeps workload defaults, an integer forces chunk size, and `none`/`unchunked`
  forces one full scan for comparison runs.
- [x] Run a local observer-only `time_chunk_steps` smoke (`n=5`, `nt=500`,
  warm CPU) and record it in
  `docs/benchmarks/p9_runtime_closeout_2026_07_04.md`.
- [x] Decide not to add a rotated or one-case-per-process cold comparison in
  P9. Keep that for future publication-grade per-path cold-start evidence if
  it becomes important.
- [x] Keep shape bucketing internal and opt-in until benchmarks show an
  end-to-end cold-run win. Current guardrail: backend-only
  `AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING`.
- [x] Decide not to add an AxonScope-owned persistent JAX compilation/cache
  policy in P9. Use compile logging/tracing for diagnosis; revisit only if
  cold compilation becomes a product requirement.
- [x] Keep recruitment amplitude sweeps sequential by default. Micro-batching is
  a future benchmark axis, not a P9 runtime default.
- [x] Keep `BatchOptions.none()` defaulting to
  `DEFAULT_OBSERVER_TIME_CHUNK_STEPS`; explicit `time_chunk_steps=None` remains
  the unchunked comparison path.
- [x] Park runtime optimization outside P9. The next optimization round starts
  from large synthetic/GPU profiling and separate validation evidence.
- [x] Keep GPU dispatch scheduling, double-cable rank-K compact `Vext`, and
  exact GPU solver improvements as future benchmark/optimization work rather
  than current P9 tasks.

### P10 - Model/Compiler Surface Cleanup And JAX Optimization Prep

Goal: flatten the public membrane/model authoring surface and the internal
compiler/runtime contracts before deeper solver work. This is the active bridge
between P7 model authoring and P11 JAX solver optimization.

- [x] Audit `src/axonscope/membranes/`, `src/axonscope/model_ir/`,
  `src/axonscope/runtime/jax/model_ir_lowering.py`,
  `src/axonscope/runtime/jax/membrane_program.py`, generated-code cache code,
  custom membrane examples, and model-codegen benchmarks against the desired
  public vocabulary and optimization contract. Initial audit recorded in
  `docs/architecture/p10_model_compiler_surface_audit_2026_07_04.md`.
- [x] Tighten rejected Python construct diagnostics for mutation,
  data-dependent Python loops, I/O, dynamic imports, object construction inside
  equations, arbitrary NumPy/JAX calls, hidden global state, and side effects.
- [x] Complete and document the public helper surface:
  `exp`, `expm1`, `log`, `log1p`, `sqrt`, `abs`, `minimum`, `maximum`, `clip`,
  `where`, `tanh`, `sigmoid`, `vtrap`, `q10`, `rates_from_tau_inf`, and
  `safe_exp`.
  - [x] Make `rates_from_tau_inf(x_inf, tau)` the canonical public helper for
    tau/inf gates. Source models now use
    `alpha_x, beta_x = rates_from_tau_inf(x_inf, tau)`, and the compiler lowers
    that tuple assignment to scalar internal alpha/beta expressions.
  - [x] Reject/defer `boltzmann` as a public helper for now: no built-in model,
    example, or benchmark uses it directly.
  - [x] Reject `nernst` as a public helper for now: current need is
    Schild-family-specific; shared Schild 94/97 Nernst logic now lives in
    `src/axonscope/membranes/models/schild_common.py`.
  - [x] Audit concentration/current conversion formulas before exposing any
    helper. Tigerholm Na/K concentration dynamics and Schild Ca/NaCa/pump
    formulas are model- or family-specific today, so they stay local unless at
    least two independent model families need the same public operation.
- [x] Add explicit public source syntax for currents whose conductance/reversal
  cannot be inferred from `I_x = g_x * (Vm - E_x)`. Use
  `@currents(conductances={"I_x": "g_x"}, reversals={"I_x": "E_x"})`; both
  terms are required and must reference current outputs/source symbols.
- [x] Extend mechanism semantics beyond ordered sections: expose
  mechanism-level dependencies in reports, preserve boundaries for
  optimization/fusion, and apply the same readable shape to complex built-ins
  where useful. Source compilation now records `source_sections` and
  `source_mechanisms`, and `explain()` reports each mechanism's assignments and
  external dependencies.
- [x] Extend semantic validation to purity/source provenance, unsupported
  helper calls, duplicate exports, duplicate observable names, and
  recording/output compatibility.
  - [x] Validate current linearization terms: conductance expressions must be
    conductance density and reversal expressions must be voltage.
  - [x] Validate source metadata consistency for source-backed models:
    `source_outputs`, `source_provenance`, `source_sections`, and
    `source_mechanisms`.
  - [x] Keep generic duplicate-observable aggregation and public recording
    output semantics explicit before exposing custom observables broadly:
    public gate/state/generic-observable names are component-qualified, while
    current/conductance groups remain the only automatic aggregates.
  - [x] Reject duplicate Model IR observable and step-diagnostic names while
    keeping current/conductance duplicate aggregation explicit.
- [x] Make `explain()` and compiler reports useful to humans: show public model
  names, equations, gates, currents, observables, generated targets, cache
  identity, and optimization summaries without exposing Model IR as required
  user knowledge.
  - [x] Add a model-level `explain()` summary for component labels and public
    recording output names. Composite reports now show `label -> model_kind`,
    qualified gate/state/observable names, and current/conductance aggregates.
  - [x] Add explicit optimization summaries and target-specialized identity
    details without making Model IR required user knowledge. Generated target
    reports now include a backend lowering key, cache/source identity,
    static-shape policy, recording policy, precision policy, and optimization
    level.
- [x] Broaden generated-artifact identity to the report-time target-specialized
  key:
  internal graph hash, optimized graph hash, backend lowering key, static
  shapes, recording policy, parameter specialization, dtype/precision,
  optimization level, compiler/helper versions, and dependency hashes. Actual
  runtime/cache-key specialization is moved to P11's benchmark-gated optimizer
  track.
  - [x] Expose generated `graph.json` and `optimized_graph.json` content hashes
    in `inspect_generated_code()` and `explain()` reports.
  - [x] Add report-time backend lowering key, static-shape/recording policy,
    precision policy, and optimization-level details.
  - [x] Add parameter-specialization, compiler/helper version, and dependency
    hash details to the report-time target-specialized identity.
- [x] Define target-specific lowering hooks for JAX and future NumPy intrinsics
  while keeping scientific semantics target-neutral. Generated source now routes
  imports and intrinsic helper prelude through target specs instead of inline
  target conditionals.
- [x] Make recording-aware output pruning part of the compiler plan:
  requested Vm/probes/observables should determine retained outputs before
  backend lowering. Runtime allocation/transport pruning is moved to P11's
  JAX optimization track.
  - [x] Split solver-required generated outputs from recording-requested output
    groups in `OutputPruningPlan`, including gates, currents, conductances,
    membrane state, generic observables, and step diagnostics.
- [x] Move backend-neutral and JAX-specific optimizer/fusion closeout to P11 so
  common subexpression elimination, unused diagnostic pruning, stable
  optimized-graph hashing, generated conductance terms, state prepare/finalize
  updates, diagnostics, requested-observable pruning, composite generated
  programs, and unrequested-array transport pruning are gated by realistic
  benchmarks.
  - [x] Make the composite/generated boundary explicit: single-source membranes
    report loaded generated model steps, while multi-source composites fall
    back to the interpreter path and expose `multi_source_fallback` in benchmark
    metadata.
- [x] Move generated execution beyond the P7-supported class subset to P11:
  composite generated programs, more aggressive recording-aware pruning, and
  direct solver-kernel fusion remain tracked below.
- [x] Define duplicate-name aggregation semantics for generic observables before
  exposing custom observables as public recording outputs. `Composite` now uses
  component labels as the public namespace; duplicate component kinds require
  explicit labels, and generic observables are not silently aggregated.
- [x] Update affected docs and examples in the same work. P10 updated the
  architecture audit, changelog, and composite recording example; broad README
  and tutorial refresh remains tracked outside P10.
- [x] Keep model-codegen/model-step benchmarks as the performance gate for
  model/compiler changes; only make speed claims after fresh runs with recorded
  cache state, target, dtype/precision, backend, device, and git state. Fresh
  timing claims are deferred to P11.

### P11A - Benchmark Reset And Evidence Surface

Goal: restart benchmarking from a clean, reproducible surface before optimizing
the JAX solver. `src/axonscope/benchmarking` should be the small public
interface for enabling traces and collecting reports; real benchmark workloads,
campaigns, presets, launchers, analysis, and generated outputs belong under
`benchmark/`.

- [x] Archive the current benchmark tree before rewriting it. Move old scripts,
  notebooks, reports, and results that are not part of the new surface to
  `benchmark/legacy/pre_p11/` with a short inventory README. Keep unfinished
  ideas visible there; do not treat old raw results as current evidence.
- [x] Clean generated junk from the benchmark surface (`__pycache__`,
  `.DS_Store`, stale ad-hoc outputs) and make `benchmark/results/` and new
  generated reports ignored/generated-only.
- [x] Redesign `src/axonscope/benchmarking` as an interface, not a workload
  home:
  `BenchmarkOptions`, `BenchmarkSession`, `enable_benchmark`,
  `benchmark(...)`, `benchmark_span`, `record_benchmark_metadata`,
  environment capture, report writing, and a generic profiling option. The
  public modules are now thin facades; concrete session/memory/report runtime
  support lives in private runtime code, and workflow-specific benchmark cases
  stay out of `src`.
- [x] Add a backend-neutral profiling option to the benchmark interface:
  `profile=True`, `profile_backend="auto|jax|none"`, `profile_output=...`,
  with JAX `start_trace`/TensorBoard/Perfetto traces delegated through
  `axonscope.runtime.execution`, so benchmark scripts do not import JAX
  internals directly.
- [x] Add optional profiler stage filters after the two curve case lists are
  validated. Keep whole-session profiling as the default until the runtime
  stage map is stable. Curve scripts expose
  `--jax-device-memory-profile-stage`, default to `kernel.wait`, and reserve
  broad stage capture for tiny trace cases.
- [x] Move or replace legacy solver benchmark helpers currently living in
  `src/axonscope/benchmarking/benchmark.py`. Public/runtime workloads should
  live in `benchmark/workloads/`; `src` should expose only reusable
  instrumentation and serialization primitives.
- [x] Instrument the actual runtime path where work happens rather than putting
  benchmark logic in the interface layer. Required stages: model/source
  compile, dispatch planning, runtime preparation, input lowering,
  stimulation/materialization, kernel dispatch, result assembly,
  analysis/post-processing, and NRV handoff where relevant.
  Runtime instrumentation now emits spans such as `dispatch.build_plan`,
  `runtime.prepare.*`, `inputs.*`, `kernel.enqueue`, `kernel.wait`,
  `results.split_batch`, `results.to_public`, plus curve-level
  `curve.build_pool`, `curve.simulate`, and `curve.analyze_activation`.
  NRV handoff remains tied to the future baseline adapter.
- [x] Standardize timing and memory tracing for every benchmark span:
  wall-clock time, warm/cold timing, synchronization boundaries,
  process RSS, `tracemalloc` current/peak and top allocation deltas, JAX device
  memory stats, `nvidia-smi`/`jax-smi`-style GPU memory when available, and
  optional JAX device-memory profile artifacts.
  `BenchmarkSession` writes span wall time and optional `rss`, `tracemalloc`,
  device, and `nvidia-smi` memory deltas; curve scripts label warmup/repeat
  phases and can save JAX device-memory profiles on `kernel.wait`.
- [x] Record machine and runtime metadata with every run: OS, Python, package
  versions, git commit/dirty state, CPU model, host RAM, GPU model/VRAM,
  backend, device, precision policy, execution policy, JAX platform, cache
  state, recording policy, observer policy, and NRV availability/version.
  `metadata.json` and `environment.json` now include host environment,
  packages, git state, CPU/GPU/RAM where available, JAX details, benchmark
  options, recording/platform/precision, and active profile settings.
- [x] Define canonical output files for all benchmark runs:
  `environment.json`, `events.jsonl`, `summary.csv`, `memory_summary.csv`,
  `cases.csv`, `artifacts/`, and optional `plots/`. Make runs resumable and
  comparable without scraping console output.
  Curve runs also write `results.csv`, `curve_summary.csv`, and
  `manifest.json`; `--resume` skips an existing `results.csv` directory.
- [x] Add a trace-analysis tool under `benchmark/analysis/` that can summarize
  generated JAX traces alongside `events.jsonl`: stage timeline, compile vs
  execute time, long-running ops, device idle gaps where detectable, memory
  profile artifacts, and links/instructions for opening the trace in
  TensorBoard/Perfetto. Initial `trace_summary.py` summarizes event durations
  and lists trace/profile artifacts; deeper JAX trace parsing remains future
  analysis work.
- [x] Create a new `benchmark/README.md` around the new surface only:
  benchmark philosophy, command map, output schema, local/GPU/NRV prerequisites,
  and how to decide whether a result is publishable.
- [x] Document the user-facing benchmark instrumentation API with two equivalent
  styles: context-manager for scripts and `enable_benchmark(...)` /
  `disable_benchmark(...)` for notebooks/debugging. The example must show time,
  memory, and JAX trace options:

  ```python
  import axonscope as axs

  with axs.benchmarking.benchmark(
      "benchmark/results/example",
      sync_device=True,
      record_shapes=True,
      memory_trace="all",
      memory_top_n=10,
      profile=True,
      profile_backend="jax",
      jax_device_memory_profile=True,
  ):
      result = axs.AxonSimulation(...).run()
  ```

- [x] Rebuild the `benchmark/` layout from scratch around two canonical curve
  scripts:
  `benchmark/run.py`, `benchmark/curves/threshold_curves.py`,
  `benchmark/curves/recruitment_curves.py`, `benchmark/workloads/`,
  `benchmark/presets/`, `benchmark/baselines/`, `benchmark/analysis/`,
  `benchmark/kaggle/`, `benchmark/results/`, and `benchmark/legacy/`.
- [x] Implement explicit scale presets shared by both curve scripts:
  `quick`, `local_smoke`, `local_realistic`, `cpu_publication`, `gpu_smoke`,
  `gpu_trace_smoke`, `gpu_realistic`, `nrv_smoke`, and `nrv_full`. Presets
  must define repeats, warmups, `tsim`, `dt`, `Nx`, `Naxons`, precision,
  recording mode, platform, memory tracing, profiling, and output directory
  defaults. Keep GPU tracing on deliberately tiny cases only: one small pool
  and two or three amplitude evaluations, otherwise Perfetto/XPlane artifacts
  and device-memory profiles explode.
- [x] Build the local/GPU launcher on the shared script/preset interface:
  `python benchmark/run.py --script threshold_curves --preset quick --platform cpu`,
  `python benchmark/run.py --script recruitment_curves --preset gpu_smoke --platform gpu`,
  `python benchmark/run.py --list`, `--dry-run`, `--resume`, `--output`,
  `--memory-trace`, `--profile`, and `--case-filter`.
- [x] Rebuild the Kaggle runner around the same script/preset interface:
  package the repo, choose GPU shape, pass script/preset/options, stream logs,
  download `benchmark/results`, and record Kaggle hardware metadata.
  `benchmark/kaggle/run_kernel.py` now packages a generated Kaggle kernel under
  the local run directory, publishes/clones a stable branch, forwards
  `benchmark/run.py --script ... --preset ... --platform ...` plus extra
  options, streams available logs while polling, downloads the zipped result
  archive, and the Kaggle entry writes `kaggle_hardware.json` with sensitive
  Kaggle environment values redacted. The runner also supports CPU Kaggle
  comparisons: `--cpu` or `--platform cpu` selects a CPU-only Kaggle run, while
  `--platform cpu --machine-shape NvidiaTeslaP100` runs AxonScope's CPU path on
  a Kaggle GPU machine for closer CPU/GPU environment comparisons.
- [x] Build `benchmark/curves/threshold_curves.py` as the activation/block
  threshold script. Validate its concrete case list together before expanding
  implementation. Required axes: point-source AxonScope first, future NRV nerve
  baseline, `tsim`, `dt`, `Nx`, `Naxons`, FP32/FP64, full Vm, probe Vm,
  observer-only output, single-cable, double-cable, mixed populations,
  same-diameter and different-diameter cohorts, CPU, GPU, and future NRV.
  Current real execution supports point-source AxonScope activation-threshold
  runs through `AxonSimulation(..., execution_policy=...)`, with observer-only
  VmRaster and recorded-Vm post-processing paths. Block thresholds and NRV
  remain listed but intentionally rejected for real execution until their
  semantics/adapters are defined.
- [x] Build `benchmark/curves/recruitment_curves.py` as the recruitment script.
  Validate its concrete case list together before expanding implementation.
  It must share the same axes as `threshold_curves.py` so threshold and
  recruitment results can be compared table-for-table. Current real execution
  supports point-source AxonScope recruitment sweeps with the same output
  schema as threshold runs; NRV remains future baseline work.
- [x] Give both curve scripts the same core CLI vocabulary:
  `--source point_source_axonscope|nrv_nerve`, `--tsim`, `--dt`, `--nx`,
  `--n-axons`, `--precision fp32|fp64`, `--recording full_vm|probe_vm|observer_only`,
  `--cable single_cable|double_cable`, `--population single_model|mixed_models`,
  `--diameters same_diameter|different_diameters`, `--platform cpu|gpu|nrv`, `--execution-policy`,
  `--repeats`, `--warmups`, `--memory-trace`, `--profile`, `--output`,
  `--dry-run`, and `--case-filter`.
- [x] Give both curve scripts the same advanced CLI vocabulary where relevant:
  spatial recording policy (`center`, `probes`, explicit indices), observer
  criterion, amplitude bounds/tolerance/max-iterations, stimulation preset,
  seed, cache mode (`cold`, `warm`, `clear_codegen_cache`), chunking
  (`time_chunk_steps`, `amplitude_batch_size`), and result retention level
  (`summary_only`, `raw_traces`, `debug_artifacts`).
- [x] Prepare the publication-grade campaign from those two scripts only. Keep
  fixed presets, saved raw data, plots, and publication-ready summary tables.
  NRV comparison is included only after the baseline adapter contract is
  defined. The campaign matrix and publication outputs are documented in
  `benchmark/campaigns/README.md`.
- [x] Clarify baseline scope before writing adapters. Baselines are external
  comparison entry points in `benchmark/baselines/`, never AxonScope runtime
  paths. First define the NRV comparison contract; keep dense/reference JAX only
  as an equivalence/performance sanity route; add a NumPy solver baseline only
  after that solver exists.
  The baseline contract is documented in `benchmark/baselines/README.md`.
- [x] Complete P11A acceptance criteria. Local CPU `quick` threshold and
  recruitment runs now finish without NRV/GPU and write time, memory, metadata,
  case, raw-result, curve-summary, and manifest artifacts. A reduced Kaggle P100
  trace smoke passed on 2026-07-05 for commit `b6c3c92` with
  `threshold_curves`, `observer_only`, `n_axons=4`, `nx=21`, `tsim=1 ms`,
  `dt=0.05 ms`, `max_iterations=1`, `memory_trace=all`, JAX profiling, and JAX
  device-memory profiles; it produced time, memory, metadata, hardware, and
  trace artifacts under `benchmark/results/kaggle/`. A later non-reduced
  `gpu_smoke` attempt was interrupted because the preset still mixed smoke
  validation with full tracing. The corrected lightweight `gpu_smoke` passed on
  Kaggle P100 for commit `33da04a` on both `threshold_curves` and
  `recruitment_curves`, with observer-only GPU outputs, time/memory metadata,
  hardware metadata, and redacted Kaggle environment values. Use
  `gpu_trace_smoke` only for trace artifacts. No speed/memory claim is allowed
  without a fresh artifact directory and git metadata.

### P11B - Benchmark-Gated JAX Solver Optimization

Goal: optimize the current JAX solver only after P11A produces realistic,
stage-level evidence. Hotpath microbenchmarks remain diagnostic; product
decisions need realistic workflow evidence.

- [ ] Capture a clean P11A baseline before changing solver behavior:
  `quick`, `local_realistic`, key NRV smoke cases, and GPU smoke/realistic
  where available.
  CPU `quick` threshold and recruitment baselines were captured on 2026-07-05
  at commit `ecddf36` under `benchmark/results/p11b_baseline/`, with clean git
  metadata and full P11A output artifacts. Additional CPU baselines were
  captured on 2026-07-05 at commit `7ebe7c3`: small `quick` observer-only runs
  with `memory_trace=all` for a full tracing sanity check, matching `quick`
  calibration runs with `memory_trace=rss`, and bounded `local_realistic`
  threshold/recruitment runs (`Naxons=64`, `Nx=101`, `tsim=20 ms`,
  `dt=0.005 ms`, observer-only, one repeat, no warmup) with RSS tracing.
  Operational rule: keep `memory_trace=all`, JAX profiling, and device-memory
  profiles for tiny trace cases only, such as one pool and a few amplitudes;
  use `memory_trace=off` or `rss` for timing-focused larger local/GPU sweeps,
  and run separate tiny `device`/`all` memory traces when the question is
  allocation or profiler cartography. Device traces sample JAX memory stats and
  `nvidia-smi` around spans, so they can visibly perturb fine GPU timing.
  Benchmark presets now follow that split: `gpu_trace_smoke` keeps heavy GPU
  tracing, while local realistic, CPU publication, GPU smoke, and GPU
  realistic presets default to lightweight RSS timing.
  Remaining before optimization claims: selected NRV smoke and GPU
  smoke/realistic artifacts on the agreed hardware.
- [x] Start optimization from a cold-path audit for large synthetic/GPU
  populations (`n=1000`): split `build pool`, `dispatch.build_plan`,
  `runtime.prepare`, `inputs.*`, `kernel.dispatch_jax`, memory pressure, and
  result assembly before changing kernel routes or scheduling.
  `benchmark/analysis/cold_path_audit.py` now converts fresh curve benchmark
  result directories into `cold_path_stage_rows.csv`,
  `cold_path_group_summary.csv`, and plots for pool build, dispatch, runtime
  preparation, input lowering, kernel, and result assembly. First bounded CPU
  audit was generated on 2026-07-05 for `threshold_large_cpu_7ebe7c3` and
  `recruitment_large_cpu_7ebe7c3`. A clean `n=1000` CPU scout was generated on
  2026-07-05 at commit `f895a03` for short observer-only threshold and
  recruitment runs (`Nx=31`, `tsim=2 ms`, `dt=0.02 ms`, RSS tracing) under
  `benchmark/results/p11b_baseline/cold_path_n1000_cpu_scout_f895a03`.
  The matching Kaggle P100 scout was generated on 2026-07-05 at commit
  `f225afd` for short observer-only threshold and recruitment runs
  (`Naxons=1000`, `Nx=31`, `tsim=2 ms`, `dt=0.02 ms`, device tracing) and
  summarized under
  `benchmark/results/p11b_baseline/cold_path_n1000_gpu_p100_scout_f225afd`.
  `benchmark/analysis/bottleneck_report.py` adds the next triage layer from
  `events.jsonl`: exclusive self-time ranking, group shares, cache signals,
  memory context, and a Markdown bottleneck summary across CPU/GPU runs.
  The first optimization target is now pool/plan/runtime reuse between
  amplitude evaluations before changing solver kernels.
- [ ] Optimize current JAX preparation and lowering before new solver routes:
  runtime/cohort caches, input lowering, static-footprint factorized `Vext`,
  zero/sparse `Iinj`, recording-aware pruning, and result assembly.
  First cleanup landed in `33535ee`: curve benchmarks now build each phase pool
  once and update only extracellular drive stimuli between amplitude
  evaluations, matching the public protocol examples. This removes repeated
  benchmark-side pool construction as a confounder; it is not yet a solver
  speed claim. Next cleanup: prepared-cohort caching now reuses static row
  positions while refreshing replaced stimulation rows, and factorized
  footprint cache keys now follow static footprints rather than transient drive
  objects, so amplitude sweeps can update waveform current without recomputing
  footprints. Validated with targeted unit tests and a small CPU `quick`
  threshold run under
  `benchmark/results/p11b_cache_smoke/threshold_quick_static_footprint_cache`.
  Second cleanup: the curve benchmark pool builder now reuses one descriptive
  axon template per `(cable, diameter)` and keeps row-specific
  `AxonInstance` stimulations separate. This removes same-diameter workload
  model reconstruction as a benchmark confounder before interpreting solver
  bottlenecks. Validated with benchmark-suite unit tests and clean CPU scouts
  under
  `benchmark/results/p11b_baseline/bottleneck_n1000_cpu_template_reuse_8330374`;
  this is still a benchmark-harness cleanup, not a solver speed claim.
  Third cleanup: amplitude updates now reuse one benchmark stimulus build per
  repeated amplitude value while keeping each row's stimulation object separate.
  This reduces recruitment-curve update overhead for common-amplitude sweeps
  and was validated with benchmark-suite unit tests plus clean CPU scouts under
  `benchmark/results/p11b_baseline/bottleneck_n1000_cpu_stimulus_cache_038b7a6`.
  Fourth cleanup: VmRaster observer plan cache keys no longer depend on
  replaced stimulation object identities, because the lowered plan depends on
  observer definitions, positions, original indices, and dtype. Clean CPU
  scouts under
  `benchmark/results/p11b_baseline/bottleneck_n1000_cpu_vm_raster_plan_cache_69f06da`
  now show one observer-plan miss followed by hits across amplitude updates.
  Fifth cleanup: AxonScope template axon/fiber diameters are now quantized at
  construction time to 0.01 um for diameters up to 1 um and 0.1 um above 1 um,
  so solver signatures and benchmark cohorts naturally reuse nearby diameter
  variants before CPU/GPU Kaggle comparisons. The matching Kaggle P100
  recruitment scouts were captured on 2026-07-05 at commit `9b0dda8` for
  `Naxons=1000`, `Nx=31`, `tsim=2 ms`, `dt=0.02 ms`, observer-only output,
  three amplitudes, and different-diameter cohorts. Both CPU and GPU outputs
  report 63 unique quantized diameters for 1000 axons, with clean git metadata,
  and are summarized under
  `benchmark/results/p11b_baseline/bottleneck_kaggle_rounded_diam_cpu_gpu_9b0dda8`.
  The scout shows a modest GPU kernel enqueue improvement but extra GPU
  dispatch/runtime overhead on this small workload, so larger GPU-realistic
  cases are still required before any speed claim.
  Follow-up bounded-realistic Kaggle recruitment scouts were captured on
  2026-07-05 at commit `a48fc36` for `Naxons=1000`, `Nx=101`, `tsim=10 ms`,
  `dt=0.01 ms`, observer-only output, five amplitudes, and different-diameter
  cohorts. They are summarized under
  `benchmark/results/p11b_baseline/bottleneck_kaggle_realistic_rounded_diam_cpu_gpu_a48fc36`
  and plotted under
  `benchmark/results/p11b_baseline/cold_path_kaggle_realistic_rounded_diam_cpu_gpu_a48fc36`.
  This larger run shows the GPU path clearly reducing solver dispatch time
  (`curve.simulate` about 17.8 s GPU versus 34.4 s CPU), while the CPU path
  spends about 24.6 s in 100 `kernel.dispatch_jax` spans. Because the effective
  chunking is 20 chunks per amplitude (`time_chunk_steps=50`), the immediate
  next campaign should vary `time_chunk_steps` before changing kernel code.
  The bounded recruitment `time_chunk_steps` campaign was then captured on
  Kaggle P100 at commit `f72ed02` for effective/default 50, 100, 250, 500, and
  1000 steps per chunk, with CPU and GPU runs for the same observer-only
  workload. The summary lives under
  `benchmark/results/p11b_baseline/time_chunk_campaign_kaggle_a48fc36_summary_with_default50`.
  On this case, GPU improves as chunks get larger, with `curve.simulate` about
  17.8 s at 50, 15.1 s at 100, 14.1 s at 250, 15.0 s at 500, and 12.4 s at
  1000. CPU does not show a clean monotonic win: 50 and 100 stay around
  34 s, 250 is about 35.4 s, 500 rises to about 41.8 s, and 1000 is about
  37.7 s with a large `kernel.finalize_observer` span. Next step before a
  default change: inspect VmRaster finalization/full-scope observer handling
  and repeat a narrower CPU sanity run if needed.
  Follow-up cleanup: explicit `time_chunk_steps >= Nt` is now clamped to one
  local chunk instead of being normalized to the unchunked/full-scope observer
  path. A single full VmRaster chunk also bypasses Python repacking during
  combination. This fixes trace interpretability for `chunk_steps == Nt` before
  any default change; it is not yet a new benchmark claim.
  Remaining optimization targets are broader dispatch/runtime reuse,
  lowering/transport pruning, and result assembly.
- [ ] Explore low-level observer/kernel bottlenecks before high-level workflow
  scheduling changes. Start from the corrected `time_chunk_steps == Nt`
  behavior and compare explicit one-chunk versus unchunked/full-scope observer
  paths, `kernel.dispatch_jax`, `kernel.enqueue`, `kernel.wait`,
  `kernel.finalize_observer`, host/device materialization boundaries, and
  VmRaster combine/finalize costs. Use tiny traced cases for JAX profiling and
  device-memory artifacts, then confirm with bounded realistic CPU/GPU runs.
  First tooling step: curve benchmarks now distinguish `time_chunk_policy`
  values `default`, `unchunked`, and `explicit`; CLI forms
  `--time-chunk-steps unchunked`, `none`, and `N` map to the intended
  `BatchOptions` paths. A local quick CPU smoke under
  `benchmark/results/p11b_chunk_policy_smoke` confirms default observer-only
  uses two 50-step local chunks, `unchunked` uses the full-scope observer path,
  and explicit `100` on `Nt=100` uses one local chunk.
  Second tooling step: `benchmark/campaigns/time_chunk_sweep.py` now runs each
  policy in a separate Python process and writes one result directory per
  policy plus `time_chunk_sweep_summary.csv` and
  `time_chunk_sweep_report.md`. A local quick CPU campaign smoke under
  `benchmark/results/p11b_time_chunk_sweep_quick` validated the summary
  columns for `curve.simulate`, `kernel.enqueue`, `kernel.dispatch_jax`,
  `kernel.wait`, `kernel.finalize_observer`, and observed chunk metadata.
  CPU bounded sweep evidence was captured under
  `benchmark/results/p11b_time_chunk_sweep_cpu` for `Naxons=1000`, `Nx=101`,
  `tsim=10 ms`, `dt=0.01 ms`, five amplitudes, observer-only output, and
  different-diameter cohorts. Warm `curve.simulate` means were about
  2.95 s/amplitude for explicit 50, 3.00 s for default/250, 3.19 s for 500,
  and 3.27 s for unchunked/1000. Explicit 1000 and unchunked still spend about
  2.84 s/amplitude in `kernel.finalize_observer`, consistent with deferred
  JAX materialization rather than a cheap observer finalization path.
  Matching Kaggle P100 CPU/GPU observer-only sweeps were captured at commit
  `a237734`: CPU-path artifacts live under
  `benchmark/results/kaggle/20260705_205140_time_chunk_sweep_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  and GPU-path artifacts under
  `benchmark/results/kaggle/20260705_205140_time_chunk_sweep_quick_NvidiaTeslaP100/outputs/extracted_gpu`.
  On Kaggle CPU, explicit 50 is fastest at about 34.2 s total, then 250
  around 35.0 s and default around 36.0 s; 500, 1000, and unchunked are slower
  around 36.8-37.2 s, with unchunked/1000 dominated by about 30.5-30.9 s in
  `kernel.finalize_observer`. On Kaggle GPU, large chunks win clearly:
  explicit 1000 and unchunked are about 12.35 s total, 250 is about 13.9 s,
  500 about 14.4 s, and 50/default about 17.5-18.1 s. This points toward a
  backend-dependent observer chunk policy, but full/probe Vm still need their
  own evidence before changing defaults broadly.
  Third tooling step: observer-only chunk combination is now traced explicitly
  as `kernel.combine_observer_chunks` inside `kernel.enqueue`, and
  `time_chunk_sweep_summary.csv`/`time_chunk_sweep_report.md` include combine
  and finalize timings side by side. Local smoke artifacts under
  `benchmark/results/p11b_time_chunk_sweep_combine_multichunk_smoke` confirm
  that multi-chunk local VmRaster paths expose combine time, while unchunked
  full-scope observer paths keep that column at zero. This is trace plumbing,
  not yet an optimization claim.
  Fourth tooling step: batch result assembly now has generic result-side spans:
  `results.materialize_vm`, `results.assemble_rows`, and
  `results.assemble_cohort_record`. The time-chunk sweep CSV includes these
  alongside `results.split_batch`, so full Vm, probe Vm, and observer-only
  paths can be compared on the same output-boundary vocabulary before making
  any recording-mode-specific optimization. Tiny local smoke artifacts live
  under `benchmark/results/p11b_result_trace_full_smoke`,
  `benchmark/results/p11b_result_trace_probe_smoke`, and
  `benchmark/results/p11b_result_trace_observer_smoke`.
  Fifth tooling step: `benchmark/campaigns/time_chunk_sweep.py` now accepts
  `--recordings full_vm,probe_vm,observer_only` and writes matrix outputs under
  `<recording>/<policy>` while keeping single `--recording` commands
  compatible with the old layout. A local matrix smoke under
  `benchmark/results/p11b_recording_matrix_smoke` validated `default` and
  explicit `100` across full Vm, probe Vm, and observer-only outputs.
  Sixth tooling step: low-level JAX/runtime spans now split kernel preparation,
  chunk setup, JAX dispatch, chunk bookkeeping, trace concatenation, VmRaster
  to-host finalization, result trimming, and Vm to-host materialization:
  `kernel.prepare_inputs`, `kernel.prepare_arrays`, `kernel.prepare_state`,
  `kernel.prepare_observer_tables`, `kernel.materialize_inputs`,
  `kernel.prepare_factorized_forcing`, `kernel.chunk_setup`,
  `kernel.dispatch_jax`, `kernel.chunk_bookkeeping`,
  `kernel.concat_trace_chunks`, `kernel.finalize_observer.to_host`,
  `results.trim_padded_batch`, and `results.materialize_vm.to_host`. Curve
  scripts also expose `curve.activation_definition`,
  `curve.runtime_options`, and `curve.construct_simulation` before
  `curve.simulate`. These timings are included in
  `time_chunk_sweep_summary.csv`, `time_chunk_sweep_report.md`, and
  `benchmark/analysis/time_chunk_matrix_report.py` without changing solver
  behavior or default time-chunk policy. Tiny CPU smoke artifacts live under
  `benchmark/results/p11b_low_level_span_smoke`,
  `benchmark/results/p11b_low_level_span_sweep_smoke_ok`, and
  `benchmark/results/p11b_low_level_span_matrix_smoke`.
  First Kaggle low-level recruitment cartography with the new spans was
  captured on 2026-07-06 at commit `a73b1f0` on the same P100 image for CPU
  and GPU AxonScope paths, with policies `default`, explicit `1000`, and
  `unchunked`, recordings `full_vm`, `probe_vm`, and `observer_only`,
  `Naxons=1000`, `Nx=101`, `tsim=10 ms`, `dt=0.01 ms`, three amplitudes,
  different-diameter cohorts, one repeat, and no warmup. CPU artifacts live
  under
  `benchmark/results/kaggle/20260706_205203_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_205122_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and plots/report under
  `benchmark/results/p11b_low_level_recruitment_cpu_gpu_a73b1f0`.
  All 18 cases passed. Best rows are still kernel-dominated; the new spans
  make `kernel.chunk_setup` visible as a large CPU observer-only/default cost,
  while GPU best rows keep substantial unattributed curve/setup time that
  should be split further before optimizing. Treat this as bottleneck
  cartography, not a policy decision.
  Matching Kaggle low-level threshold cartography was captured on 2026-07-06
  at commit `af439cc` on the same P100 image, with the same reduced policies,
  recordings, size, precision, and tracing options. CPU artifacts live under
  `benchmark/results/kaggle/20260706_210319_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_210319_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  threshold plots/report under
  `benchmark/results/p11b_low_level_threshold_cpu_gpu_af439cc`, and the
  combined threshold/recruitment CPU/GPU report under
  `benchmark/results/p11b_low_level_threshold_recruitment_cpu_gpu_af439cc`.
  All 18 threshold cases passed. The threshold map reinforces the low-level
  split: CPU full/probe best rows are dominated by `kernel.wait`, CPU
  observer-only with explicit `1000` exposes a large
  `kernel.finalize_observer.to_host` cost, and GPU rows still retain large
  curve/setup time outside the current sub-spans. Use the combined report as
  the next optimization baseline.
  Follow-up trace hygiene: `benchmark/analysis/bottleneck_report.py` now
  supports `--phase repeat`, inherited phase labels for nested spans, and
  unique run labels across CPU/GPU/recording/policy directories. A
  representative repeat-phase report over current threshold/recruitment Kaggle
  runs lives under
  `benchmark/results/p11b_repeat_bottleneck_cpu_gpu_current`. It shows CPU
  full-Vm paths dominated by `kernel.wait`, CPU observer-only explicit `1000`
  dominated by `kernel.finalize_observer.to_host`, and GPU device-traced paths
  dominated by `kernel.enqueue` self-time. Direct event inspection shows
  repeated ~50 ms gaps between spans on GPU device traces, consistent with
  memory tracing overhead; treat those GPU `device` timings as memory
  cartography, not pure solver timing. Next measurement before low-level solver
  changes: rerun representative CPU/GPU timing with `memory_trace=off` or
  `rss`, keep device/all tracing to one pool and a few amplitudes, and compare
  the new `kernel.prepare_inputs` span against `kernel.dispatch_jax` and
  `kernel.wait`.
  Clean RSS timing was captured next on Kaggle P100 at commit `f5862eb` for a
  reduced recruitment matrix: CPU and GPU AxonScope paths, recordings
  `full_vm`, `probe_vm`, and `observer_only`, policies `default`, explicit
  `1000`, and `unchunked`, `Naxons=1000`, `Nx=101`, `tsim=10 ms`,
  `dt=0.01 ms`, three amplitudes, different-diameter cohorts, one repeat, no
  warmup, RSS tracing only, and no JAX/device profiling. CPU artifacts live
  under
  `benchmark/results/kaggle/20260706_213755_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_213808_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and the combined plots/reports under
  `benchmark/results/p11b_clean_timing_cpu_gpu_f5862eb`. All 18 cases passed.
  Best-policy times are now coherent: CPU full/probe/observer are about
  22.6 s, 21.7 s, and 21.9 s, while GPU full/probe/observer are about 10.0 s,
  8.9 s, and 8.7 s, giving roughly 2.3-2.5x GPU speedup on this workload.
  CPU full/probe best rows are still dominated by `kernel.wait`; CPU
  observer-only depends strongly on the observer finalization/chunk path. GPU
  best rows show a mixed bottleneck rather than pure solver time:
  `kernel.enqueue` and `kernel.dispatch_jax` are visible, but
  `curve.analyze_activation`, `curve.build_pool`, `runtime.prepare`, and input
  lowering remain large enough to split or cache before deeper solver-kernel
  optimization. This replaces the previous device-traced GPU timing as the
  timing baseline; keep heavy device traces for tiny allocation/profiler cases.
  Follow-up instrumentation now splits benchmark workflow overhead without
  changing solver behavior: `curve.build_pool` exposes diameter grid, spatial
  layout, row/stimulation construction, and template builds;
  `curve.update_amplitudes` exposes row updates and stimulus builds; and
  `curve.analyze_activation` exposes VmRaster extraction/value computation,
  result-side analysis, and value materialization. `time_chunk_sweep_summary.csv`
  and `time_chunk_sweep_report.md` now include build-pool, simulation
  construction, and activation-analysis timings, and the stage classifier
  correctly groups `curve.analyze_activation*` with result assembly. The matrix
  report also accepts direct single-recording campaign layouts as well as
  `recording/policy` matrix layouts. A tiny CPU smoke lives under
  `benchmark/results/p11b_workflow_span_smoke`.
  Matching Kaggle P100 CPU/GPU workflow-span evidence was captured on
  2026-07-06 at commit `c4e3e53` with the same reduced recruitment matrix as
  the clean RSS baseline: three recordings, policies `default`, explicit
  `1000`, and `unchunked`, `Naxons=1000`, `Nx=101`, `tsim=10 ms`,
  `dt=0.01 ms`, three amplitudes, different-diameter cohorts, one repeat, no
  warmup, RSS tracing only, and no JAX/device profiling. CPU artifacts live
  under
  `benchmark/results/kaggle/20260706_220316_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_220331_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and combined plots/reports under
  `benchmark/results/p11b_workflow_spans_cpu_gpu_c4e3e53`. All 18 cases
  passed. The new spans show that on this workflow `curve.build_pool` is about
  1.25-1.4 s and activation analysis is a real post-run cost: about 4.0-4.3 s
  for full Vm, 2.3-2.5 s for probe Vm, and 3.2-3.6 s for observer-only. GPU
  `curve.simulate` remains much faster than CPU, but the visible end-to-end
  workflow is now substantially shaped by pool construction and activation
  analysis. Treat this as bottleneck cartography before low-level solver work:
  next split or cache the activation analysis/result-view path so solver
  timings are not hidden by result-side workflow overhead.
  First outside-solver cleanup after this map: public VmRaster activation
  decoding now tests packed words directly instead of unpacking the full
  `(batch, raster, probe, time)` bool array; dense population
  `Activation.evaluate(...)` has a vectorized cohort path for public result
  analysis; curve benchmarks build point-source footprints directly while
  staying numerically equivalent to the public analytical helper; and curve
  activation reads only dense boolean values for full/probe Vm campaign cases
  instead of materializing one full `ActivationEvent` per row. Local CPU smoke
  artifacts live under
  `benchmark/results/p11b_outside_solver_optim_max_smoke` for
  `Naxons=1000`, `Nx=101`, `tsim=10 ms`, `dt=0.01 ms`, three amplitudes,
  different-diameter cohorts, one repeat, no warmup, and no memory/profiling
  overhead. On this local smoke, visible full/probe/observer workflow time is
  about 16.2 s, 21.7 s, and 19.3 s respectively; `curve.build_pool` is about
  0.61 s, 0.63 s, and 0.80 s; and `curve.analyze_activation` is about
  1.18 s, 1.51 s, and 0.03 s. Treat these as local validation only: rerun the
  reduced CPU/GPU Kaggle matrix before making a fresh speed or policy claim.
  Matching Kaggle P100 CPU/GPU validation passed on 2026-07-06 at commit
  `6caf6dc`, using the same reduced recruitment matrix, RSS tracing only, and
  no JAX/device profiling. CPU artifacts live under
  `benchmark/results/kaggle/20260706_224708_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_224708_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and combined plots/report under
  `benchmark/results/p11b_outside_solver_optim_cpu_gpu_6caf6dc`. All 18 cases
  passed. Versus the `c4e3e53` workflow-span baseline, best-policy visible time
  improved by about 12.6%/8.6%/14.3% on CPU full/probe/observer and
  28.1%/20.4%/32.1% on GPU full/probe/observer. `curve.build_pool` is now
  about 0.48-0.54 s on Kaggle, dense full/probe activation analysis is about
  0.9-1.2 s, and observer-only activation analysis is about 0.03 s. This
  confirms those outside-solver costs are no longer the dominant blocker for
  the next low-level solver/kernel optimization pass.
  First low-level observer follow-up: single-cable observer-only local chunk
  execution now reuses one VmRaster zero-state template per run instead of
  recreating it for every local chunk. Local CPU before/after probes live under
  `benchmark/results/p11b_lowlevel_observer_chunk_setup_probe` and
  `benchmark/results/p11b_lowlevel_observer_chunk_template_probe` for the same
  `Naxons=1000`, `Nx=101`, `tsim=10 ms`, `dt=0.01 ms`, three-amplitude
  observer-only smoke. On this local default policy probe, `kernel.chunk_setup`
  drops from about 6.5 s to about 0.09 s, while CPU work is reattributed mostly
  to `kernel.dispatch_jax`; total `curve.simulate` falls from about 14.48 s to
  about 13.44 s. Treat this as trace cleanup plus a small runtime win, not a
  policy decision. The analogous double-cable observer path intentionally stays
  untouched until the dedicated double-cable optimization pass.
  Matching Kaggle P100 CPU/GPU validation passed on 2026-07-06 at commit
  `27c86c6`. CPU artifacts live under
  `benchmark/results/kaggle/20260706_231351_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260706_231407_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and combined plots/report under
  `benchmark/results/p11b_vmraster_template_cpu_gpu_27c86c6`. All 18 cases
  passed. Versus the `6caf6dc` reduced-matrix run, CPU observer-only/default
  `kernel.chunk_setup` drops from about 15.0 s to about 0.10 s, with the work
  reattributed mostly to `kernel.dispatch_jax`; total CPU observer-only best
  visible time improves by about 3.1%. Full/probe CPU rows also improve by
  about 3.6-3.9% in this run, while GPU changes stay within small run-to-run
  variation. This confirms the fix mainly improves trace attribution and
  removes per-chunk state setup overhead, not the core solver bottleneck.
  Second low-level single-cable follow-up: factorized extracellular forcing now
  keeps a runtime cache keyed by the static footprint, cable coefficients, and
  dtype, so amplitude sweeps can reuse the lowered single-cable forcing
  footprint after the first miss. The lowering formula was also rewritten from
  scatter-style `.at[].set` updates to direct slicing/concatenation. Local CPU
  validation on 2026-07-06 passed `compileall`, `tests/unit/solvers/test_batch.py`,
  and a six-case quick recruitment smoke under
  `benchmark/results/p11b_prepare_forcing_local_final_bis`. The useful trace
  signal is cache behavior, not end-to-end speed: first `kernel.prepare_factorized_forcing`
  misses are still about 0.5-0.9 s on the local Mac, while second amplitude
  hits are about 0.1 ms. The total local smoke remained too noisy for a speed
  claim, so confirm with a reduced Kaggle CPU/GPU matrix before claiming
  workflow improvement. Next low-level target: build the same cartography for
  double-cable preparation/forcing paths before changing double-cable kernels.
  Matching quick Kaggle P100-image validation was captured on 2026-07-07 for
  commit `8ab92c3` with observer-only recruitment sweeps over policies
  `default`, `unchunked`, 50, 250, 500, and 1000. CPU-path artifacts live under
  `benchmark/results/kaggle/20260707_004323_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  and GPU-path artifacts under
  `benchmark/results/kaggle/20260707_004350_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`.
  All 12 cases passed. The cache signal is stable: CPU misses are about
  253-277 ms and GPU misses about 291-319 ms, while all CPU/GPU hits are about
  0.04-0.08 ms. This confirms the forcing cache behavior on Kaggle; it remains
  observer-only quick evidence, so broader full/probe workflow speed claims
  still need a reduced matrix if they matter.
  Double-cable cartography follow-up: double-cable kernel preparation now
  exposes separate spans for `kernel.prepare_double_coefficients`,
  `kernel.prepare_factorized_vext`, and `kernel.prepare_observer_state`.
  These spans are surfaced in `benchmark/campaigns/time_chunk_sweep.py`
  summaries without double-counting nested coefficient work. Local CPU
  validation on 2026-07-07 passed compileall on active paths, the focused
  batch/dispatcher/time-chunk tests, and a four-case double-cable recruitment
  smoke under `benchmark/results/p11b_double_cable_cartography_local`.
  On this small CPU smoke, double-cable coefficient preparation is under 1 ms,
  factorized-Vext prep is about 0.25 ms for observer-only, while chunk-local
  VmRaster observer-state preparation is about 45-50 ms. Treat this as
  instrumentation/cartography only. Next, run the same reduced CPU/GPU Kaggle
  double-cable matrix before changing double-cable solver kernels.
  Matching Kaggle P100-image CPU/GPU double-cable recruitment matrices were
  then captured at commit `890c252` with policies `default`, explicit `1000`,
  and `unchunked`, recordings `full_vm`, `probe_vm`, and `observer_only`,
  `Naxons=1000`, `Nx=101`, `tsim=10 ms`, `dt=0.01 ms`, three amplitudes,
  different-diameter cohorts, one repeat, no warmup, RSS tracing only, and no
  JAX/device profiling. Both runs passed but were very long: extracted CPU
  artifacts live under
  `benchmark/results/kaggle/20260707_085822_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260707_085839_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  combined plots/report under
  `benchmark/results/p11b_double_cable_cartography_cpu_gpu_890c252`, and the
  bottleneck report under
  `benchmark/results/p11b_double_cable_bottleneck_report_890c252`. The clear
  bottleneck was not the solver: `runtime.prepare.stack_membrane` consumed
  about 138-140 s on CPU and about 306-315 s on GPU because repeated reads of
  derived membrane-program values (`g_bar`, `E_rev`, static signatures/state
  specs) were recalculated while encoding many parametrized double-cable
  membrane compartments.
  Compiler/backend cleanup: `JaxMembraneProgram` now memoizes those derived
  static values and invalidates only the rate-table-dependent signature when
  rate tables are enabled/disabled. This keeps the runtime structural and free
  of model-family-specific shortcuts. Local validation on 2026-07-07 shows
  50k passive-program `g_bar` reads drop from about 41 s to about 0.006 s, and
  a short `Naxons=1000`, double-cable observer-only CPU run now completes in
  14 s with `runtime.prepare.stack_membrane` about 2.14 s for 89,000
  compartments.
  Matching post-fix Kaggle P100-image CPU/GPU validation passed on 2026-07-07
  at commit `64fca75` with the same reduced double-cable recruitment matrix.
  CPU artifacts live under
  `benchmark/results/kaggle/20260707_102157_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU artifacts under
  `benchmark/results/kaggle/20260707_102122_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  combined plots/report under
  `benchmark/results/p11b_double_cable_program_cache_cpu_gpu_64fca75`, and
  repeat-phase bottlenecks under
  `benchmark/results/p11b_double_cable_program_cache_bottlenecks_64fca75`.
  Versus `890c252`, best visible times dropped from about 163-165 s CPU and
  318-322 s GPU to about 23-26 s CPU and 13-16 s GPU. The previous
  `runtime.prepare.stack_membrane` bottleneck dropped to about 1.5 s on CPU
  and about 1.9-2.1 s on GPU. The remaining double-cable map is now split:
  CPU full/probe rows are solver-wait dominated, CPU observer-only still
  exposes VmRaster chunk/finalize costs depending on policy, and GPU rows are
  dominated by host-side double-cable runtime preparation, especially
  `runtime.prepare.stack_extracellular` around 5.3-6.1 s. Next low-level
  target before solver algorithm work: reduce/factor/cache double-cable
  extracellular runtime stacking and then revisit CPU observer-only
  chunk/finalize behavior.
  Local cleanup now stages double-cable extracellular rows entirely on the
  host: `_extracellular_runtime_numpy` returns NumPy arrays, stacked runtime
  preparation caches rows by cable signature, transfers two batched blocks
  (`space` and `edge`) instead of thousands of per-row JAX arrays, and records
  `extracellular_stack_unique_rows`/cache hits in benchmark metadata. VmRaster
  local chunk combination now repacks packed words instead of iterating over
  every time step, keeps the combined chunk state host-side for immediate
  finalization, and adds benchmark-only `kernel.wait` spans before chunk
  combination/finalization so deferred JAX compute is not misattributed to
  `combine` or `to_host`. Local validation passed focused double-cable and
  VmRaster tests, plus CPU smokes under
  `benchmark/results/p11b_stack_extracellular_local_smoke` and
  `benchmark/results/p11b_extracellular_vmraster_wait_local_smoke`: for
  `Naxons=128`, `runtime.prepare.stack_extracellular` is about 0.10 s with
  5 unique rows and 123 cache hits, `kernel.finalize_observer.to_host` is
  about 0.05 ms/repeat, and remaining observer chunk time is explicitly
  attributed to `kernel.wait`. Next validation: rerun the reduced double-cable
  CPU/GPU Kaggle matrix before claiming the GPU `stack_extracellular` bottleneck
  is fixed, then proceed to true low-level double-cable solver cartography.
  Matching Kaggle P100-image CPU/GPU validation passed on 2026-07-07 at commit
  `9df793b` with the same reduced double-cable recruitment matrix. CPU-path
  artifacts live under
  `benchmark/results/kaggle/20260707_105820_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU-path artifacts under
  `benchmark/results/kaggle/20260707_105839_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  combined plots/report under
  `benchmark/results/p11b_double_cable_stage_cpu_gpu_9df793b`, and repeat-phase
  bottlenecks under
  `benchmark/results/p11b_double_cable_stage_bottlenecks_9df793b`. The
  targeted GPU bottleneck is resolved: `runtime.prepare.stack_extracellular`
  falls from about 5.3-6.1 s at `64fca75` to about 0.12-0.14 s, with 5 unique
  extracellular rows and 995 cache hits for 1000 axons. Best GPU visible times
  improve by about 32-37% across full Vm, probe Vm, and observer-only. CPU
  `stack_extracellular` also falls from about 0.62-0.69 s to about 0.12-0.13 s,
  but the total CPU run is slower than `64fca75` because kernel/result spans
  dominate this Kaggle run; do not treat it as a CPU regression claim without a
  repeat. The remaining map before solver-algorithm work: CPU full/probe are
  `kernel.wait` dominated, CPU observer-only/default still exposes large
  `kernel.chunk_setup`/inter-chunk sync attribution, and GPU is now dominated
  by solver/wait plus full/probe result assembly and the remaining
  double-cable runtime preparation.
  First true low-level solver-stage cartography tool added:
  `benchmark/analysis/double_cable_solver_stage_profile.py` profiles synthetic
  double-cable numerical stages outside the public runtime path:
  `assemble_system`, `block_solve` variants (`thomas_vmap`,
  `thomas_batched_scan`, `pcr_matrix_vmap`, `pcr_soa_vmap`,
  `pcr_soa_batched`), `vm_gate_update`, current VmRaster
  `observer_write`, and a compact `full_numeric_step` proxy. It writes
  repeat CSVs, summary CSVs, metadata, a markdown report, and plots. Local CPU
  smoke passed under
  `benchmark/results/p11b_double_cable_solver_stage_local_smoke_v2` with
  `Nx=21`, `batch_size=4`, fp32, batched coefficients, and two measured
  solver variants. Treat this as bounded low-level cartography only; it does
  not replace realistic curve benchmarks or choose runtime policy. The Kaggle
  runner now exposes it as
  `--campaign double_cable_solver_stage_profile`, and the profiler can run
  `--coefficient-mode both` to compare shared versus per-row coefficient
  layouts in one artifact. Matching Kaggle P100-image CPU/GPU solver-stage
  cartography was captured at commit `08566b3` with `Nx=21,51,101`,
  `batch_size=4,32,128`, fp32, shared+batched coefficients, five repeats, and
  one warmup. CPU-path artifacts live under
  `benchmark/results/kaggle/20260707_120045_double_cable_solver_stage_profile_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  GPU-path artifacts under
  `benchmark/results/kaggle/20260707_115954_double_cable_solver_stage_profile_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`,
  and the combined report under
  `benchmark/results/p11b_double_cable_solver_stage_cpu_gpu_08566b3`. The
  synthetic map is clear: CPU should stay on Thomas-style scan variants
  (`thomas_vmap`/`thomas_batched_scan`), while GPU should focus on PCR/SoA
  variants (`pcr_soa_vmap`/`pcr_soa_batched`, with `pcr_matrix_vmap` sometimes
  competitive). GPU only wins once batch/shape is large enough; for small
  batches CPU Thomas is faster. At the largest tested shape, best GPU PCR
  solve time is about 0.35-0.47 ms while synthetic assembly/gate/observer
  writes are about 0.25-0.35 ms each, so after solver-choice cleanup kernel
  fusion/launch overhead may become the next low-level target. Do not turn
  this into high-level runtime policy yet; validate any backend solver-layer
  change in real double-cable curve benchmarks.
  Real-workflow solver-choice validation was then captured on the same Kaggle
  P100 image at commit `ff78c4f` for double-cable observer-only recruitment,
  `Naxons=512`, `Nx=101`, `tsim=5 ms`, `dt=0.01 ms`, two amplitudes, RSS
  tracing, and no profiling. The benchmark-only solver override compares CPU
  `auto`/Thomas versus forced `pcr_soa`, and GPU `auto`/PCR-SoA versus forced
  Thomas. Artifacts live under
  `benchmark/results/p11b_real_solver_choice_cpu_gpu_ff78c4f`, with the
  dedicated cold/warm report generated by
  `benchmark/analysis/solver_choice_report.py`. Warm-path evidence confirms
  that the solver is now the dominant low-level target on CPU and strongly
  solver-sensitive on GPU: CPU forced `pcr_soa` is about 51.7x slower than
  CPU `auto`/Thomas on kernel sync, while GPU forced Thomas is about 11.2x
  slower than GPU `auto`/PCR-SoA. GPU `auto` warm still spends only about
  47.5% of `curve.simulate` in kernel sync, so GPU work should keep separating
  solver execution from compile/launch/finalize/result boundaries instead of
  assuming every remaining millisecond is numerical solve time.
  Critical reading of the solver idea docs plus a compiler/runtime audit is
  recorded in
  `docs/architecture/p11b_double_cable_solver_compiler_audit_2026_07_07.md`.
  Current decision: do not reopen old split/associative-transfer/PCR-layout
  variants without a new hypothesis; keep high-level amplitude batching for
  later. First real compiler-stage profiler added under
  `benchmark/analysis/double_cable_real_stage_profile.py`: it builds public
  AxonScope double-cable workloads, reuses current backend/runtime preparation,
  and writes `real_stage_repeats.csv`, `real_stage_summary.csv`, metadata,
  plots, and `real_stage_report.md` for generated membrane gate/conductance
  work, extracellular RHS drive, system assembly, selected block solvers, a
  one-step proxy, and VmRaster observer writes. The Kaggle runner exposes it as
  `--campaign double_cable_real_stage_profile`. Next use this to separate
  solver execution from generated membrane, assembly, launch/finalize, and
  result boundaries before adding any new solver route. First CPU/GPU Kaggle
  run passed on 2026-07-07 at commit `4053985` for double-cable
  observer-only, `Naxons=128`, requested `Nx=101`, actual kernel `Nx=89`,
  fp32, different diameters, factorized extracellular input, five repeats, and
  one warmup. Artifacts live under
  `benchmark/results/kaggle/20260707_130525_double_cable_real_stage_profile_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`
  and
  `benchmark/results/kaggle/20260707_130525_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`;
  summary note:
  `docs/architecture/p11b_real_double_cable_stage_profile_2026_07_07.md`.
  CPU remains Thomas-first; GPU is clearly PCR/SoA-first. The active one-step
  proxy is solver-sensitive, but generated membrane work, system assembly,
  factorized RHS drive, and observer writes remain visible when measured as
  separate kernels, so the next low-level pass should inspect solver internals
  and membrane/compiler output before adding a new runtime route.
  Follow-up Kaggle v2 runs at commit `6cdb966` extended this to `Naxons=512`,
  requested `Nx=101`, actual kernel `Nx=89`, for both different-diameter and
  same-diameter cohorts on CPU/GPU. Valid artifact roots are
  `benchmark/results/kaggle/20260707_133000_real_stage_diff_cpu_512_v2/outputs/extracted`,
  `benchmark/results/kaggle/20260707_133000_real_stage_diff_gpu_512_v2/outputs/extracted`,
  `benchmark/results/kaggle/20260707_134000_real_stage_same_cpu_512_v2/outputs/extracted`,
  and
  `benchmark/results/kaggle/20260707_134000_real_stage_same_gpu_512_v2/outputs/extracted`;
  summary note:
  `docs/architecture/p11b_real_double_cable_stage_profile_512_2026_07_07.md`.
  The larger run confirms GPU PCR/SoA as the immediate low-level target:
  fused GPU one-step is about 0.49-0.56 ms and PCR/SoA solve is about
  0.44-0.46 ms, while GPU Thomas is about 2.1-2.2 ms. CPU remains
  Thomas-first, but different-diameter CPU still has visible generated
  membrane and assembly costs; same-diameter coefficient sharing reduces CPU
  one-step from about 4.32 ms to 2.50 ms. Keep high-level grouping/policy work
  for later; next inspect GPU PCR/SoA lowering/layout and CPU membrane compiler
  output before adding a new runtime route.
  First lowering/codegen audit tool added:
  `benchmark/analysis/double_cable_solver_lowering_audit.py` lowers real
  prepared `pcr_soa_vmap`, `pcr_soa_batched`, and the active one-step proxy to
  StableHLO/HLO, optionally compiles optimized HLO, and writes
  `lowering_summary.csv`, `lowering_metrics.json`, raw IR files, and
  `lowering_report.md`. A local CPU smoke passed under
  `benchmark/results/p11b_solver_lowering_smoke_v2` for `Naxons=2`,
  requested `Nx=21`, different diameters, and forced `pcr_soa`. The Kaggle
  runner exposes it as `--campaign double_cable_solver_lowering_audit`.
  Matching Kaggle P100 GPU lowering audits passed on 2026-07-07 at commit
  `e1101ab` for `Naxons=512`, requested `Nx=101`, actual kernel `Nx=89`,
  observer-only, fp32, and both different-diameter and same-diameter cohorts.
  Artifacts live under
  `benchmark/results/kaggle/20260707_141000_solver_lowering_diff_gpu_512/outputs/extracted`
  and
  `benchmark/results/kaggle/20260707_141000_solver_lowering_same_gpu_512/outputs/extracted`;
  summary note:
  `docs/architecture/p11b_double_cable_solver_lowering_audit_2026_07_07.md`.
  Optimized HLO shows `pcr_soa_batched` and `pcr_soa_vmap` are almost
  identical after XLA GPU compilation: gather/select/fusion counts match, no
  transpose remains in the isolated solver HLO, and the batch-native path is
  only slightly smaller by line/broadcast count. Same-diameter coefficient
  sharing does not simplify the isolated solver HLO. The one-step proxy changes
  more, but that comparison also changes membrane backend/generated-model
  status, so treat it as a compiler/backend signal rather than a pure solver
  result. Decision: do not add a new vmap-vs-batched runtime route; next drill
  into current PCR/SoA fusion bodies/memory layout and CPU generated-membrane
  lowering before any solver candidate or policy change.
  Follow-up tooling now adds `benchmark/analysis/hlo_fusion_summary.py` and
  automatic `hlo_fusion_summary.csv`, `hlo_layout_summary.csv`,
  `hlo_fusion_metrics.json`, and `hlo_fusion_report.md` outputs to the
  lowering audit when optimized HLO is available. The audit can also include
  real generated membrane/system stages with `--include-membrane-stages` and
  `--include-system-stages`. A local CPU smoke passed under
  `benchmark/results/p11b_lowering_membrane_smoke` for `Naxons=2`,
  requested `Nx=21`, different diameters, generated membrane stages, system
  assembly, extracellular drive, and observer write. Offline analysis of the
  existing Kaggle GPU HLO roots confirms the dominant solver-level shape:
  each isolated PCR/SoA variant has seven fusions, with repeated
  `loop_select_subtract*` fusions returning 22 `[512,89]` arrays each
  (~3.82 MiB fp32 output per fusion) before the final two-array
  `loop_add_fusion`. Next low-level action: inspect whether those large live
  tuple/fusion outputs can be reduced, recomputed, or staged differently while
  preserving exactness.
  First benchmark-only candidate added:
  `benchmark/analysis/double_cable_solver_candidates.py` provides
  `pcr_soa_symmetric_batched`, which exploits the exact double-cable symmetric
  matrix invariant to carry only one side of the PCR coupling state and rebuild
  the opposite side by shifted transpose. It is wired only into benchmark
  analysis tools, not runtime `auto` policy. Targeted unit tests pass against
  current masked PCR/SoA for batched and shared synthetic symmetric
  coefficients. Local CPU smoke evidence:
  `benchmark/results/p11b_symmetric_pcr_lowering_smoke` lowers the tiny case
  from `23393` to `16751` optimized-HLO lines and from `228` to `186` fusions;
  `benchmark/results/p11b_symmetric_pcr_real_stage_smoke` runs the candidate on
  real prepared double-cable inputs and is directionally faster on the tiny CPU
  case. A tiny real prepared fp32 correctness check gives current-vs-candidate
  max absolute difference `6.7e-4` on a roughly `80 mV` solution range, with
  max relative residual `9.0e-6` for the candidate versus `8.4e-6` for current
  PCR/SoA and `8.4e-7` for Thomas. Next gate: run the same candidate on Kaggle
  P100 for `Naxons=512`,
  requested `Nx=101`, comparing optimized HLO/fusion tuple shape and hot
  block-solve timing against current `pcr_soa_batched`.
  Kaggle P100 gate completed under
  `benchmark/results/kaggle/20260707_140519_symmetric_pcr_lowering_gpu_512/outputs/extracted`
  and
  `benchmark/results/kaggle/20260707_140519_symmetric_pcr_real_gpu_512/outputs/extracted`.
  Result: candidate optimized HLO is smaller (`2267` -> `1855` lines, gathers
  `184` -> `134`, selects `117` -> `87`, largest PCR fusion output `22` arrays
  / `3.82 MiB` -> `14` arrays / `2.43 MiB`, total estimated fusion outputs
  `21.90 MiB` -> `14.25 MiB`), but hot isolated block-solve timing only moves
  `0.415 ms` -> `0.404 ms` (`-2.6%`). Decision: keep the candidate as
  benchmark-only evidence, do not promote it to runtime policy as-is. Next
  low-level direction: keep attacking PCR/fusion/gather/layout costs inside
  solver lowering, and validate any stronger candidate on full real double-cable
  curves before runtime integration.
  Next low-level sweep: benchmark existing PCR/SoA probes
  `pcr_soa_nomask_batched`, `pcr_soa_shift_batched`,
  `pcr_soa_transposed_batched`, `pcr_soa_padded_batched`, and
  `pcr_soa_hybrid_batched` against current `pcr_soa_batched` using the same
  local correctness gate, HLO/fusion summary, real-stage hot timing, and P100
  artifact capture. Promote nothing to runtime policy without a hot-path win and
  a real curve-level validation.
  P100 sweep completed under
  `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_lowering_gpu_512/outputs/extracted`
  and
  `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_real_gpu_512/outputs/extracted`.
  `pcr_soa_shift_batched` is the cleanest compiler diagnostic: optimized HLO
  `2267` -> `1821` lines, gathers `184` -> `0`, selects `117` -> `0`, first-run
  `2426 ms` -> `1636 ms`, but hot block solve is `0.417 ms` -> `0.425 ms`
  (`+1.9%`). `pcr_soa_padded_batched` is runtime-equivalent within noise
  (`0.416 ms`) and shrinks the largest fusion output to `14` arrays / `3.50
  MiB`, but does not provide a robust win. `nomask`, `transposed`, and `hybrid`
  are slower. Decision: promote none as-is; keep `shift`/`padded` as diagnostic
  references and require one-step or curve-level wins before any runtime route.
  Final check before leaving this branch: compare fused `one_step_proxy`
  variants `pcr_soa_batched`, `pcr_soa_shift_batched`, and
  `pcr_soa_padded_batched` on P100. If no clear one-step gain appears, stop
  spending time on these PCR/SoA micro-variants and move to the next low-level
  optimization family.
  Final P100 one-step check completed under
  `benchmark/results/kaggle/20260707_145211_one_step_probe_real_gpu_512/outputs/extracted`
  on commit `c89ff93`: baseline `pcr_soa_batched_real` is `0.472 ms`, `shift`
  is `0.529 ms` (`+12.1%`), and `padded` is `0.485 ms` (`+2.8%`). `shift`
  lowers first-run time (`2342 ms` -> `1954 ms`) but not hot one-step runtime.
  Decision: close this PCR/SoA micro-variant branch for now; move to another
  low-level optimization family.
- [x] Validate the benchmark-only `precomputed_static` double-cable assembly
  candidate on P100 before any runtime change. The profiler now compares
  baseline `system_assembly/real_double_cable` with
  `system_assembly/precomputed_static`, and adds matching
  `one_step_proxy/*_precomputed_static` rows. It precomposes static diagonal
  bases, offsets, and absolute correction/background currents outside the
  measured assembly call, with a numerical equality guard against the baseline
  assembled system. Local smoke passed for a tiny thomas CPU case
  (`benchmark/results/p11b_precomputed_assembly_smoke`) and a tiny PCR/SoA CPU
  case (`benchmark/results/p11b_precomputed_assembly_pcr_smoke`); the PCR/SoA
  smoke did not show a hot one-step win, so this remains a GPU diagnostic
  rather than a runtime direction.
  P100 validation completed under
  `benchmark/results/kaggle/20260707_151450_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-precomputed-assembly-gpu-512/outputs/extracted`
  on commit `38dcfe5`: isolated assembly is slower (`0.299 ms` baseline vs
  `0.339 ms` precomputed), but the fused one-step proxy improves in the same
  run (`0.543 ms` baseline vs `0.500 ms` precomputed, median `0.558 ms` vs
  `0.479 ms`). This is not enough for a runtime policy decision because the
  isolated stage moved the wrong way and the production scan already precomputes
  part of these terms; next step is a HLO/fusion audit of the baseline and
  precomputed one-step bodies on P100.
- [x] Audit HLO/fusion for the baseline and `precomputed_static` one-step
  double-cable bodies. Compare fusion counts, largest fusion outputs, tuple
  pressure, gather/select counts, and layout changes before deciding whether to
  prototype a runtime scan version.
  P100 HLO audit completed under
  `benchmark/results/kaggle/20260707_151930_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-precomputed-lowering-gpu-512/outputs/extracted`
  on commit `38dcfe5`. The compiled one-step HLO changes only modestly:
  optimized-HLO lines `2862` -> `2831`, fusions stay `10`, gathers stay `182`,
  selects stay `125`, and broadcasts `130` -> `129`. The useful signal is that
  the non-PCR assembly-side loop fusion changes from `5` outputs / `0.87 MiB`
  to `4` outputs / `0.70 MiB`; the large PCR loop fusions remain unchanged at
  `22` outputs / `3.82 MiB`. Next prototype, if we pursue it, should be a clean
  runtime scan-body precompute of the remaining diagonal/RHS bases, followed by
  real curve-level validation, not another PCR micro-variant.
- [x] Time-box the runtime scan-body precompute prototype on real curve
  benchmarks. The runtime double-cable PCR/SoA scan now precomputes the static
  diagonal bases and absolute background/correction helpers outside the scan
  step. Local validation passed `git diff --check`, `compileall`, focused and
  full `tests/unit/solvers/test_batch.py`, plus a local double-cable
  recruitment smoke. P100 A/B artifacts:
  `benchmark/results/kaggle/20260707_153410_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-baseline-gpu/outputs/extracted`,
  `benchmark/results/kaggle/20260707_153636_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-prototype-gpu/outputs/extracted`,
  `benchmark/results/kaggle/20260707_154008_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-baseline-gpu-r3/outputs/extracted`,
  and
  `benchmark/results/kaggle/20260707_154232_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-prototype-gpu-r3/outputs/extracted`.
  Results were numerically identical for the recruitment curve. The short
  single-repeat run improved total simulation time by about `5.6%`. The R3 run
  split the picture: cold/warmup simulation worsened by about `9.5%`, but repeat
  simulation improved by about `8.1%` (`runtime.prepare` repeats `-13.2%`,
  `kernel.wait` repeats `-2.9%`, `kernel.dispatch_jax` unchanged). Decision:
  keep this as a bounded runtime cleanup for repeated curve workloads, but stop
  exploring this precompute family for now because it does not move the core
  solver lowering or PCR fusion bottleneck.
- [ ] Run the full `time_chunk_steps` campaign across default, unchunked, 50,
  250, 500, 1000, and adaptive policies for full Vm, probe Vm, and
  observer-only outputs.
  Track peak memory, chunk overhead, cold/warm time, GPU utilization, result
  equivalence, and whether defaults should depend on `nt`, `Naxons`,
  recording mode, or backend.
  CPU recruitment matrix evidence was captured under
  `benchmark/results/p11b_time_chunk_recording_matrix_cpu` for `Naxons=1000`,
  `Nx=101`, `tsim=10 ms`, `dt=0.01 ms`, five amplitudes, different-diameter
  cohorts, and policies `default`, `unchunked`, 50, 250, 500, and 1000. On CPU,
  full Vm and probe Vm favor large/unchunked time chunks: full Vm is fastest at
  explicit 1000 (~19.6 s) then unchunked (~20.0 s), while probe Vm is fastest
  unchunked (~18.0 s) then explicit 1000 (~18.1 s). Small chunks increase
  dispatch overhead for full/probe modes. Observer-only behaves differently:
  default/50/250 are close (~19.2-19.3 s), while 500, 1000, and unchunked are
  slower (~20.6-21.0 s) due to shifted VmRaster combine/finalize
  materialization. This supports backend/recording-specific evidence gathering
  before any default change.
  Matching Kaggle P100-image CPU/GPU recruitment matrix evidence was captured
  on 2026-07-06 at commit `091992b` with the same bounded workload and all
  18 cases passing. CPU-path artifacts live under
  `benchmark/results/kaggle/20260706_182247_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  and GPU-path artifacts under
  `benchmark/results/kaggle/20260706_182247_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`.
  The Kaggle CPU path keeps the same qualitative shape as the local CPU run but
  is slower overall: full Vm is best with explicit 1000 (~35.4 s), probe Vm is
  best with default (~34.0 s, close to unchunked/1000), and observer-only is
  best with default (~33.2 s). Full/probe CPU cases are dominated by
  `kernel.wait` (~26-27 s), while observer-only time moves between dispatch,
  combine, and finalize depending on chunk policy. The Kaggle GPU path shifts
  the best policy toward large/unchunked chunks: full Vm is best with explicit
  1000 (~14.3 s), probe Vm is essentially tied across default/1000/unchunked
  (~12.7-12.8 s), and observer-only is best unchunked (~12.4 s) with explicit
  1000 close behind (~12.7 s). Default/50-step observer-only on GPU remains
  much slower (~17.5 s), so current small-chunk behavior is a GPU bottleneck.
  Result-boundary traces now show full/probe GPU materialization/assembly costs
  around 1.2-2.3 s. Keep this as bottleneck cartography for now: at this
  point, threshold-curve evidence, adaptive policy candidates, and
  repeated/realistic confirmation were still required before any default
  change.
  Threshold-curve matrix evidence was then captured on 2026-07-06 at commit
  `6c92042` after fixing `benchmark/campaigns/time_chunk_sweep.py` to map the
  campaign-scale `--amplitude-count` knob to `--max-iterations` for
  `threshold_curves`. All 18 Kaggle CPU-path cases and all 18 Kaggle GPU-path
  cases passed. CPU-path artifacts live under
  `benchmark/results/kaggle/20260706_185153_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`,
  and GPU-path artifacts under
  `benchmark/results/kaggle/20260706_185207_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`.
  On Kaggle CPU threshold curves, full Vm and probe Vm again favor large
  chunks: full Vm is best with explicit 1000 (~16.3 s) and probe Vm is best
  with explicit 1000 (~15.7 s). Observer-only is nearly flat across default,
  unchunked, and explicit 1000 (~14.4-14.5 s), but its time moves between
  dispatch, combine, and finalize depending on policy. On Kaggle GPU threshold
  curves, unchunked is best across all recording modes: full Vm ~10.0 s,
  probe Vm ~9.0 s, and observer-only ~8.9 s, with explicit 1000 close behind
  for probe/observer and 50-step chunks clearly slower. This confirms the
  broad GPU bottleneck map from recruitment: small chunks cost too much on GPU,
  large/unchunked policies are consistently competitive, and full/probe GPU
  result materialization/assembly remains visible (~0.5-1.1 s depending on
  recording). Remaining before changing defaults: adaptive policy candidates,
  repeated runs, and a bounded-realistic confirmation pass.
  Cross-run bottleneck plots were generated with
  `benchmark/analysis/time_chunk_matrix_report.py` under
  `benchmark/results/p11b_time_chunk_matrix_report_20260706_stage_groups`.
  The report combines threshold/recruitment CPU/GPU matrices into heatmaps,
  best-policy stage-group breakdowns, kernel/result sub-stage breakdowns,
  CPU/GPU speedups, CPU RSS memory plots, GPU JAX device-memory plots, and
  GPU `nvidia-smi` context plots. Use this visual report to pick the next
  low-level optimization target before writing adaptive policy code.
  A no-policy-decision interpretation note lives at
  `benchmark/results/p11b_time_chunk_matrix_report_20260706_stage_groups/p11b_low_level_bottleneck_notes.md`;
  it frames the next step as deeper low-level tracing and optimization, not
  adaptive policy selection.
- [ ] Keep recruitment amplitude micro-batching as a later high-level
  optimization axis, not the immediate P11B target. When low-level
  observer/kernel bottlenecks are understood, compare candidate
  `amplitude_batch_size` values such as 1, 2, 4, and 8 against peak memory,
  footprint duplication, cold/warm time, observer-only result assembly, and
  scientific equivalence before changing defaults.
- [ ] Carry over P10 backend-neutral optimizer closeout under benchmark
  control: common subexpression elimination, unused diagnostic pruning, stable
  optimized-graph hashing, explainable before/after summaries, and deciding
  which report-time target identity fields become real cache-key inputs.
- [ ] Carry over P10 JAX-specific generated fusion closeout under benchmark
  control: generated conductance terms, state prepare/finalize updates,
  diagnostics, requested-observable pruning, composite generated programs or a
  stricter fail-fast boundary, avoiding transport of unrequested intermediate
  arrays, and direct solver-kernel fusion beyond the P7 class subset.
- [ ] Remove or reject dense internal fallback paths only after factorized or
  compact equivalents have equivalence tests and realistic benchmark evidence.
- [ ] GPU dispatch scheduling: memory-aware bucket/coalesce first, optional
  async enqueue second, only after memory budgets and group-route inspection
  exist. See `ideas/axonscope_dispatch_scheduling_gpu_note.md`.
- [ ] Double-cable rank-K compact `Vext`: future optimization/validation slice.
  Current double-cable keeps dense materialization for unsupported compact
  cases. Only broaden compact forcing after equivalence tests against dense
  results and benchmark evidence for memory/time benefits.
- [ ] Improve GPU solver: see
  `ideas/axonscope_gpu_tridiagonal_solver_literature_synthesis.md`,
  `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`, and
  `ideas/axonscope_single_double_cable_gpu_solver_options_with_precompute.md`.
  Before implementation, do a critical reading pass over these documents:
  identify what already exists in the current code, what is relevant versus
  not relevant for AxonScope's present solver boundary, and which ideas deserve
  staged implementation. Also audit the membrane/model compiler and generated
  JAX lowering path for low-level optimization opportunities before blaming
  the runtime/solver layer for generated-code costs. Implement candidates
  incrementally, each gated by correctness checks plus small benchmarks first,
  then larger realistic benchmarks before keeping or promoting the change.
  Initial critical-reading/audit pass is recorded in
  `docs/architecture/p11b_double_cable_solver_compiler_audit_2026_07_07.md`.
  The first candidate gate is not a policy change: extend benchmark
  cartography to real generated double-cable membrane stages, then consider a
  benchmark-only batched Thomas-family GPU prototype against current GPU
  PCR-SoA if the profile still supports that direction.
  First gate implemented in `benchmark/analysis/double_cable_real_stage_profile.py`:
  the real MRG/double-cable profiler now reports hot-step group shares,
  solver share, membrane backend metadata, and gated/leak-stack diagnostic
  rows (`*_gated_only`, `*_mask_mix`). The lowering audit now includes those
  membrane sub-stages when `--include-membrane-stages` is enabled, and
  `hlo_fusion_summary.py` recognizes the new stage names. Local CPU smokes:
  `benchmark/results/p11b_mrg_hot_step_profile_smoke_diff` and
  `benchmark/results/p11b_mrg_hot_step_lowering_smoke`. On the small
  different-diameter MRG smoke (`Naxons=8`, actual `Nx=56`), the isolated
  PCR/SoA block solve is about `53%` of the one-step proxy, while full
  membrane gate+conductance work is about `15%`.
  Kaggle CPU/GPU hot-step and lowering runs then passed on 2026-07-07 at
  commit `ffd54fe` for MRG different-diameter double-cable, `Naxons=512`,
  requested `Nx=101`, actual kernel `Nx=89`, observer-only, fp32, and forced
  `pcr_soa`. Artifact roots:
  `benchmark/results/kaggle/20260707_164147_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-mrg-hot-step-gpu-512/outputs/extracted`,
  `benchmark/results/kaggle/20260707_164203_double_cable_real_stage_profile_quick_cpu_NvidiaTeslaP100_axonscope-p11b-mrg-hot-step-cpu-512/outputs/extracted`,
  `benchmark/results/kaggle/20260707_164448_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-mrg-lowering-gpu-512-r2/outputs/extracted`,
  and
  `benchmark/results/kaggle/20260707_164501_double_cable_solver_lowering_audit_quick_cpu_NvidiaTeslaP100_axonscope-p11b-mrg-lowering-cpu-512-r2/outputs/extracted`.
  GPU one-step and isolated `pcr_soa_batched` are both about `0.499 ms`; the
  isolated generated membrane stages are visible but not additive with the
  fused one-step. GPU lowering points next at PCR/SoA fusion/memory traffic
  (`2267` optimized-HLO lines, `7` fusions, no transpose). Never add
  runtime branches for a specific membrane model family; MRG is a realistic
  validation benchmark for generic solver/runtime/compiler optimizations, not
  a specialization target. Forced-PCR CPU is a bad target (`~208.7 ms`,
  `99.7%` solver, `32122` optimized-HLO lines, `2576` transposes); CPU work
  should remain Thomas-first unless a dedicated CPU benchmark says otherwise.
  A benchmark-only `one_step_without_solve` probe was then added in commit
  `3c3bdb0`: it fuses gate update, conductance, extracellular RHS, and system
  assembly, then materializes assembled coefficients/RHS plus updated gates to
  avoid dead-code elimination. Local smoke artifacts:
  `benchmark/results/p11b_mrg_no_solve_profile_smoke` and
  `benchmark/results/p11b_mrg_no_solve_lowering_smoke`. Kaggle P100 artifacts
  for the same `Naxons=512`, actual `Nx=89` MRG case:
  `benchmark/results/kaggle/20260707_172311_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-no-solve-hot-step-gpu-512/outputs/extracted`
  and
  `benchmark/results/kaggle/20260707_172327_double_cable_real_stage_profile_quick_cpu_NvidiaTeslaP100_axonscope-p11b-no-solve-hot-step-cpu-512/outputs/extracted`.
  CPU forced-PCR is clearly solver-dominated (`one_step_without_solve`
  `~3.20 ms`, fused one-step `~175.35 ms`, isolated PCR/SoA `~170.07 ms`).
  GPU no-solve materialized is `~0.418 ms` versus fused one-step `~0.488 ms`
  and isolated PCR/SoA `~0.466 ms`, but it writes about `2.7 MiB` of assembled
  outputs; treat it as a materialization/traffic probe, not a subtractive
  decomposition. Next GPU evidence should inspect PCR/SoA fusion bodies and
  memory traffic, and optionally add a non-materializing/reduced-output
  no-solve diagnostic before changing solver code.
  The HLO fusion analyzer now reports benchmark-only per-fusion input/output
  byte estimates. Offline analysis of the existing P100 MRG lowering artifact
  under `benchmark/results/p11b_gpu_hlo_fusion_io_audit` confirms the GPU focus:
  repeated PCR/SoA `loop_select_subtract*` fusions each carry about `3.82 MiB`
  of HLO-shaped inputs and `3.82 MiB` of tuple outputs (`~7.65 MiB` estimated
  I/O). Next action should stay GPU solver-local: reduce PCR stage live
  state/tuple outputs or change staging, with no membrane-model-specific
  runtime path.
  GPU-only critical review of the three older idea documents is recorded in
  `docs/architecture/p11b_gpu_solver_ideas_critical_review_2026_07_07.md`.
  Decision from that pass: do not reopen a broad solver-zoo sweep, do not
  promote Thomas/PTA/custom-kernel routes yet, and do not add public solver
  policy. The next benchmark slice should audit PCR/SoA stage state per stride
  and prototype one benchmark-only state/staging variant that tries to reduce
  the dominant fusion tuple/I/O pressure, then gate it on local correctness,
  P100 lowering, P100 hot-step, and only then realistic curve behavior if it
  actually wins.
  Internal solver structure now starts to follow that boundary:
  `DoubleCableLinearSystemStaticTerms` and `DoubleCableLinearSystem` name the
  backend-owned static terms and assembled SoA system before solve, and the two
  batch-native PCR/SoA runtime paths use this `prepare -> assemble -> solve`
  shape. The first stage-state audit tool,
  `benchmark/analysis/pcr_soa_stage_state_audit.py`, produced
  `benchmark/results/p11b_gpu_pcr_soa_stage_state_audit` from the existing P100
  HLO artifact. For `B=512`, actual `Nx=89`, fp32, the algorithmic state is 14
  arrays (`~2.43 MiB`), while HLO commonly exposes 22 fusion outputs and up to
  `~7.65 MiB` estimated fusion I/O. The tool now supports explicit variants,
  and `benchmark/results/p11b_gpu_pcr_soa_stage_state_audit_symmetric` maps the
  already-tested benchmark-only `pcr_soa_symmetric_batched` route: state drops
  to 10 arrays (`~1.74 MiB`) and largest available HLO output estimate drops to
  14 arrays (`~2.43 MiB`), but previous P100 hot solve timing only moved from
  `0.415 ms` to `0.404 ms`. Conclusion: the symmetry/state-reduction probe is
  useful negative evidence, not a runtime route. Next action: inspect or
  prototype a low-level candidate that changes measured hot-path cost, not only
  HLO tuple size.
  Rejected low-level arithmetic candidate: rewriting the generic 2x2 inverse
  helpers to compute one reciprocal per determinant reduced HLO `count_divide`
  on P100 (`60` to `15`) but regressed hot solve timing. See
  `docs/architecture/p11b_reciprocal_inverse_gate_2026_07_07.md`. On the
  existing `Naxons=512`, requested `Nx=101`, actual `Nx=89`, observer-only MRG
  shape, isolated `pcr_soa_batched` moved from `0.417 ms` to `0.474 ms`; the
  code was reverted. Do not pursue isolated instruction-count reductions
  without a measured hot-path win. Gate artifacts for commit `21ffaef`:
  `benchmark/results/kaggle/20260707_183755_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-recip-inv-lowering-gpu-512`
  and
  `benchmark/results/kaggle/20260707_183812_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-recip-inv-real-gpu-512`.
  Next action: change class of optimization rather than keep tuning PCR/SoA
  instruction shape. Focus on a measured structural runtime cost: launch/stage
  count, memory traffic across stages, or an actually different GPU solve
  structure from the solver-roadmap documents.
  Rejected structural GPU candidate: benchmark-only `thomas_batched_scan`, the
  existing batch-native block-Thomas scan
  (`solve_block_tridiagonal_2x2_scalar_batched`), was gated against
  `thomas_vmap` and `pcr_soa_batched` on real double-cable inputs. It remains
  useful for CPU/compiler diagnostics, but not as a GPU runtime policy. See
  `docs/architecture/p11b_thomas_batched_scan_gate_2026_07_07.md`. On Kaggle
  P100, `Naxons=512`, requested `Nx=101`, actual `Nx=89`, observer-only MRG
  shape, `thomas_batched_scan` had compact optimized HLO (`795` lines versus
  `2267` for PCR-SoA) but regressed hot time badly: isolated solve was
  `3.698 ms` versus `0.453 ms` for `pcr_soa_batched`, and the fused one-step
  proxy was `2.172-2.176 ms` versus `0.535-0.564 ms` for PCR-SoA. Artifact
  roots:
  `benchmark/results/kaggle/20260707_190539_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-thomas-scan-lower-gpu/outputs/extracted`
  and
  `benchmark/results/kaggle/20260707_190552_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-thomas-scan-real-gpu/outputs/extracted`.
  Next action: keep GPU `auto` on PCR-SoA and move to measured hot-path cost,
  not HLO size alone: PCR-SoA live state/memory traffic, launch/stage boundaries
  around the one-step proxy, or a genuinely GPU-native tiled/custom/Pallas-style
  solver candidate only after a narrow benchmark-backed hypothesis.
- [ ] Re-run NRV validation only for numerical behavior changes, but always
  pair optimization claims with fresh hotpath or realistic benchmark evidence.

### P11C - Large-Population Exact GPU Solver

Goal: implement a benchmark-gated exact double-cable GPU execution family for
large axon populations. This is not a selection between existing solver options.
If the benchmarks show a meaningful large-population simulation win, promote the
route into the backend. If not, close P11C with evidence and keep the current
JAX GPU solver path.

Design reference:
`docs/architecture/p11c_large_population_gpu_solver_plan_2026_07_07.md`.
PTA/block-Thomas GPU gate:
`docs/architecture/p11c_pta_block_thomas_gpu_gate_2026_07_08.md`.

- [x] P11C non-goals and boundaries:
  no public solver option, no `auto` policy change, no membrane-model-specific
  runtime path, no split/fixed-K approximate route, and no Triton/Pallas/CUDA
  FFI/custom-kernel work in this slice. Keep custom kernels as the later path
  only if P11C proves the layout direction is worth deeper GPU integration.
- [x] Define the backend-private candidate:
  `large_population_exact_double_cable_jax`. It must own a large-population GPU
  execution shape rather than wrapping an existing solver name.
- [ ] Extend the backend-private candidate into a measured real-stage route:
  prepare layout, packed state, static terms, factorized forcing, dynamic
  membrane/RHS assembly, exact solve, and compact recording as one measured
  backend-owned path.
- [x] Define and implement the internal layout vocabulary for benchmarks:
  `BX`, `XB`, and `TILED`. The target production shape is
  `[tile, Nx_pad, BLOCK_B]`; the JAX prototype may use `XB` or a tiled
  representation only if it avoids extra packing/transposes in the hot path.
- [x] Define broader `Nx_pad` candidates and neutral padding:
  benchmark `32, 48, 64, 80, 96, 128, 160, 192, 256`; use electrically neutral
  double-cable padding rows (`D=I_2`, `L/U=0`, `rhs=[0,0]`); keep the final
  bucket set data-driven rather than copied from old idea docs.
- [x] Define `BLOCK_B` candidates by bucket:
  for `Nx_pad <= 64`, test `64/128/256`; for `Nx_pad <= 128`, test
  `32/64/128`; for larger buckets, test `16/32/64`. Record occupancy-oriented
  metrics indirectly through throughput, memory, compile time, and hot timing
  until a custom-kernel path exists.
- [x] Build the synthetic P11C benchmark harness before runtime integration:
  large-pop solver/layout benchmark with raw CSVs, metadata, git state, hardware
  metadata, memory, compile/first-run time, hot time, node-solves/s,
  axon-steps/s, output bytes, and a Markdown decision report.
- [x] Add the P11C-PTA critical design gate:
  document why the literature-backed `PTA_BLOCK_THOMAS_2X2_TILED` hypothesis is
  not answered by the first tiled/padded PCR-SoA candidate; keep PfSolve/NVIDIA
  lessons as a custom-kernel-ready design reference without adding Triton,
  Pallas, FFI, public solver options, or runtime policy changes in this slice.
- [x] Add `thomas_batched_scan` to the P11C synthetic profiler as the JAX
  feasibility baseline for Thomas/PTA at large `Naxons`.
- [x] Run the P11C-PTA JAX feasibility matrix on GPU:
  compare `current_pcr_soa`, `thomas_batched_scan`, and
  `large_population_exact_double_cable_jax` at `Naxons=1024/4096/8192/16384`,
  `Nx=47/89/129`, shared and batched coefficients, fp32 first. Result on P100
  at commit `b4f4618`: `thomas_batched_scan` is slower at `Naxons=1024`, close
  but usually slower at `4096`, and fastest for all tested shapes at
  `8192/16384`. Artifact:
  `benchmark/results/kaggle/20260708_181129_large_population_double_cable_solver_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-pta-jax-gpu-b4f4618/extracted`.
  Treat this only as a synthetic solver-only gate, not runtime policy.
- [x] Build the real-stage P11C benchmark harness before runtime integration:
  large-pop real-stage benchmark with layout/packing, membrane/RHS assembly,
  exact solve, compact recording, and result-boundary costs.
- [x] P11C-A gate: local smoke and correctness.
  Run tiny synthetic cases against current exact solvers and Thomas reference;
  validate shape/padding behavior, layout packing/unpacking, and no public API
  surface changes.
- [ ] P11C-B gate: JAX-only benchmark-private prototype.
  Start with an exact PCR-SoA-like body in the large-pop layout because it
  exposes parallelism over both `B` and `Nx`. It must differ from
  `pcr_soa_batched` by owning explicit `Nx_pad`, `B` tiling/layout, persistent
  static-term preparation, and direct assembly into the chosen layout.
- [ ] P11C-C gate: synthetic GPU large-population matrix.
  Test `B/Naxons=1024,4096,8192,16384` plus optional `32768` if memory allows,
  `Nx_true` values that cover all candidate buckets, shared and batched
  coefficients, fp32 first, and fp64 spot checks. Compare against the current
  exact JAX route only as baseline evidence, not as the objective.
  CPU synthetic cross-check on Kaggle P100 machine at commit `ed0ccff` is now
  available:
  `benchmark/results/kaggle/20260708_174928_large_population_double_cable_solver_profile_quick_cpu_NvidiaTeslaP100_axonscope-p11c-large-pop-cpu-ed0ccff/outputs/benchmark/results/large_population_double_cable_solver_profile_quick_cpu_20260708_174928`.
  It shows the tiled/padded candidate only wins modestly for `Nx=47`
  (`~3.5%-9.8%`) and loses or ties for `Nx=89/129`, especially because
  `Nx=129 -> Nx_pad=160` is costly. Do not move CPU execution toward P11C;
  keep P11C as a GPU large-population track.
- [ ] P11C-D gate: real-stage large-population matrix.
  Use real double-cable MRG-like prepared inputs with observer-only and probe
  outputs first. Target `Naxons=4096/8192`, optional `16384` if memory allows.
  Measure membrane, assembly, solve, one-step proxy, compact recording,
  layout/packing, and result-boundary costs. Proceed to reduced
  recruitment/threshold curves only if this gate wins.
  First GPU real-stage result on Kaggle P100 at commit `b233fc7`,
  observer-only fp32, target `Nx=101` / actual `Nx=89`, different diameters,
  `Naxons=8192`: `thomas_batched_scan` keeps the large-population signal with
  `2.669 ms` block solve versus `3.796 ms` for `pcr_soa_batched`
  (`1.42x` speedup), and `3.117 ms` one-step proxy versus `4.385 ms`
  (`1.41x` speedup) on the precomputed-static real-stage proxy. Artifact:
  `benchmark/results/kaggle/20260708_182559_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-real-pta-gpu-n8192-b233fc7/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_182559`.
  Follow-up `Naxons=16384` on the same P100 gate confirms and strengthens the
  signal: `thomas_batched_scan` is `3.287 ms` block solve versus `7.223 ms`
  for `pcr_soa_batched` (`2.20x` speedup), and `4.104 ms` one-step proxy
  versus `8.241 ms` (`2.01x` speedup) on the precomputed-static real-stage
  proxy. Artifact:
  `benchmark/results/kaggle/20260708_183319_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-real-pta-gpu-16k-b233fc7-r2/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_183319`.
  Next decide whether to implement the backend-private large-population route
  behind benchmark gates, without adding a public solver option or broad runtime
  policy.
- [x] Add the first paper-inspired jax-triton solver candidate:
  `jax_triton_tiled_thomas` uses `jax_triton.triton_call`, an internal
  `[Nx, B]` tile/lane layout, one Triton program per axon tile, and coalesced
  same-compartment lane loads. This is closer to the GPU-tridiagonal papers
  than the pure JAX scan, but still not a full PfSolve-equivalent
  shared-memory/scratch implementation. Keep it benchmark-only.
- [x] Run the jax-triton P11C gate on Kaggle P100:
  compare `pcr_soa_batched`, `thomas_batched_scan`, and
  `jax_triton_tiled_thomas` at least on the real-stage `Naxons=8192/16384`,
  fp32, observer-only, target `Nx=101` / actual `Nx=89`. Install the optional
  dependency with `--pip-package jax-triton`; do not promote anything until the
  artifact shows correctness and hot-step timing.
  Result at commit `f915367`: warm Triton block solve is `0.678 ms` at
  `Naxons=8192` versus `3.766 ms` PCR and `2.748 ms` JAX scan, then
  `0.878 ms` at `Naxons=16384` versus `7.148 ms` PCR and `3.186 ms` JAX scan.
  Warm one-step precomputed proxy is `1.519 ms` at `8192` and `2.421 ms` at
  `16384`, about `2.89x` and `3.35x` faster than PCR respectively. Artifacts:
  `benchmark/results/kaggle/20260708_185401_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-thomas-gpu-8k-f915367/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_185403`
  and
  `benchmark/results/kaggle/20260708_185428_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-thomas-gpu-16k-f915367/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_185428`.
  Caveat: first standalone Triton block-solve compile/run is about
  `40-41 min` on Kaggle P100, so cold compilation/cache behavior must be
  controlled before any backend integration.
- [x] Add a numerical P11C validation mode before interpreting the jax-triton
  timing as physically coherent: `benchmark/analysis/double_cable_real_stage_profile.py`
  now supports `--validate-solvers`, writes `real_stage_validation.csv`, compares
  selected block-solve and one-step proxy outputs against a non-Triton reference,
  records max absolute/relative diffs including `Vm = Vi - Ve`, records
  block-solve residual norms, and fails the script if validation fails. The
  validation deliberately runs after timing measurements so it does not hide the
  first-run/cold compilation cost.
- [x] Run a small P11C jax-triton numerical gate on Kaggle P100 before the
  cold-start audit: include `pcr_soa_batched`, `thomas_batched_scan`, and
  `jax_triton_tiled_thomas` for block solve and one-step proxy, with
  `--validate-solvers`, `--pip-package jax-triton`, fp32, observer-only, target
  `Nx=101` / actual `Nx=89`, and a modest `Naxons` so the first Triton compile
  is the main cost. Only after `real_stage_validation.csv` passes should P11C
  treat the warm jax-triton timings as numerically acceptable.
  Result at commit `d9aa339`, Kaggle P100, `Naxons=1024`: validation passed
  `9/9` rows. `jax_triton_tiled_thomas` versus `thomas_batched_scan` has max
  absolute block-solve diff `0.00352 mV`, max absolute `Vm` diff `0.00335 mV`,
  and max residual norm `8.28e-7`. Warm block solve is not yet a win at this
  small population (`0.810 ms` Triton versus `0.710 ms` PCR), but one-step
  proxy is slightly faster (`0.694 ms` Triton versus `0.746 ms` PCR). Artifact:
  `benchmark/results/kaggle/20260708_195453_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-validation-gpu-d9aa339/outputs/extracted`.
  Cold-start is confirmed even at `Naxons=1024`: first standalone Triton
  block-solve call is `2,315,522 ms`, about `38.6 min`.
- [x] Audit the jax-triton cold-start path after the numerical gate passes:
  isolate install time, import time, JAX tracing/lowering time, Triton kernel
  compilation time, cache behavior inside one process, and whether a persistent
  cache can make the first usable simulation acceptable. Do not promote the
  route while the first standalone Triton block solve costs about `40-41 min`.
  First audit implemented in `benchmark/analysis/jax_triton_cold_start_audit.py`
  and run on Kaggle P100 at commit `7bb5250`, `Naxons=1024`, `Nx=89`,
  `block_b=128`, explicit JAX/Triton cache directories, cleared at startup.
  Artifact:
  `benchmark/results/kaggle/20260708_212036_jax_triton_cold_start_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-cold-start-gpu-7bb5250/outputs/extracted`.
  Result: the cold cost is in JAX/jax-triton lowering, not explicit compile or
  execution. `lower` is `2,340,136 ms` (`39.0 min`), `compile` is `154 ms`,
  `compiled_first_call` is `1,341 ms`, and `compiled_call_2` is `3.84 ms`.
  JAX persistent cache gains files after compile, but the explicit Triton cache
  directory stays empty in this run. Next inspect whether the `tl.static_range`
  Thomas forward/backward loops cause unrolled lowering time, then test smaller
  `Nx` and a non-unrolled/looped kernel variant before doing any integration.
  Micro scaling checks on the same P100 stack, with `B=128`, `block_b=64`, and
  no cache snapshots to protect Kaggle quota:
  `Nx=16` lower `15.8 s`, compile `119 ms`, first call `246 ms`; artifact
  `benchmark/results/kaggle/20260708_220512_jax_triton_cold_start_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-cold-nx16-b128-f93fda4/outputs/extracted`.
  `Nx=32` lower `118.1 s`, compile `152 ms`, first call `468 ms`; artifact
  `benchmark/results/kaggle/20260708_220802_jax_triton_cold_start_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-cold-nx32-b128-f93fda4/outputs/extracted`.
  Stop the static-range Kaggle matrix here for quota reasons: the lowering cost
  scales much faster than linearly with `Nx`, consistent with static loop
  unrolling.
  Follow-up at commit `fe899c4` added a benchmark-only `tl.range` looped
  jax-triton candidate and ran one tiny Kaggle P100 gate at `Nx=32`, `B=128`,
  `block_b=64`: `lower` fell from `118.1 s` to `3.67 s`, `compile` was
  `125 ms`, and first compiled call was `158 ms`. Artifact:
  `benchmark/results/kaggle/20260708_221928_jax_triton_cold_start_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-loop-cold-nx32-b128-fe899c4/outputs/extracted`.
  Interpretation: the cold cliff is caused by the static unrolled kernel shape,
  not by all jax-triton usage.
- [x] Validate the looped jax-triton candidate before any integration:
  run one small Kaggle P100 gate at the real `Nx=89` shape and then one small
  real-stage numerical/timing gate if `Nx=89` cold-start stays acceptable.
  Keep this benchmark-only and do not add public solver options or runtime
  policy changes.
  Cold-start gate at commit `8146a72`, `Nx=89`, `B=128`, `block_b=64`:
  `lower` `3.72 s`, `compile` `138 ms`, first compiled call `154 ms`,
  same-process hot call `3.63 ms`. Artifact:
  `benchmark/results/kaggle/20260708_222735_jax_triton_cold_start_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-loop-cold-nx89-b128-8146a72/outputs/extracted`.
  Small real-stage gate at commit `44534e2`, `Naxons=1024`, target `Nx=101` /
  actual `Nx=89`, fp32, observer-only: validation passed `9/9`; looped Triton
  block solve is `0.483 ms` hot versus `0.643 ms` PCR and `2.199 ms` JAX scan,
  with first standalone block solve `4109 ms`. One-step precomputed-static is
  `0.541 ms` hot versus `0.758 ms` PCR and `2.445 ms` JAX scan. Artifact:
  `benchmark/results/kaggle/20260708_223254_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-loop-real-44534e2/outputs/extracted`.
- [x] Run the looped jax-triton large-population scale gate:
  test `Naxons=8192/16384`, actual `Nx=89`, fp32, observer-only, comparing
  `pcr_soa_batched`, `thomas_batched_scan`, and
  `jax_triton_tiled_thomas_loop`. Keep it benchmark-only; promote only if the
  large-population real-stage win survives with acceptable first-run cost and
  validation.
  Result at commit `c21a610`, Kaggle P100, target `Nx=101` / actual `Nx=89`,
  different diameters, fp32, observer-only: validation passed `9/9` rows for
  both `8192` and `16384`. At `Naxons=8192`, looped Triton block solve is
  `0.619 ms` hot versus `3.791 ms` PCR and `2.715 ms` JAX scan; one-step
  precomputed-static is `1.504 ms` versus `4.485 ms` PCR and `3.304 ms` JAX
  scan. At `Naxons=16384`, looped Triton block solve is `0.992 ms` hot versus
  `7.172 ms` PCR and `3.310 ms` JAX scan; one-step precomputed-static is
  `2.360 ms` versus `8.333 ms` PCR and `4.218 ms` JAX scan. First standalone
  looped block solve is now seconds, not minutes: `4319 ms` at `8192` and
  `5881 ms` at `16384`. Artifacts:
  `benchmark/results/kaggle/20260708_224000_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-loop-real-8k-c21a610/outputs/extracted`
  and
  `benchmark/results/kaggle/20260708_224256_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-loop-real-16k-c21a610/outputs/extracted`.
- [x] Run the looped jax-triton `block_b` and direct-XB layout gate before
  runtime integration:
  commit `66d9b6a` adds benchmark-only `--jax-triton-block-b` support in
  `benchmark/analysis/double_cable_real_stage_profile.py`, extends the
  synthetic large-population profiler to sweep `block_b`, and compares the
  current batch-first wrapper against a direct `[Nx, B]` input route. Kaggle
  P100 synthetic solve-only gate, actual `Nx=89`, fp32, batched coefficients:
  at `Naxons=8192`, PCR is `3.907 ms`, best wrapper Triton is `0.638 ms`
  (`block_b=64`), and direct-XB Triton is `0.469 ms` (`block_b=32`);
  at `Naxons=16384`, PCR is `7.277 ms`, best wrapper Triton is `0.871 ms`
  (`block_b=128`), and direct-XB Triton is `0.634 ms` (`block_b=32`).
  Artifact:
  `benchmark/results/kaggle/20260708_230508_large_population_double_cable_solver_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-blockb-large-gpu-66d9b6a/outputs/extracted`.
  Real-stage gates confirm the signal with validation passing: `19/19` rows at
  `Naxons=8192` and `15/15` rows at `Naxons=16384`. At `8192`, PCR block solve
  is `3.872 ms`, best wrapper Triton is `0.636 ms`, and direct-XB is `0.467 ms`
  (`8.3x` versus PCR, `1.36x` versus wrapper). At `16384`, PCR block solve is
  `7.129 ms`, best wrapper Triton is `0.868 ms`, and direct-XB is `0.628 ms`
  (`11.3x` versus PCR, `1.38x` versus wrapper). Precomputed XB assembly is also
  slightly faster than batch-first (`0.427 ms` versus `0.449 ms` at `8192`,
  `0.541 ms` versus `0.573 ms` at `16384`). Artifacts:
  `benchmark/results/kaggle/20260708_230811_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-blockb-real-gpu-66d9b6a/outputs/extracted`
  and
  `benchmark/results/kaggle/20260708_231205_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axs-p11c-blockb-real16k-gpu-66d9b6a/outputs/extracted`.
- [x] P11C decision rule:
  promote only if the candidate improves large-population real-stage or curve
  throughput with acceptable memory and correctness, and without shifting cost
  to host packing, compile, or result assembly. Reject if the win exists only in
  solver-only synthetic tests or HLO counters. If rejected, stop P11C and keep
  the current GPU solver path rather than adding more JAX micro-variants.
  Decision after the `block_b`/direct-XB gates: proceed to a backend-private
  integration of the looped jax-triton exact double-cable solver for large
  GPU batches, keeping the public API unchanged. Public routes keep the current
  JAX/PCR behavior unless a future benchmark-backed policy explicitly changes
  them. The integration should assemble directly into the solver's XB layout;
  do not integrate the batch-first wrapper as the production hot path.
- [x] P11C-E backend-private integration:
  wire the looped jax-triton XB double-cable route behind an explicit benchmark
  override only. Do not add a public solver option and do not add automatic
  internal policy selection in this slice. The explicit benchmark override
  should fail loudly when Triton/GPU constraints are not met, so benchmark
  artifacts cannot silently measure the fallback route under a Triton label.
- [ ] P11C-F complete policy benchmark:
  after integration, run a full CPU/GPU benchmark matrix with realistic curve
  workloads, recording modes, `Naxons`, `Nx`, dtype, cold/warm cache state,
  timing traces, memory traces, correctness checks, and git metadata before
  deciding any runtime/default policy.
  Policy-campaign tooling now lives in
  `benchmark/campaigns/double_cable_solver_policy.py`. It compares typed public
  double-cable solver policies through `threshold_curves` and
  `recruitment_curves`, writes a manifest, summary CSV, and markdown report,
  and is wired through the Kaggle runner as
  `--campaign double_cable_solver_policy`. Local CPU smoke passed under
  `benchmark/results/p11c_solver_policy_cpu_smoke`; this validates the runner,
  not the final policy.
  CPU Kaggle policy gate at commit `c75dce6` passed `32/32` cases for
  `auto`/`thomas`; every row resolved to effective `thomas`. CPU double-cable
  policy decision for the cleanup: keep only Thomas on CPU, with
  `auto -> thomas` and explicit
  `axs.runtime.jax.cpu.DoubleCableSolver.thomas()`. Do not keep, revive, or
  optimize CPU PCR/PCR-SoA/Triton/tiled routes in the active runtime; non-Thomas
  CPU double-cable routes are unsupported and have crashed in policy testing.
  Artifact:
  `benchmark/results/kaggle/20260710_091924_double_cable_solver_policy_quick_cpu_NvidiaTeslaP100_axonscope-p11c-policy-cpu-c75dce6/outputs/extracted`.
  Full GPU policy gate at commit `c75dce6` completed on Kaggle P100 in about
  `4h50` and passed `192/192` cases. It covered `threshold_curves` and
  `recruitment_curves`, `observer_only` and `probe_vm`, `Naxons=64/1024/4096/8192`,
  `Nx=89/129`, fp32, and `auto`, `thomas`, `pcr`, `pcr_soa`,
  `tiled_thomas(block_b=32)`, and `tiled_thomas(block_b=64)`. Winner counts by
  warm mean over the 32 shape/workflow groups: `tiled_thomas_b32` wins `12`,
  `tiled_thomas_b64` wins `9`, `auto` wins `7`, and explicit `pcr_soa` wins
  `4`; explicit `pcr` and explicit `thomas` win no groups. Interpretation:
  Triton/tiled Thomas is the clear candidate for large GPU populations
  (`Naxons >= 4096`, all groups in this run), while `auto`/PCR-SoA remains best
  or statistically tight for tiny `Naxons=64` and mixed/tight around
  `Naxons=1024`, especially probe Vm. Keep the default policy decision open
  until this is turned into an explicit policy threshold and validated with a
  smaller repeat/check run. Artifact:
  `benchmark/results/kaggle/20260710_091941_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11c-policy-gpu-c75dce6/outputs/benchmark/results/double_cable_solver_policy_gpu_smoke_gpu_20260710_091941`.
- [x] Start reducing GPU costs outside the solver before changing double-cable
  solver policy. The first generic runtime fix keeps padded `center`/`probe`
  Vm recording row-aware instead of lowering it to full Vm, and keeps multi-row
  Vm batch outputs as one `DispatchCohortRecord` when no per-row posthoc
  observer evaluation is needed. This directly targets the P11C-F `probe_vm`
  bottleneck where large populations were dominated by `results.assemble_rows`
  rather than by the solver. Covered by dispatcher tests; rerun the GPU policy
  slice before making a fresh speedup claim.
- [x] Continue reducing observer-only costs outside the solver before launching
  the next Kaggle policy gate. Factorized extracellular input preparation now
  reuses temporal-equivalent stimulus evaluations within one batch and records
  `vstim_temporal_cache_*` metadata, including the rank-1 shared-current path
  used by large recruitment-style observer-only cohorts even when each drive
  owns a distinct normalized `Stimulus` object. Chunked VmRaster combination
  now repacks chunk words with vectorized NumPy slices instead of a Python loop
  per packed word. These are solver-independent runtime plumbing changes
  intended to make large observer-only GPU simulations more solver-bound.
  Local CPU smoke with `Naxons=4096`, double-cable, observer-only,
  `tsim=2 ms`, `dt=0.02 ms`, and two 50-step chunks improved
  `inputs.extracellular` from about 878 ms to about 192 ms and changed
  temporal cache metadata from 4096 misses to 1 miss plus 4095 hits. The next
  generic cleanup caches fully encoded gated/leak membrane rows by structural
  row signature before stacking; the same smoke then reduced
  `runtime.prepare.stack_membrane` from about 8.46 s to about 0.23 s with
  5 unique membrane rows and 4091 cache hits, bringing local total visible time
  to about 5.48 s. Covered by focused factorized-input, observer-runtime,
  gated/leak membrane-stack, dispatcher, and batch tests; rerun the CPU/GPU
  Kaggle policy slice before making a fresh speedup claim.
  Follow-up generic hot-path cleanup: `runtime.prepare` and prepared-cohort
  cache keys now use compact digests plus per-`DispatchGroup` identity caches
  instead of huge per-row tuple keys; prepared cohort position/transverse arrays
  are read-only so repeated extracellular footprint cache keys can reuse
  content digests; and curve workloads reuse one
  `Activation`/policy/`AxonSimulation` context per phase instead of
  reconstructing it for every amplitude. Local CPU smoke
  `benchmark/results/p11_non_solver_opt_local_smoke_v3` confirms hot
  intra-phase `runtime.prepare` hits around `0.05-0.08 ms`, prepared-cohort
  hits around `0.2 ms`, and construct-simulation events only once per
  warmup/repeat phase. This is solver-independent plumbing only; validate on
  P100 before making a GPU speedup claim.
  Mini P100 gate at `a77f174` passed one `recruitment_curves`,
  `tiled_thomas_b32`, observer-only, `Naxons=4096`, `Nx=89`, fp32,
  same-diameter case in about 90 s wall time. It confirmed the hot
  `runtime.prepare` fix but exposed `curve.update_amplitudes.rows` as the next
  orchestration cost, around `0.55 s` per amplitude update for 4096 rows.
  Follow-up cleanup makes `Stimulus.as_unit(...)` reuse already-canonical
  stimuli and builds benchmark curve stimuli directly in ampere units; local
  no-solver micro-profiling updates 4096 rows in about `0.164 s`. The matching
  P100 mini gate at `a4b3f88` passed the same case and reduced
  `curve.update_amplitudes.rows` from about `2.216 s` total to `0.488 s` total
  over 4 updates (`~0.554 s` to `~0.122 s` per amplitude update). Artifacts:
  `benchmark/results/kaggle/20260710_184500_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11-hotpath-gpu-a77f174/outputs/extracted`
  and
  `benchmark/results/kaggle/20260710_185247_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11-hotpath-gpu-a4b3f88/outputs/extracted`.
  Next generic input cleanup caches the factorized host footprint already
  converted to mV and detects shared rank-1 stimuli by identity before falling
  back to content keys. A local 4096-row, `Nx=89`, `tsim=2 ms`, `dt=0.02 ms`
  factorized-input micro-profile now shows hot `inputs.extracellular` around
  `25-43 ms` with `vstim_footprint_mv_cache=hit` and
  `vstim_shared_current_detection=identity`. Validate this on the same P100
  mini gate before claiming GPU speedup or touching solver policy.
  Matching P100 mini gate at commit `a5effea` passed the same single
  recruitment case. `inputs.extracellular` fell from about `494.7 ms` total at
  `a4b3f88` to about `353.7 ms` total, with hot events reporting
  `vstim_footprint_mv_cache=hit` and `vstim_shared_current_detection=identity`.
  `curve.update_amplitudes.rows` stayed in the same order (`~0.56 s` total)
  and `kernel.combine_observer_chunks` stayed around `0.34 s` total
  (`~56 ms/simulation`), so the next observer-only cleanup target is VmRaster
  chunk/result handling rather than factorized footprint construction.
  Artifact:
  `benchmark/results/kaggle/20260710_190758_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11-hotpath-gpu-a5effea/outputs/extracted`.
  A follow-up device-side VmRaster chunk-combine prototype at commit `c6e842b`
  was rejected and reverted: it removed the explicit `kernel.wait` span, but
  `kernel.combine_observer_chunks` increased from about `0.34 s` to about
  `0.68 s` total and `kernel.finalize_observer.to_host` increased from about
  `0.011 s` to about `0.041 s`. Treat this as negative evidence against a
  simple eager JAX repack; the next VmRaster cleanup should avoid the host
  repack structurally or revisit chunk policy with a dedicated benchmark,
  rather than dispatching several small JAX scatter/update ops after each
  chunked simulation. Artifact:
  `benchmark/results/kaggle/20260710_191549_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11-vmraster-device-combine-c6e842b/outputs/extracted`.
  New structural benchmark-only prototype:
  `--benchmark-observer-state-scope full` keeps the solver time loop chunked
  while VmRaster uses one full-duration observer state and absolute chunk start
  indices. This avoids `kernel.combine_observer_chunks` without changing
  `BatchOptions` or runtime defaults. Local unit coverage confirms identical
  VmRaster words versus default chunk-local observer state, and local smoke
  `benchmark/results/p11_observer_full_state_scope_smoke` confirms the CLI path
  reports `resolved_observer_state_scope=full` with no
  `kernel.combine_observer_chunks`. Next gate: run the same P100 mini case as
  `a5effea` with `--benchmark-observer-state-scope full`.
  First large-population GPU solver-only sweep completed on Kaggle P100 at
  commit `7fcd109` with `fp32`, `B=128..32768`, `Nx=47/89/129`, shared and
  batched coefficients, `block_b=32`, and variants `current_pcr_soa`,
  `thomas_batched_scan`, `jax_triton_tiled_thomas_loop`, and
  `large_population_exact_double_cable_jax`. Best Triton is already faster at
  `B=128` (`~1.1-1.3x` vs PCR/SoA), grows to `~3-6x` at `B=4096`, and reaches
  `~8-16x` at `B=16384..32768` depending on `Nx` and coefficient mode. Residuals
  stay around `1.3e-7`. Artifact:
  `benchmark/results/kaggle/20260709_092034_large_population_double_cable_solver_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-largepop-gpu-sweep/outputs/extracted`.
  First integrated workflow GPU gate completed on Kaggle P100 at commit
  `d722906`, comparing the current PCR route against the benchmark-only
  `jax_triton_loop_xb` override on `threshold_curves` and `recruitment_curves`
  with `Naxons=8192`, target `Nx=101` / actual `Nx=89`, fp32, double-cable,
  observer-only, different diameters, `tsim=20 ms`, `dt=0.01 ms`, unchunked,
  and RSS tracing. Artifacts:
  `benchmark/results/kaggle/20260709_134542_threshold_curves_gpu_realistic_gpu_NvidiaTeslaP100_axs-p11c-thr8192-base/outputs/extracted`,
  `benchmark/results/kaggle/20260709_134555_threshold_curves_gpu_realistic_gpu_NvidiaTeslaP100_axs-p11c-thr8192-triton/outputs/extracted`,
  `benchmark/results/kaggle/20260709_134850_recruitment_curves_gpu_realistic_gpu_NvidiaTeslaP100_axs-p11c-rec8192-base/outputs/extracted`, and
  `benchmark/results/kaggle/20260709_134904_recruitment_curves_gpu_realistic_gpu_NvidiaTeslaP100_axs-p11c-rec8192-triton/outputs/extracted`.
  Steady repeat timing improves strongly: threshold `curve.simulate` mean
  `10.75 s -> 2.98 s` (`3.61x`) and `kernel.dispatch_jax`
  `8.46 s -> 1.20 s` (`7.04x`); recruitment `curve.simulate` mean
  `11.74 s -> 3.97 s` (`2.96x`) and `kernel.dispatch_jax`
  `9.43 s -> 2.06 s` (`4.57x`). All-phase totals including cold/warmup are
  `87.9 s -> 45.2 s` for threshold (`1.95x`) and `305.7 s -> 122.5 s` for
  recruitment (`2.50x`). Threshold amplitudes were too low for physical
  threshold values in this pass (`above_range`), so use that run for timing
  only; next threshold reruns should use the physically bracketed reference
  setup from `examples/basic/07_threshold_vs_diameter.py` (`MRG` point-source
  thresholds use `1..50 uA` at `100 us`, `5..150 uA` at `50 us`, monophasic
  cathodic pulse, node-centered electrode). Recruitment activation counts match
  between PCR and Triton. Peak RSS end is about `+212 MiB` for threshold and
  `+215 MiB` for recruitment with Triton.
  A corrected threshold smoke gate then passed on Kaggle P100 at commit
  `3f60820`: the benchmark defaults were aligned with
  `examples/basic/07_threshold_vs_diameter.py`, and the threshold spatial setup
  now keeps axons on the intrinsic axis (`z=0`) with the point electrode at
  `z=100 um`. The first rerun at commit `7b67a42` still produced
  `above_range` thresholds and exposed that geometry mismatch. The corrected
  base and Triton runs both found `64/64` thresholds in range, with identical
  PCR/Triton thresholds (`max_abs_uA=0.0`) over `Naxons=64`, target `Nx=101`
  / actual `Nx=89`, fp32, observer-only, double-cable, different diameters,
  `tsim=5 ms`, `dt=0.01 ms`, `max_iterations=8`, `1..50 uA`, and monophasic
  stimulation. Artifacts:
  `benchmark/results/kaggle/20260709_141540_threshold_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p11c-thr64g-base/outputs/extracted`
  and
  `benchmark/results/kaggle/20260709_141540_threshold_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p11c-thr64g-triton/outputs/extracted`.
  Interpretation: the integrated benchmark-only Triton path survives realistic
  curve workflow timing, but policy is still open until the matrix covers
  `Naxons`, `Nx`, dtype, recording modes, CPU/GPU comparison, corrected
  threshold amplitude ranges, and cold/warm cache behavior.
  Solver-route inventory and decision map for the cleanup/promotion discussion:
  `docs/architecture/p11c_solver_path_inventory_2026_07_09.md`.
- [ ] P11C-G promote the Triton solver if evidence supports it:
  if P11C-F shows robust wins and acceptable cold-start/memory/correctness
  behavior, make the Triton double-cable solver a selectable solver option
  alongside the existing routes. Only then consider making it part of the
  default GPU policy for the shapes where it is demonstrably better.
  The objective is to validate Triton deeply enough that it can become the
  preferred GPU double-cable route for supported fp32 large-population shapes,
  unless the full benchmark matrix disproves or sharply constrains that policy.

### P11D - Solver Engine Flattening

Goal: flatten solver execution around typed single-cable/double-cable solver
policy and backend CPU/GPU engines instead of exposing scalar/batch and
solver-variant internals. The migration plan lives in
`docs/architecture/p11d_solver_engine_migration_plan_2026_07_09.md`.

- [x] Add typed public solver policy objects under `ExecutionPolicy`:
  `SolverPolicy` plus runtime-specific JAX constructors under
  `axs.runtime.jax`, and solver-specific option dataclasses. Keep
  `axs.runtime`, `Device`, and `PrecisionPolicy` as the high-level
  runtime/device/precision surface. Initial P11D-A implementation adds
  `SingleCableSolverKind`, `DoubleCableSolverKind`, `PcrSolverOptions`, and
  `TiledThomasSolverOptions` behind that runtime namespace, with focused
  public-policy tests. Runtime behavior is intentionally unchanged in this
  slice.
- [x] Move double-cable solver selection out of public `BatchOptions`.
  Recording/output policy should come from `Recording` plus observers; solver
  policy should come from `ExecutionPolicy.solvers`.
  P11D-B implementation removes the solver field from `BatchOptions`, resolves
  typed `ExecutionPolicy.solvers` through the JAX backend context, forwards
  `axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(block_b=...)` to the
  kernel, translates active
  benchmark curve flags to typed policies at the boundary, keeps internal
  benchmark overrides out of `BatchOptions`, and reports the effective policy in
  inspection.
- [x] Introduce one internal `OutputPlan` for full Vm, probe/sampled Vm, and
  VmRaster observer-only output so recording modes differ by output sink rather
  than separate orchestration routes. The JAX group runner now lowers
  `BatchOptions` into a backend-local `OutputPlan`.
- [x] Add a backend-private CPU/GPU solver-engine resolution layer and route
  current batch kernel solver selection through it. The active JAX context now
  carries a `JaxSolverEngine` resolved from typed public policy.
- [x] Normalize one-axon runs toward `B=1` in the shared population lifecycle
  where feasible. Vm/VmRaster-compatible one-row groups now use the batch route;
  dense observable recordings keep the scalar fallback because batch kernels do
  not retain gates/currents/conductances payloads yet.
- [x] Split shared solver core from engine-specific linear solvers and layouts:
  membrane updates, RHS/system assembly, extracellular drive, scan helpers,
  observers, and result packaging should be shared; CPU/GPU should specialize
  only layout and linear solve hot paths. Initial slice extracts shared batched
  membrane gate/current/linearization/state operations plus the batch-native
  double-cable linear step into `axonscope.runtime.jax.solver_core`; broader
  observer/result extraction remains part of the same migration if duplication
  reappears.
- [x] Move benchmark-only PCR/Triton probes out of production-facing solver
  vocabulary or mark them clearly as experimental/benchmark-only. Do not expose
  them as public runtime choices without P11C-F evidence. Legacy string
  double-cable solver aliases and resolver helpers now live under the internal
  JAX backend/benchmark boundary, not `axonscope.solvers`.
- [x] Update tests, inspection, performance policy, and active benchmark curve
  CLI mapping to the typed solver-policy surface. Benchmark CLIs may keep
  string flags, but they must translate them to typed policies at the boundary.
- [x] Update remaining examples, advanced examples, docs, and specialized
  benchmark/analysis scripts to the typed solver-policy surface. Benchmark CLI
  strings are acceptable only at the boundary. Source-of-truth docs and the
  runtime example now use `SolverPolicy(single_cable=..., double_cable=...)`;
  active benchmark workloads translate CLI solver strings into
  `axs.runtime.jax` solver constructors at the boundary.
- [x] Update public examples and advanced examples affected by the solver-engine
  migration. Basic examples should stay simple and use defaults unless they
  teach CPU/GPU or precision policy; targeted advanced runtime examples should
  cover typed single-cable/double-cable solver policy, precision/device policy,
  and any promoted GPU solver-specific options. Added
  `examples/advanced/runtime/04_solver_policy.py` and updated
  `examples/README.md`.
- [x] Validate with focused unit tests and a quick local CPU benchmark artifact
  before making any policy or performance claim. Local JAX devices expose only
  `cpu:0`; no GPU artifact was produced on this machine. Fresh CPU artifact:
  `benchmark/results/p11d_quick_cpu_validation`. Triton default/promotion and
  GPU performance claims remain separate P11C-F/P11C-G Kaggle/local-GPU gates.

### P12 - Studies, Serialization, Integration

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

## Completed Phase Summary

Completed work is intentionally summarized rather than kept as a detailed
ledger in this TODO.

- P0-P6: public API cleanup, one simulation workflow, protocols/results/views,
  examples-as-docs, inspection/runtime reports, and backend/lowering cleanup
  are complete for the current JAX path.
- P7: class-based public membrane models, source compiler, generated JAX/NumPy
  model-step artifacts, generated-code cache/reporting, direct
  `JaxMembraneProgram` execution, and old membrane-stack deletion are complete.
- Benchmark surface classification, post-P7 flattening, benchmark memory
  observability, and P2 model-codegen/model-step/template-simulation coverage
  are complete.
- P4/P5 are complete: public estimate/inspection modules use the backend
  execution boundary for backend-owned benchmark support, and validation policy
  now separates fast acceptance, NRV, and performance benchmark evidence.
- P9 is complete: cold-run micro baseline, scalar/batch span normalization,
  explicit hotpath chunk controls, and closeout decisions are recorded. P8
  NumPy/SciPy is parked as a future bonus until P10/P11 make the model/compiler
  surface and current JAX solver evidence cleaner.

## Key References

- Architecture reference: `GUIDELINES.md`
- Agent guide: `AGENTS.md`
- Detailed cleanup plan: `docs/architecture/cleanup_plan.md`
- Runtime-agnostic DSL/P7 plan: `docs/architecture/runtime_agnostic_dsl_plan.md`
- Solver organization: `docs/solver_organization.md`
- Axon model organization: `docs/axon_model_organization.md`
- Stimulation model: `docs/stimulation.md`
- Pool dispatch: `docs/pool_dispatch.md`
- Recording/results/analysis: `docs/results_recording_analysis.md`
- Membrane docs: `docs/membranes.md`
- Benchmark surface map: `benchmark/README.md`
- Benchmark lifecycle registry: `benchmark/registry.py`
- Hotpath workloads: `benchmark/hotpaths/README.md`
- Active solver README: `benchmark/solvers/README.md`
- Pseudo-double standby: `benchmark/pseudo_double/README.md`
- Archived solver spikes:
  `benchmark/archived_solver_spikes/`, `benchmark/triton_solver/`,
  `benchmark/jax_triton_solver/`, `benchmark/cuda_ffi_solver/`,
  `tests/archive/solver_spikes/`
