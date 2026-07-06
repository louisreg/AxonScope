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

Updated on 2026-07-04 after the P9 closeout and roadmap reprioritization.

Current state:

- P7 is closed: public membrane authoring is class-based through
  `axs.membranes.Model`; built-ins live as standalone model classes under
  `src/axonscope/membranes/models/`.
- The historical `channel_models`, `icm`, `model_ir/models`, and
  `model_ir/builtins.py` paths are removed from the active package and must
  stay absent.
- Model IR remains internal compiler/runtime vocabulary. Users write membrane
  models, equations, parameters, gates, currents, and observables.
- `Runtime.NUMPY` remains reserved for a future real NumPy/SciPy reference
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

### P4 - Backend Boundary For Inspection And Estimates

- [x] Decide whether `inspection.py` and `performance.py` may import
  `axonscope.backends.jax.*` directly, or whether input/recording lowering
  summaries should move behind a backend inspection/estimate facade.
- [x] If the current inspection remains JAX-specific, label that explicitly in
  docs and inspection records. Decision: not needed for the public modules;
  JAX-specific details now live behind the backend benchmark facade.
- [x] If a facade is preferred, route planning, estimate, and inspection through
  the same backend boundary used by execution without forcing device transfers.
  The JAX backend exposes host-side benchmark support in
  `src/axonscope/backends/jax/benchmark.py`, delegated through
  `axonscope.backends.execution`.

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

- [ ] Keep `Runtime.NUMPY` reserved/non-executable until this phase reaches
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
- [ ] Wire `ExecutionPolicy(runtime=Runtime.NUMPY)` only after executable
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
  `src/axonscope/backends/jax/model_ir_lowering.py`,
  `src/axonscope/backends/jax/membrane_program.py`, generated-code cache code,
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
  `axonscope.backends.execution`, so benchmark scripts do not import JAX
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
  use `rss`/`device` for larger local, GPU, and publication-scale sweeps.
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
  before any default change; matching Kaggle CPU/GPU and threshold-curve
  evidence are still pending.
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
  staged implementation. Implement candidates incrementally, each gated by
  correctness checks plus small benchmarks first, then larger realistic
  benchmarks before keeping or promoting the change.
- [ ] Re-run NRV validation only for numerical behavior changes, but always
  pair optimization claims with fresh hotpath or realistic benchmark evidence.

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
