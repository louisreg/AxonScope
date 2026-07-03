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

Updated on 2026-07-03 after the P4/P5 backend-boundary and validation-policy
cleanup.

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
  runtime. It must not become a JAX-backed compatibility path.
- Solver-side observer-only execution is the strict VmRaster path under
  `observations["vm_raster"]`; activation, latency, velocity, threshold, and
  recruitment summaries are post-processing.
- `PeakVoltage` remains post-hoc on recorded Vm unless a dedicated benchmarked
  solver-side design is accepted.
- The next useful work is P3 documentation/examples cleanup before starting a
  major P8 runtime project.

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

- [ ] Refaire `README.md` après la stabilisation post-P7: présenter l'API
  actuelle, le workflow `AxonSimulation`, les membranes class-based, la
  stratégie benchmark, et pointer vers les exemples/docs à jour.
- [ ] Faire une mise à plat manuelle de `docs/`, `GUIDELINES.md` et
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
- [ ] Audit public examples after benchmark flattening so examples remain
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

### P8 - Future NumPy/SciPy Reference Solver Runtime

This phase starts only after post-P7 cleanup and model benchmarks are stable.
The goal is a real reference solver runtime, not a JAX-backed compatibility
path.

- [ ] Keep `Runtime.NUMPY` reserved/non-executable until this phase reaches
  executable behavior through the same `AxonSimulation(...).run()`,
  `.estimate()`, and `.inspect()` lifecycle as JAX.
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

- [ ] Keep shape bucketing internal and opt-in until benchmarks show an
  end-to-end cold-run win.
- [ ] Investigate persistent JAX compilation/cache policy or a dedicated
  compiler-level strategy if cold `kernel.dispatch_jax` remains a product
  requirement. Keep this separate from solver-route cleanup.
- [ ] Explore recruitment amplitude micro-batching as a benchmark axis, but
  keep the runtime/protocol default at one amplitude per solver call until
  evidence says otherwise. Benchmark candidate `amplitude_batch_size` values
  such as 1, 2, 4, and 8 against peak memory, footprint duplication,
  cold/warm time, and observer-only result assembly.
- [ ] Benchmark and formalize `time_chunk_steps` policy for observer/result
  assembly. Compare unchunked, 250, 500, 1000, and adaptive values across full
  Vm, probe Vm, and observer-only outputs; track peak memory, chunk overhead,
  cold/warm time, GPU utilization, result equivalence, and whether defaults
  should depend on `nt`, `Naxons`, recording mode, or backend.
- [ ] Park performance optimization unless explicitly in scope. When
  optimization resumes, start with a cold-path audit for large synthetic/GPU
  populations (`n=1000`): split `build pool`, `dispatch.build_plan`,
  `runtime.prepare`, and `kernel.dispatch_jax`; investigate row-by-row
  planning/preparation overhead before changing kernel routes or adding
  scheduling complexity.
- [ ] GPU dispatch scheduling: memory-aware bucket/coalesce first, optional
  async enqueue second, only after memory budgets and group-route inspection
  exist. See `ideas/axonscope_dispatch_scheduling_gpu_note.md`.
- [ ] Double-cable rank-K compact `Vext`: future optimization/validation slice.
  Current double-cable keeps dense materialization for unsupported compact
  cases. Only broaden compact forcing after equivalence tests against dense
  results and benchmark evidence for memory/time benefits.
- [ ] Improve GPU solver: see
  `ideas/axonscope_gpu_tridiagonal_solver_literature_synthesis.md` and update
  `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`.

### P10 - Future Membrane-Model Extensions

These are not blockers for P7, but they are unfinished work preserved from the
previous TODO.

- [ ] Tighten rejected Python construct diagnostics for mutation,
  data-dependent Python loops, I/O, dynamic imports, object construction inside
  equations, arbitrary NumPy/JAX calls, hidden global state, and side effects.
- [ ] Complete and document the public helper surface:
  `exp`, `expm1`, `log`, `log1p`, `sqrt`, `abs`, `minimum`, `maximum`, `clip`,
  `where`, `tanh`, `sigmoid`, `vtrap`, `q10`, `boltzmann`,
  `rates_from_tau_inf`, `nernst`, and concentration/current conversion helpers.
- [ ] Extend mechanism semantics beyond ordered sections: expose
  mechanism-level dependencies in reports, preserve boundaries for
  optimization/fusion, and apply the same readable shape to complex built-ins
  where useful.
- [ ] Decide whether public authoring needs explicit syntax for currents whose
  conductance/reversal cannot be inferred from `I_x = g_x * (Vm - E_x)`.
- [ ] Extend semantic validation to purity/source provenance, unsupported
  helper calls, duplicate exports, duplicate observable names, and
  recording/output compatibility.
- [ ] Define target-specific lowering hooks for JAX and NumPy intrinsics while
  keeping scientific semantics target-neutral.
- [ ] Make recording-aware output pruning part of the compiler plan:
  requested Vm/probes/observables should determine retained outputs before
  backend lowering.
- [ ] Finish backend-neutral optimization closeout: common subexpression
  elimination, unused diagnostic pruning, stable optimized-graph hashing, and
  explainable before/after summaries.
- [ ] Implement JAX-specific fusion closeout: generated conductance terms,
  state prepare/finalize updates, diagnostics, requested-observable pruning,
  composite generated programs or an explicit fail-fast boundary, and avoiding
  transport of unrequested intermediate arrays.
- [ ] Broaden generated-artifact identity to the full target-specialized key:
  internal graph hash, optimized graph hash, backend lowering key, static
  shapes, recording policy, parameter specialization, dtype/precision,
  optimization level, and compiler/helper versions.
- [ ] Extend generated execution into the full fusion path beyond the
  P7-supported class subset: composite generated programs, more aggressive
  recording-aware pruning, and direct solver-kernel fusion.
- [ ] Define duplicate-name aggregation semantics for generic observables before
  exposing custom observables as public recording outputs.

### P11 - Studies, Serialization, Integration

- [ ] Prepare a publication-grade benchmark campaign for AxonScope versus
  baselines. Cover velocity, activation-threshold curves, block thresholds,
  recruitment curves, `dt`, `Nx`, `Naxons`, FP32 versus FP64, full Vm, probe
  Vm, observer-only outputs, single-cable, double-cable, mixed populations,
  same-diameter versus different-diameter cohorts, CPU versus GPU versus NRV.
  Keep this reproducible with fixed presets, saved raw data, plots, and
  publication-ready summary tables.
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
