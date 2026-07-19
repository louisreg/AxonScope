# AxonScope TODO

Living execution plan for AxonScope. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh benchmark
artifacts own current behavior. Historical detail before the 2026-07-12
cleanup remains in `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.
Completed performance evidence and commands live in `benchmark/README.md`.

## Snapshot

Updated on 2026-07-19 after closing the pre-P18 performance ledger.

- P7, P11, P12, the VmRaster part of P13, P14-P17, and P17B are closed.
- The production runtime is JAX. CPU double-cable uses Thomas; CUDA
  double-cable uses tiled Thomas in Triton. CPU single-cable uses JAX
  tridiagonal solve; CUDA single-cable uses exact tiled Thomas in Triton.
- Recruitment uses one source population plus a native numeric amplitude axis.
  It does not build `Namplitude x Naxon` Python simulation objects.
- Compact activation, latency, spike-count, and bounded spike-time states are
  available. VmRaster remains the temporal reference where history is needed.
- The active order is P18 membrane completion, then P19 pre-v1 convergence.
- P20 tracks the future lazy runnable-plan and distributed-runner architecture.
  It is not an unfinished P14 optimization: current structural preparation is
  below 3% of realistic warm `run_pool` time.

Latest fast validation:

```text
python -m compileall -q src tests/unit
pytest -q tests/unit --tb=short
740 passed, 1 skipped
```

## Non-Negotiables

- AxonScope is pre-release with one active user. Prefer direct convergence over
  compatibility shims, deprecated wrappers, or parallel old/new paths.
- Keep one concept, one public name, one execution path, and one canonical
  public result model.
- Promote an optimization by replacing the production route. Benchmark-only
  prototypes may coexist temporarily; remove rejected or replaced variants.
- Discuss changes to public results, serialization, progress output, or the
  canonical workflow before implementing them.
- Public orchestration enters JAX through `axonscope.runtime.execution`.
- Public examples must not import runtime or solver internals.
- External packages own nerve geometry, trajectories, world coordinates,
  electrode CAD, and FEM solves. AxonScope owns intrinsic axon coordinates,
  sampled-footprint stimulation, cable/membrane execution, recording, and
  analysis.
- Document every public feature in a runnable example or remove it.
- Do not remove an unfinished task unless it is completed, rejected, or moved
  to a named tracking document.

## Active Roadmap

### P18 - Membrane Model Completion And Validation

- [ ] Implement Nav1.x-family and other Markov-based membrane models through
  the canonical membrane-source and generated-runtime contracts.
- [ ] Re-check every built-in membrane model against its NRV implementation;
  audit formulas, defaults, states, temperature behavior, and recording
  semantics that may have been lost during translation.
- [ ] Finish missing Gaines and Markov model families.
- [ ] Add focused numerical references and runnable advanced examples for each
  retained public model.

### P19 - Pre-V1 Cleanup And Public Surface

- [ ] Reorganize `src/`, especially Python modules still at package root, after
  P18 settles the remaining ownership boundaries.
- [ ] Inventory every Python module, function, type, and public export.
- [ ] Use Graphify, `vulture`, source call sites, examples, and tests to remove
  code used only by tests, legacy paths, replaced slow routes, and duplicated
  implementations.
- [ ] Verify retained ownership, contracts, runtime boundaries, and absence of
  duplicate public concepts.
- [ ] Remove public API not documented by basic or advanced runnable examples.
- [ ] Run a final Graphify-guided whole-package cleanup. Treat `runtime/jax` as
  the already completed first cleanup slice.
- [ ] Clean `pyproject.toml`, including explicit optional CUDA/Triton
  dependencies and removal of obsolete extras.
- [ ] Revisit artifact caching globally: build only requested runtimes, define
  first-call versus install-time built-ins, document clean/disable/retention,
  and keep `.axonscope_cache` deterministic and inspectable. Evaluate final
  time-chunk shape buckets only with numerically inert observer padding.

### P3 Remainder - Documentation And Examples

README and public-example audits are complete.

- [ ] Write the indexed notebook mini-course under `examples/tutorials/`.
- [ ] Prepare proper Sphinx documentation.
- [ ] Complete and update all public docstrings.

### P13 Remainder - Dense Vm Recording

- [ ] Benchmark dense/full Vm recording separately from VmRaster. It may need
  a distinct chunk policy, but it must not complicate the established
  `DEFAULT_OBSERVER_TIME_CHUNK_STEPS = 512` default without evidence.

## Future Product Phases

### P20 - Lazy Runnable Plans And Distributed Runner

Axons, populations, simulations, recruitment curves, sweeps, and studies should
produce immutable runnable simulation plans. One runner executes one or many
plans and owns all eager work. This is a future architecture replacement, not a
wrapper around the current dispatcher.

- [ ] Before implementation, use Graphify to audit the complete pre-runner
  architecture: `Axon`, `AxonPopulation`, `AxonSimulation`, protocols, studies,
  dispatch plans, preparation caches, runtime execution, results, inspection,
  and progress reporting. Identify what can move, merge, or disappear.
- [ ] Define one backend-neutral immutable `RunnablePlan` contract for a single
  simulation and composed plans for populations, sweeps, recruitment curves,
  and studies. Plans describe work and expected results; they do not allocate
  runtime arrays, place device data, compile kernels, or execute eagerly.
- [ ] Make everything outside the runner lazy. Canonicalize unit-bearing values
  once per unique scientific template, but defer numerical materialization,
  grouping, signatures, generated-code loading, device placement, compilation,
  scheduling, and result allocation until runner execution or an explicit
  `estimate()`/`inspect()` request.
- [ ] Implement one runner for one or many plans. It owns dependency ordering,
  compatible grouping, reusable prepared state, execution policies, device
  assignment, failure propagation, cancellation, progress, and canonical
  result assembly.
- [ ] Replace the existing simulation/protocol dispatcher and execution routes
  with the runner once validated. Remove superseded builders, caches, wrappers,
  and call paths; do not retain legacy aliases, a parallel optimized path, or a
  hidden fallback.
- [ ] Design the runner dispatcher for one or many local GPUs and HPC workers.
  Separate backend-neutral scheduling from JAX device execution; model device
  topology, memory budgets, affinity, work placement, cache locality, and
  deterministic result ordering without exposing backend internals publicly.
- [ ] Support synchronous execution first, then benchmark asynchronous overlap
  only for genuinely independent heterogeneous groups. Require bounded pending
  memory, observable device-idle reduction, deterministic failures/results, and
  an end-to-end gain before enabling it by default.
- [ ] Preserve the canonical `AxonSimulation(...).run()` workflow, public
  results, inspection, and progress semantics unless a change is discussed
  first. Public objects may become plan factories, but users should not need to
  understand dispatcher or runtime internals.
- [ ] Validate single plans, mixed populations, native numeric axes, studies,
  cache invalidation, cancellation, CPU, single GPU, multi-GPU, and an HPC
  smoke path. Benchmark fresh miss, same-object and structural reuse, dynamic
  operands, transfers, compilation, solve, result assembly, RSS, device memory,
  scheduler overhead, and scaling efficiency at Naxon=1024/4096.

### Future Runtime Targets

- [ ] When a non-JAX runtime is introduced, generate its own target-specific
  model artifact, metadata, and callables from the canonical membrane source
  contract. It must not interpret or depend on `jax_model.py`.

### Propagation And Conduction-Block Validation

The removed `ConductionBlock` definition must not return as inverted distal
activation. A public block result requires a dedicated scientific campaign.

- [ ] Add canonical propagation analysis with one source probe and one or more
  targets. Classify `propagated`, `blocked`, `not_initiated`, and `ambiguous`.
- [ ] Require target activation after source activation, optionally inside a
  valid delay window, so direct or reverse activation is not misclassified.
- [ ] Support proximal/distal probes and distinguish bidirectional,
  proximal-only, distal-only, local-only, no-initiation, and direct activation.
- [ ] Define repeated-spike propagation separately. Use counters and last-event
  matching where possible; use a bounded FIFO only for required event pairing.
- [ ] Run dedicated CPU/GPU validation for propagation and true conduction
  block before exposing a public analysis or block-threshold study.
- [ ] Add a didactic KES/block example covering local activation, failed
  initiation, propagation, true block, and the required filtering workflow.

### Studies And Persistence

- [ ] Implement callable threshold curves, block-threshold curves, recruitment
  curves, conduction validation, parameter sweeps, reuse/retention policies,
  and canonical study results.
- [ ] Evaluate optimization/search algorithms built from or compatible with
  scikit-learn for study parameter exploration, including grid/random sampling
  and surrogate-model-guided search. Keep the study contract independent of
  scikit-learn, compare it with SciPy or dedicated optimization packages, and
  make any dependency optional.
- [ ] Define final schemas, typed serialization, and persistence strategy.

### External Integration

- [ ] Continue NRV hardening only where its package contract is stable. Keep
  geometry in `examples/with_nrv` or benchmarks and avoid duplicating the
  sampled-footprint path in `axonscope.integrations.nrv`.
- [ ] Build HPC integration through the P20 runner after its local multi-device
  contract is stable, including cache sharing, remote scheduling, artifact
  retention, failure recovery, and reproducible benchmark execution.
- [ ] Implement the CPU/NRV FEM-footprint path from
  `ideas/fem_axon_gpu_coupling_design.md` before GPU FEM. Separate FEM solve,
  first footprint, cached sampling, and AxonScope solve; cache field bases and
  avoid repeated point location through axon embedding/projection.
- [ ] Test Apple Metal acceleration with `jax-mps`:
  https://github.com/tillahoffmann/jax-mps

### Deferred NumPy/SciPy Reference Runtime

This is a future debugging/reference backend, not a JAX wrapper or the next
implementation phase.

- [ ] Keep `axs.runtime.numpy` reserved until it uses the same
  `AxonSimulation(...).run()`, `.estimate()`, and `.inspect()` lifecycle.
- [ ] Define a deterministic v1 subset: single cable, intracellular current,
  sampled footprints, recording, observers, and selected membranes.
- [ ] Implement tridiagonal Crank-Nicolson with readable NumPy/SciPy primitives.
- [ ] Consume a NumPy-specific generated membrane target; never call into JAX
  or retain a second user-facing membrane-authoring path.
- [ ] Add JAX comparisons for Vm, activation, block, latency, thresholds,
  probes, retained membrane values, and model-step equivalence.
- [ ] Add runtime policy, examples, docs, estimates, and inspection records
  only after executable behavior exists.
- [ ] Document its use for tiny deterministic debugging and numerical
  regression, not population performance or GPU parity.

## Completed Phase Summary

- **P7/P11/P12:** converged the public simulation workflow and JAX runtime,
  removed replaced runtime paths, retained factorized Vext, and established
  strict CPU/CUDA route guards plus persistent JAX/Triton compilation caches.
- **P13 observer slice:** retained VmRaster chunk size 512 after CPU/P100
  measurement; rejected an adaptive global policy.
- **P14:** replaced amplitude-expanded Python pools with a generic numeric axis,
  shared immutable axon/layout/membrane templates, compact factorized inputs,
  bounded preparation caches, and trusted structural signatures. At Naxon=4096
  plan plus runtime preparation is below 3% of warm `run_pool`.
- **P14B translated layouts:** three structural templates represent 4096 rows
  and 1024 translations while retaining row-specific positions and footprints.
  CPU/P100 activations match exactly; warm P100 `run_pool` is 138.3x faster.
- **P15:** added typed compact activation, latency, spike count, and bounded
  spike-time observers without restoring a generic solver-side fallback.
- **P16:** retained node-first double-cable state and fused physical-term Triton
  assembly, improving warm P100 `run_pool` by about 26% at Naxon=1024/4096.
  Rejected scan inversion, aliases, homogeneous async scheduling, and adaptive
  chunks when they did not clear the retention threshold.
- **P17:** made generated runtime artifacts the only production source of
  model-specific JAX/Triton facts. Retained generated membrane kernels; rejected
  same-step membrane/Thomas fusion, temporal blocking, and rate tables.
- **P17B:** promoted the exact tiled-Thomas Triton single-cable solver after
  numerical validation and 1.23x/1.60x warm `run_pool` gains at
  Naxon=1024/4096. CPU retains the JAX tridiagonal route.
- Basic examples 06/07/08 and `with_nrv/01` were validated locally and on
  Kaggle CPU/P100. Detailed commands, timings, decisions, and artifact paths
  remain in `benchmark/README.md`.

## Key References

- Product and architecture: `GUIDELINES.md`
- Working guide: `AGENTS.md`
- Benchmark surfaces and completed evidence: `benchmark/README.md`
- Validation policy: `docs/validation.md`
- Examples map: `examples/README.md`
- P11 closeout: `docs/architecture/p11_closeout_2026_07_12.md`
- P12 runtime cleanup:
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md`
- Pre-cleanup TODO archive:
  `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`
