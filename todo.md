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
- Reusable runnable plans are a future lifecycle phase, not an unfinished P14
  optimization: current structural preparation is below 3% of realistic warm
  `run_pool` time.

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

### Reusable Runnable Plans

Revisit this as a deliberate lifecycle architecture phase only when repeated
calls are a concrete product workflow or new measurements show material cold
or startup cost.

- [ ] Keep construction outside execution lazy: axons, populations, protocols,
  sweeps, and studies compose lightweight immutable simulation descriptions.
  Canonicalize unit-bearing values once per unique scientific template;
  materialization, code generation, and device placement begin only in
  `run()`, `estimate()`, or `inspect()`.
- [ ] Map those descriptions onto one immutable internal prepared plan consumed
  by `AxonSimulation(...).run()`. Do not introduce a parallel public execution
  path. Discuss any lifecycle, result, inspection, progress, or output change.
- [ ] Reuse compatible dispatch groups, spatial rows, observer probe plans,
  membrane/cable rows, factorized footprints, device arrays, and executable
  lookup across calls while accepting only typed dynamic operands.
- [ ] Benchmark fresh miss, same-object hit, reconstructed structural hit,
  dynamic-value reuse, and explicit invalidation at Naxon=1024/4096. Require at
  least 15% repeated-call wall-time gain or a multi-second cold reduction.

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
- [ ] Define final schemas, typed serialization, and persistence strategy.

### External Integration

- [ ] Continue NRV hardening only where its package contract is stable. Keep
  geometry in `examples/with_nrv` or benchmarks and avoid duplicating the
  sampled-footprint path in `axonscope.integrations.nrv`.
- [ ] Work on HPC integration, including cache sharing, scheduling, artifact
  retention, and reproducible benchmark execution.
- [ ] Benchmark async JAX scheduling for forced heterogeneous groups
  (incompatible membrane contracts, cable/Nx shapes, or temporal signatures).
  Sweep 2/4/8 groups and require device-idle evidence, bounded pending memory,
  deterministic ordering, and end-to-end gain.
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
