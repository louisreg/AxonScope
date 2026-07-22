# AxonScope TODO

Living execution plan for AxonScope. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh validation or
benchmark artifacts own current behavior. Completed implementation detail lives
in the phase audits and `benchmark/README.md`, not in this checklist.

## Snapshot

Updated on 2026-07-22 after closing P18 and consolidating the remaining roadmap.

- P7, P11, P12, the VmRaster part of P13, and P14-P18 are closed.
- The production runtime is JAX. CPU double-cable uses Thomas; CUDA
  double-cable uses generated Triton tiled Thomas. CPU single-cable uses JAX
  tridiagonal solve; CUDA single-cable uses exact Triton tiled Thomas.
- Recruitment uses one source population plus a native numeric amplitude axis;
  it does not materialize `Namplitude x Naxon` Python simulation objects.
- Compact activation, latency, spike-count, bounded spike-time, and VmRaster
  observers are available. Dense recording remains a separate memory/performance
  concern.
- P19 pre-v1 convergence is the only active phase. P20 and later work is future
  architecture or product expansion.

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
- Public examples must use public APIs, never runtime or solver internals.
- External packages own nerve geometry, trajectories, world coordinates,
  electrode CAD, and FEM solves. AxonScope owns intrinsic axon coordinates,
  sampled-footprint stimulation, cable/membrane execution, recording, and
  analysis.
- Document every public feature in a runnable example or remove it.
- Do not remove an unfinished task unless it is completed, explicitly rejected,
  or moved to a named tracking document.

## Active Roadmap

### P19A - Source And Public-Surface Convergence

- [ ] Build one Graphify- and `vulture`-assisted inventory of every Python
  module, public export, function, and type. Record production, example,
  benchmark, and test-only call sites before deleting anything.
- [ ] Remove code used only by tests, legacy wrappers, replaced slow routes,
  duplicate implementations, and public names without runnable examples. Do
  not preserve compatibility aliases or hidden fallbacks.
- [ ] Reorganize `src/`, especially modules still at package root, after dead
  code is removed. Preserve ownership boundaries and avoid moves that only
  exchange one flat namespace for another.
- [ ] Verify that public simulation, inspection, estimates, protocols,
  recording, results, and analyses each have one canonical contract and route
  through `axonscope.runtime.execution` where runtime work is required.
- [ ] Run a final whole-package Graphify audit after cleanup and confirm that
  no duplicate public concepts, model-specific solver branches, or stale
  runtime paths remain.

### P19B - Packaging, Caches, And Repository Hygiene

- [ ] Clean `pyproject.toml`: remove obsolete extras and expose explicit,
  tested optional CUDA/Triton dependencies without making GPU packages a CPU
  installation requirement.
- [ ] Converge artifact caching globally. Build only requested runtime targets;
  define first-call versus install-time behavior for built-ins; document
  inspect, clean, disable, invalidation, and retention policies; keep
  `.axonscope_cache` deterministic. Evaluate additional time-chunk shape
  buckets only if observer padding is numerically inert and benchmarked.
- [ ] Audit local and remote Git branches, preserve any still-relevant unmerged
  work, then delete every branch except `main`.

### P19C - Documentation And Pre-V1 Gate

- [ ] Complete public docstrings and generate maintainable Sphinx API and
  concept documentation from the canonical public surface.
- [ ] Write the indexed notebook mini-course under `examples/tutorials/` and
  keep it aligned with the runnable Python learning path.
- [ ] Re-run every basic and advanced example, fast unit checks, relevant NRV
  campaigns, and supported CPU/GPU realistic benchmarks after cleanup. Refresh
  validation counts and retained evidence instead of carrying stale numbers.
- [ ] Perform the final pre-v1 API/export/package-data audit and remove anything
  still undocumented, duplicated, or outside the product boundary.

## Future Architecture And Product Work

### P20 - Lazy Runnable Plans And Distributed Runner

Axons, populations, simulations, protocols, sweeps, and studies should produce
immutable runnable plans. One runner executes one or many plans and owns all
eager work. This replaces the current dispatcher/execution architecture; it is
not a wrapper or a parallel optimized route.

- [ ] Use Graphify to audit `Axon`, `AxonPopulation`, `AxonSimulation`,
  protocols, studies, dispatch plans, preparation caches, runtime execution,
  results, inspection, and progress before defining the replacement.
- [ ] Define one backend-neutral immutable `RunnablePlan` contract plus
  composed plans for populations, numeric axes, sweeps, recruitment, and
  studies. Plans describe work and expected results but allocate nothing.
- [ ] Make all work outside the runner lazy. Defer numerical materialization,
  grouping, signatures, generated-code loading, device placement, compilation,
  scheduling, and result allocation until execution or explicit
  `estimate()`/`inspect()`.
- [ ] Implement one runner that owns dependency ordering, compatible grouping,
  reusable prepared state, execution policy, cancellation, progress, failures,
  and deterministic canonical result assembly.
- [ ] Replace the current simulation/protocol dispatcher and execution routes
  once validated, then remove superseded builders, caches, wrappers, and call
  paths. Preserve `AxonSimulation(...).run()` unless an API change is discussed
  first.
- [ ] Add backend-neutral scheduling for local multi-GPU and HPC workers,
  including topology, memory budgets, affinity, cache locality, remote artifact
  retention, failure recovery, and reproducible placement.
- [ ] Support synchronous execution first. Retain asynchronous device overlap
  only for independent heterogeneous groups when it reduces device idle time
  and improves end-to-end runtime with bounded pending memory.
- [ ] Validate single and composed plans, mixed populations, native numeric
  axes, studies, cache invalidation, cancellation, CPU, single/multi-GPU, and an
  HPC smoke path. Benchmark fresh miss, structural reuse, dynamic operands,
  transfers, compilation, solve, result assembly, RSS, device memory, scheduler
  overhead, and Naxon=1024/4096 scaling.

### Full Recording And Membrane Introspection

- [ ] Benchmark dense/full Vm recording separately from VmRaster and define a
  justified chunk policy without changing the established VmRaster default
  globally.
- [ ] Extend the canonical `Recording` and result contracts so users can request
  every quantity declared by a membrane model: gates, ionic and aggregate
  currents, conductances, auxiliary states, and Markov occupancies. Derive the
  surface from the membrane contract; do not add model-specific recording paths.
- [ ] Define full, probe, downsampled, and bounded-retention policies before
  population-scale use. Report estimated host/device memory through
  `estimate()` and preserve one result model on CPU and GPU.
- [ ] Add didactic examples and CPU/GPU numerical tests for retained membrane
  values, including mixed HH/Markov compositions and overflow/retention cases.

### Independent Reference Validation

- [ ] Add an optional test-only NEURON mode that compiles and runs canonical
  reference MOD mechanisms for every retained membrane model. NEURON is an
  independent oracle, not a public runtime or package dependency.
- [ ] Pin upstream revisions and mechanism checksums, then compare parameters,
  initialization, clamp responses, cable behavior, gates, currents,
  conductances, auxiliary states, and Markov occupancies where available.

### Propagation And Conduction-Block Validation

The removed `ConductionBlock` definition must not return as inverted distal
activation. Public block results require an independent scientific campaign.

- [ ] Define propagation from one source probe to one or more targets with
  timing/order constraints and the statuses `propagated`, `blocked`,
  `not_initiated`, and `ambiguous`.
- [ ] Support proximal/distal classification for bidirectional, one-sided,
  local-only, no-initiation, and direct activation. Define repeated-spike
  matching separately with bounded state.
- [ ] Validate propagation and true block on CPU/GPU before exposing a public
  analysis or block-threshold study.
- [ ] Add a didactic KES/block example covering local activation, failed
  initiation, propagation, true block, and filtering.

### Studies, Optimization, And Persistence

Current threshold and recruitment protocols remain valid. This phase builds
composable study plans, retention, and persisted results above those protocols.

- [ ] Define study-level callable threshold/block-threshold curves, recruitment
  studies, conduction validation, parameter sweeps, reuse/retention policies,
  and canonical study results.
- [ ] Define final schemas, typed serialization, artifact provenance, and a
  persistence strategy that does not serialize backend/runtime internals.
- [ ] Evaluate optional optimization/search adapters for grid/random sampling
  and surrogate-guided search. Keep the study contract independent of
  scikit-learn and compare it with SciPy or dedicated optimization packages.

### External Integrations And Examples

- [ ] Continue NRV hardening only where its package contract is stable. Keep
  geometry construction in `examples/with_nrv` or benchmarks and avoid
  duplicating `axonscope.integrations.nrv` sampled-footprint behavior.
- [ ] Implement the CPU/NRV FEM-footprint workflow from
  `ideas/fem_axon_gpu_coupling_design.md` in integration examples/benchmarks,
  not the core solver. Separate FEM solve, first footprint, cached sampling,
  and AxonScope solve; keep field bases and embedding/projection ownership in
  the external geometry/FEM layer. Evaluate GPU FEM only after this path is
  measured.
- [ ] Add didactic multi-electrode and multi-stimulus examples, including a
  multipolar cuff over several externally defined fascicles using canonical
  sampled footprints.
- [ ] Evaluate Apple Metal through `jax-mps` as an optional experimental runtime
  target; require numerical and performance evidence before documenting it as
  supported.

### Candidate Membrane Families

Use Channelpedia only as a discovery index. Implementations require an original
publication or redistribution-compatible source with explicit provenance.

- [ ] Implement and validate an axonal Kv1.1/Kv1.2 heteromeric mechanism with
  canonical juxtaparanodal placement; do not duplicate existing delayed
  rectifiers as separate homomeric channels.
- [ ] Implement and validate a Kv7.2/Kv7.3 M-current for nodal and near-rest
  excitability.
- [ ] Evaluate HCN1/HCN2 or Cav2.2/Cav3.2 only if an isoform-specific campaign
  demonstrates behavior not represented by Tigerholm or Schild models.
- [ ] Extend the membrane contract for non-voltage inputs before considering
  sensory K2P channels. Keep receptor-terminal TRP channels outside scope until
  AxonScope explicitly adopts a validated terminal-transduction product scope.
- [ ] Add BK/SK, Kv3.4, or Kv4.3 only for a concrete axonal workflow that is not
  represented by the existing `KA` and `KCa` machinery.

### Deferred NumPy/SciPy Reference Runtime

This is a future deterministic debugging backend, not a JAX wrapper or the next
implementation phase. Any non-JAX backend must consume its own generated target
from the canonical membrane source contract.

- [ ] Reserve `axs.runtime.numpy` until it can use the same
  `AxonSimulation(...).run()`, `.estimate()`, and `.inspect()` lifecycle.
- [ ] Define a small v1 subset: single cable, intracellular current, sampled
  footprints, recording, observers, and selected membranes.
- [ ] Implement readable tridiagonal Crank-Nicolson with NumPy/SciPy and a
  NumPy-specific generated membrane target; never call into JAX.
- [ ] Validate Vm, activation, block, latency, thresholds, probes, retained
  membrane values, and model steps against JAX before adding public policy,
  examples, docs, estimates, or inspection records.
- [ ] Document it for tiny deterministic debugging and numerical regression,
  not population performance or GPU parity.

## Key References

- Product and architecture: `GUIDELINES.md`
- Working guide: `AGENTS.md`
- Benchmark surfaces and completed evidence: `benchmark/README.md`
- Validation policy: `docs/validation.md`
- Examples map: `examples/README.md`
- P18 model audit: `docs/architecture/p18_nrv_model_audit_2026_07_19.md`
- P11 closeout: `docs/architecture/p11_closeout_2026_07_12.md`
- P12 runtime cleanup:
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md`
- Historical checklist before the 2026-07-12 cleanup:
  `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`
