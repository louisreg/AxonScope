# AxonFleet TODO

Living execution plan for AxonFleet. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh validation or
benchmark artifacts own current behavior. Completed implementation detail lives
in the phase audits and `benchmark/README.md`, not in this checklist.

## Snapshot

Updated on 2026-07-23 during the P19 source and public-surface audit.

- P7, P11, P12, the VmRaster part of P13, and P14-P18 are closed.
- The production runtime is JAX. CPU double-cable uses Thomas; CUDA
  double-cable uses generated Triton tiled Thomas. CPU single-cable uses JAX
  tridiagonal solve; CUDA single-cable uses exact Triton tiled Thomas.
- Recruitment uses one source population plus a native numeric amplitude axis;
  it does not materialize `Namplitude x Naxon` Python simulation objects.
- Compact activation, latency, spike-count, bounded spike-time, and VmRaster
  observers are available. Dense recording remains a separate memory/performance
  concern.
- P19A, P19B, and P20 are closed. P19C is intentionally deferred; later
  phases remain future architecture or product expansion.

## Non-Negotiables

- AxonFleet is pre-release with one active user. Prefer direct convergence over
  compatibility shims, deprecated wrappers, or parallel old/new paths.
- Keep one concept, one public name, one execution path, and one canonical
  public result model.
- Promote an optimization by replacing the production route. Benchmark-only
  prototypes may coexist temporarily; remove rejected or replaced variants.
- Discuss changes to public results, serialization, progress output, or the
  canonical workflow before implementing them.
- Public orchestration enters JAX through `axonfleet.runtime.execution`.
- Public examples must use public APIs, never runtime or solver internals.
- External packages own nerve geometry, trajectories, world coordinates,
  electrode CAD, and FEM solves. AxonFleet owns intrinsic axon coordinates,
  sampled-footprint stimulation, cable/membrane execution, recording, and
  analysis.
- Document every public feature in a runnable example or remove it.
- Do not remove an unfinished task unless it is completed, explicitly rejected,
  or moved to a named tracking document.

## Active Roadmap

### P19A - Source And Public-Surface Convergence

- [x] Build one Graphify- and `vulture`-assisted inventory of every Python
  module, public export, function, and type. Record production, example,
  benchmark, and test-only call sites before deleting anything. Track the
  exhaustive decisions in
  `docs/architecture/p19_repository_audit_checklist.md`.
- [x] Audit concepts as well as symbols: identify responsibilities implemented
  more than once, historical variants with only marginal remaining use, and
  old abstractions that survive only through a handful of internal call sites.
  Converge each retained concept on one owner and one execution path.
- [x] Simplify and strengthen contracts while converging the source tree. Audit
  every intermediate record, payload, plan, conversion, and runtime boundary;
  merge equivalent representations, rewrite weak contracts where needed, and
  minimize orchestration layers between the public workflow and solver kernels.
- [x] Complete the first convergence slice: remove orphan compiler/dispatcher
  wrappers and the duplicate utility `vtrap`, retain only host-side diffusion
  construction and node-first double-cable assembly, share dispatch result
  validation across schedulers, and delete the duplicate archived test wrappers.
- [x] Complete the second convergence slice: remove the historical pre-P11
  benchmark source archive and unused protocol facade; shrink the package root
  to example-backed workflow names; remove transverse world coordinates and
  derivable row facts from prepared runtime cohorts; and delete the no-op
  execution-context options adapter.
- [x] Complete the third convergence slice: remove the standalone test-only
  analysis observer and redundant analysis aliases; audit every
  `AxonInstance` method; remove unused mutation/evaluation adapters; make
  `Section.periaxonal` the only cable-layer owner; and converge runtime input on
  one optional `ExtracellularStimulation` containing multiple drives.
- [x] Complete the fourth convergence slice: remove write-only analysis
  capability fields and ignored activation options; delete the duplicate
  preparation-signature API; move benchmark environment metadata to its
  runtime owner; collapse benchmark instrumentation onto one public facade;
  and remove unused profiler and global JAX-settings paths.
- [x] Remove the empty `SolverOptions` contract and its no-op cache-key and
  dispatcher/runtime plumbing. Solver choices belong to typed
  `ExecutionPolicy`; only add a future numerical-options contract when it has
  a real, example-backed option.
- [x] Converge activation onto one public definition: `Activation.detect()`
  returns a compact per-axon event, while `Activation.evaluate()` returns the
  structured analysis result. Remove `ActivationCriterion` and all protocol
  conversion paths instead of preserving two equivalent configurations.
- [x] Converge direct conduction-velocity calculation onto
  `ConductionVelocity.detect()` and structured evaluation onto
  `ConductionVelocity.evaluate()`. Keep `rasterize()` as the distinct
  multi-spike detector and use `result.recorded_axis` for position ownership.
- [x] Converge dispatcher, inspection, and estimation onto the actual
  batch-only contract. Remove the constant route predicate, unreachable scalar
  estimates, ignored observable-routing flag, redundant route metadata, and
  internal dispatcher facade; retain observer compatibility as its own check.
- [x] Converge threshold protocols on batched `find_threshold()`. Remove the
  sequential factory-based `find_activation_threshold()`, its standalone
  history/result/view family, and aliases that duplicated `Activation` and
  `PoolUpdate`.
- [x] Simplify result assembly and pool results: inline the sole post-hoc row
  adapter, remove duplicate `Any` typing modules, make `OutputPlan` the actual
  assembly contract, converge row selection on sequence indexing, and keep the
  recording manifest focused on requested and available public signals.
- [x] Converge stimulation on its executable contracts: remove the unsupported
  generic intracellular base, collection add/remove adapters, and unused
  drive/footprint aliases; retain one clamp type, one sampled multi-drive
  stimulation, and the full waveform construction/composition surface now
  demonstrated in `basic/02`.
- [x] Finish the analysis audit: retain one definition/event/result route per
  scientific concept and the distinct multi-spike rasterizer, while making
  missing-input exceptions, requirement records, typing protocols, and the
  raw activation detector private implementation details.
- [x] Converge descriptive axons on `Section -> Layout -> Axon`: remove
  flattened/runtime materialization and raw formulation helpers from the
  public facade, make `MRGLikeDoubleCableTemplate` the advanced MRG
  construction path, and remove duplicate layout, compartment, and node
  accessors while preserving every example-backed model workflow.
- [x] Converge the NRV integration on its example-backed bridge: external NRV
  objects produce one intrinsic AxonFleet population and sampled electrode
  footprints, which then produce the canonical stimulated population. Remove
  test-only slicing, concatenation, forwarding views, activation-comparison
  utilities, and low-level bridge helpers from the public surface.
- [x] Remove the reserved but non-executable `axs.runtime.numpy` path and its
  public target-construction vocabulary. Keep NumPy host materialization and
  membrane reference execution internal; expose a NumPy/SciPy runtime target
  only after the deferred backend supports the complete public lifecycle.
- [x] Converge the runtime-neutral recording and benchmark boundaries: lower a
  public recording through one function, retain group-aware padding as a
  separate runtime concern, and keep benchmark session state, array metadata,
  synchronization, and internal report records out of the public facade.
- [x] Converge the membrane authoring facade on models, equation/state
  declarations, composition/layout, and inspection commands. Remove the
  redundant built-ins facade, internal report-record exports, and the
  public-looking descriptor conversion while preserving the generic `section`
  extension point and compiler-supported equation vocabulary.
- [x] Audit the internal model compiler end to end. Retain one scalar equation
  graph for composition, validation, reference execution, hashing, and
  NumPy/JAX/Triton generation; remove its import facade, unused function/local
  schema, and unexercised symbolic shapes; reject stale serialized schemas and
  invalidate generated caches explicitly.
- [x] Converge membrane execution on generated artifacts: remove the
  test-only JAX Model IR interpreter and `JaxMembraneProgram.from_model_ir()`,
  make generated JAX/NumPy modules the only runtime path, and fix generic
  named-current dependencies in generated state-update functions.
- [x] Converge JAX stimulation input on one batch-payload route: remove the
  scalar `JaxStimulus` compiler, callable intracellular/extracellular wrappers,
  and stimulation-bearing `SolverRuntime`; prepare only static solver state and
  require kernels to receive explicit lowered inputs and drive-state metadata.
- [x] Remove the unvalidated double-cable shape-bucketing experiment and its
  duplicate public/kernel group plumbing. Retain one dispatch group through
  cohort preparation, lowering, memory estimation, and kernel execution.
- [x] Converge compact extracellular kernel inputs: precompute single-cable
  forcing footprints once during host lowering, remove the duplicate JAX
  computation/cache, and reject unsupported double-cable factorized payloads
  instead of silently materializing a second dense route inside the kernel.
- [x] Converge double-cable execution after policy resolution: GPU physical
  steps and JIT scans enter the selected tiled-Thomas route directly, while
  CPU scans call the sole scalar block-Thomas implementation directly. Remove
  test-only solver wrappers, repeated kernel validation, and private facades.
- [x] Converge generated membrane runtime contracts: cache each immutable typed
  contract on its generated module, retain only runtime-relevant typed
  metadata, remove tautological generated-path flags and test-only helpers,
  and narrow row-indexed backends to the operations they execute.
- [x] Converge JAX solver policy on choices that affect execution: remove the
  false public single-cable selector and its obsolete benchmark campaign,
  retain one platform-selected single-cable route, and require resolved engine
  descriptors to contain concrete solver routes without redundant permission
  bits or test-only cache-reset APIs.
- [x] Audit JAX recording end to end: retain one cached cohort-to-observer-plan
  lowering, bounded activation/latency/spike states, packed VmRaster retention,
  and one kernel-result finalization route. Move recording-mode selection to
  group orchestration and remove one-call trimming and internal export facades.
- [x] Close the JAX input audit: retain runtime-neutral compact payloads and
  planning plus one JAX lowering/materialization path. Make each cable
  `RuntimeInputContract` the sole owner of extracellular capabilities instead
  of copying them into every lowered payload, and remove internal export
  facades without changing dense, sparse, factorized, or zero routes.
- [x] Audit JAX benchmark support: retain distinct hot-path metadata, device
  memory snapshots, and profiler/inspection adapters behind
  `runtime.execution`. Confirm their production benchmark and estimate
  consumers, and remove package/module export facades without weakening
  benchmark evidence.
- [x] Close the JAX runtime audit: retain one enqueue/finalize group runner and
  one single-/double-cable kernel route. Make `DispatchGroup` the sole owner of
  diagnostic method labels, derive preparation and initial-previous behavior
  from group mode, remove dead double-cable state and internal facades, and
  narrow the explicit diffusion helper to the coefficients it uses.
- [x] Close the source utility audit: retain one Pint-compatible public unit
  boundary, the membrane source-unit vocabulary, shared value invariants, and
  progress reporting. Remove unused compatibility aliases, dead conversion
  wrappers, and internal export facades.
- [x] Remove code used only by tests, legacy wrappers, replaced slow routes,
  duplicate implementations, public names without runnable examples, and
  unused directories, documents, generated artifacts, or miscellaneous files.
  Do not preserve compatibility aliases or hidden fallbacks.
- [x] Reorganize `src/`, especially modules still at package root, after dead
  code is removed. Preserve ownership boundaries and avoid moves that only
  exchange one flat namespace for another.
- [x] Verify that public simulation, inspection, estimates, protocols,
  recording, results, and analyses each have one canonical contract and route
  through `axonfleet.runtime.execution` where runtime work is required.
- [x] Run a final whole-package Graphify audit after cleanup and confirm that
  no duplicate public concepts, model-specific solver branches, or stale
  runtime paths remain.

### P19B - Packaging, Caches, And Repository Hygiene

- [x] Rename the project to `AxonFleet` across the distribution, Python import
  namespace, source tree, docs/examples, metadata, cache/artifact identifiers,
  benchmark tooling, and repository-facing text. Make one direct pre-release
  cutover without an `axonscope` compatibility package or alias.
- [x] Clean `pyproject.toml`: remove obsolete extras and expose explicit,
  tested optional CUDA/Triton dependencies without making GPU packages a CPU
  installation requirement.
- [x] Converge artifact caching globally. Build only requested runtime targets;
  define first-call versus install-time behavior for built-ins; document
  inspect, clean, disable, invalidation, and retention policies; keep
  `.axonfleet_cache` deterministic. Evaluate additional time-chunk shape
  buckets only if observer padding is numerically inert and benchmarked.
- [x] Audit local and remote Git branches, preserve any still-relevant unmerged
  work, then delete every branch except `main`.

### P19C - Documentation And Pre-V1 Gate

- [ ] Complete public docstrings and generate maintainable Sphinx API and
  concept documentation from the canonical public surface.
- [ ] Write the indexed notebook mini-course under `examples/tutorials/` and
  keep it aligned with the runnable Python learning path.
- [ ] Prepare the CI/CD pipeline for supported Python versions, CPU unit tests,
  packaging/build checks, documentation, optional GPU validation, release
  artifacts, and protected publication credentials.
- [ ] Re-run every basic and advanced example, fast unit checks, relevant NRV
  campaigns, and supported CPU/GPU realistic benchmarks after cleanup. Refresh
  validation counts and retained evidence instead of carrying stale numbers.
- [ ] Perform the final pre-v1 API/export/package-data audit and remove anything
  still undocumented, duplicated, or outside the product boundary.

## Future Architecture And Product Work

### P20 - Lazy Runnable Plans And Local Runner

Axons, populations, simulations, protocols, sweeps, and studies should produce
immutable runnable plans. One runner executes one or many plans and owns all
eager work. This replaces the current dispatcher/execution architecture; it is
not a wrapper or a parallel optimized route.

- [x] Use Graphify to audit `Axon`, `AxonPopulation`, `AxonSimulation`,
  protocols, studies, dispatch plans, preparation caches, runtime execution,
  results, inspection, and progress before defining the replacement.
- [x] Complete the first replacement slice: add immutable `SimulationPlan` and
  composed `NumericAxisPlan`; make `Runner` own dispatch-plan reuse, execution
  context, runtime invocation, estimate/inspection entry, and canonical result
  assembly; remove the duplicate global dispatch-plan cache and the execution
  lifecycle from `simulation.py`. Record the migration audit in
  `docs/architecture/p20_runner_audit_2026_07_23.md`.
- [x] Complete the second replacement slice: add generic immutable `SweepPlan`;
  make pool and recruitment sweeps build it; move numeric-axis preparation,
  value chunking, progress, and observation assembly into `Runner`; remove the
  protocol-private numeric sweep plan and `AxonSimulation._run_numeric_axis()`.
- [x] Complete the third replacement slice: add immutable `ThresholdPlan`;
  make `find_threshold()` build it; move callable-bound resolution, per-row
  bisection, progress, solver execution, and `ThresholdCurve` assembly into
  `Runner`; remove the protocol-owned threshold execution loop.
- [x] Complete the fourth replacement slice: move synchronous and bounded
  asynchronous group scheduling into `Runner`; remove the raw `run_pool()`
  facade and `dispatcher/execution.py`; keep backend group execution solely
  behind `runtime.execution`.
- [x] Complete the fifth replacement slice: add immutable `PopulationPlan`;
  keep `AxonSimulation` construction descriptive; make `Runner` exclusively
  materialize and cache `AxonPopulation` for run, estimate, inspect, and
  explicit population access; reuse it across composed protocol execution.
- [x] Complete the sixth replacement slice: add a backend-neutral named task
  graph; replace `run_many()`'s tuple loop with stable dependency
  execution; retain completed results in structured fail-fast errors; add
  cooperative cancellation between tasks and protocol iterations while
  allowing in-flight kernels to finish safely.
- [x] Complete the seventh replacement slice: make `Runner.estimate()` and
  `Runner.inspect()` consume simulation, numeric-axis, sweep, threshold, and
  study plans; preserve plain-simulation reports; report one-execution peak
  memory separately from exact or bounded repeated simulation work.
- [x] Complete the backend-neutral immutable `RunnablePlan` family with study
  plans. Existing simulation, numeric-axis, pool sweep, recruitment, and
  threshold plans describe work without lowering solver groups or allocating
  runtime arrays.
- [x] Converge the provisional graph vocabulary into canonical `StudyTask`,
  `StudyPlan`, and `StudyResult`; execute named study DAGs through the same
  runner and leave retention, persistence, and result-dependent factories for
  their dedicated future contracts.
- [x] Make all work outside the runner lazy. Defer numerical materialization,
  grouping, signatures, generated-code loading, device placement, compilation,
  scheduling, and result allocation until execution or explicit
  `estimate()`/`inspect()`. Membrane `.params`, `.explain()`, and generated-code
  inspection are explicit compiler introspection and intentionally materialize
  their requested artifacts.
- [x] Implement one runner that owns dependency ordering, compatible grouping,
  reusable prepared state, execution policy, cancellation, progress, failures,
  and deterministic canonical result assembly.
- [x] Replace the current simulation/protocol dispatcher and execution routes
  once validated, then remove superseded builders, caches, wrappers, and call
  paths. Preserve `AxonSimulation(...).run()` unless an API change is discussed
  first.
- [x] Validate single and composed plans, mixed populations, native numeric
  axes, studies, cache invalidation, cancellation, local CPU, and single GPU.
  Benchmark fresh miss, structural reuse, dynamic operands, transfers,
  compilation, solve, result assembly, RSS, device memory, local scheduler
  overhead, and Naxon=1024/4096 scaling.
- [x] After benchmark acceptance, audit every runnable example and the public
  exports they teach. Remove or rewrite examples that expose superseded,
  ambiguous, redundant, or confusing workflows; add a concise didactic path
  for `SimulationPlan`, numeric/sweep plans, `StudyPlan`, shared `Runner`
  reuse, estimate/inspect, cancellation, and canonical results. Simplify the
  public API so examples teach one supported way to describe and execute each
  concept, without convenience paths that create a competing workflow.
- [x] After CPU/GPU validation and before closing P20, run a final Graphify
  convergence audit over plans, Runner, dispatcher lowering, preparation,
  runtime execution, caches, protocols, estimation, inspection, and results.
  Remove paths, names, builders, wrappers, fallbacks, or cache owners made
  unused or redundant by P20, then rerun the proportional validation gates.

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

### Distributed Runner, Dask, Multi-GPU, And HPC

This is a future improvement, not part of P20 completion. Do not implement or
freeze distributed contracts until a representative multi-GPU or HPC
environment is available for correctness, failure, and performance testing.

- [ ] Revisit the complete distributed-execution design from runnable-plan and
  artifact boundaries. Evaluate Dask Distributed as the primary candidate and
  compare only where necessary with MPI, Ray, or scheduler-native job arrays.
  Preserve one `Runner` and one plan/result vocabulary rather than adding a
  parallel public execution API.
- [ ] Define worker topology, CPU/GPU resources, memory budgets, affinity, data
  and compiled-cache locality, artifact transport/retention, deterministic
  placement, cancellation, retries, failure recovery, and reproducibility.
- [ ] Decide which objects cross process or cluster boundaries. Keep backend
  runtime state and device arrays local to workers; define serialization and
  provenance together with the study/persistence contracts.
- [ ] Evaluate bounded asynchronous overlap only for independent heterogeneous
  groups and only when it improves end-to-end runtime without unbounded pending
  host/device memory.
- [ ] Validate on real local multi-GPU and HPC infrastructure, including worker
  loss and retry behavior, before advertising support. Benchmark scheduler and
  transfer overhead, cold/warm cache locality, scaling efficiency, memory, and
  numerical equivalence against the local Runner.

### External Integrations And Examples

- [ ] Continue NRV hardening only where its package contract is stable. Keep
  geometry construction in `examples/with_nrv` or benchmarks and avoid
  duplicating `axonfleet.integrations.nrv` sampled-footprint behavior.
- [ ] Implement the CPU/NRV FEM-footprint workflow from
  `ideas/fem_axon_gpu_coupling_design.md` in integration examples/benchmarks,
  not the core solver. Separate FEM solve, first footprint, cached sampling,
  and AxonFleet solve; keep field bases and embedding/projection ownership in
  the external geometry/FEM layer. Evaluate GPU FEM only after this path is
  measured.
- [ ] Add multi-electrode and multi-stimulus examples, including a multipolar
  cuff with several externally defined fascicles using canonical sampled
  footprints.
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
  AxonFleet explicitly adopts a validated terminal-transduction product scope.
- [ ] Add BK/SK, Kv3.4, or Kv4.3 only for a concrete axonal workflow that is not
  represented by the existing `KA` and `KCa` machinery.

### Deferred NumPy/SciPy Reference Runtime

This is a future deterministic debugging backend, not a JAX wrapper or the next
implementation phase. Any non-JAX backend must consume its own generated target
from the canonical membrane source contract.

- [ ] Add an `axs.runtime.numpy` public target only after it supports the same
  `AxonSimulation(...).run()`, `.estimate()`, and `.inspect()` lifecycle; do
  not reserve a non-executable public name beforehand.
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
- P19 repository audit:
  `docs/architecture/p19_repository_audit_checklist.md`
