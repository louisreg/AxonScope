# AxonScope TODO

Living execution plan for AxonScope. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh benchmark
artifacts own current behavior. Historical detail before the 2026-07-12
cleanup remains in `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.
Completed performance evidence and commands live in `benchmark/README.md`.

## Snapshot

Updated on 2026-07-20 during the P18 Nav1.x implementation and validation.

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
788 passed, 1 skipped
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

- [x] Complete the existing-model NRV fidelity audit recorded in
  `docs/architecture/p18_nrv_model_audit_2026_07_19.md`.
  - [x] Pin the official NRV source revision and map every retained built-in to
    its wrapper and MOD mechanisms.
  - [x] Align HH, Rattay-Aberham, and Sundt effective defaults with NRV.
  - [x] Repair the canonical stateful single-cable path and validate all seven
    retained active models with the NRV velocity campaign.
  - [x] Restore focused Vm, current, gate, and state trajectory comparisons;
    do not infer fine observable equivalence from propagation alone.
- [ ] Implement the Balbi et al. ModelDB 230137 Nav1.x family through one
  model-agnostic Markov membrane contract and the generated runtime.
  - [x] Define one unified Python membrane-authoring contract before adding the
    Markov lowering. It must compose independent HH-like gates, coupled
    Markov-like occupancies, and general auxiliary state updates within the
    same model; these forms must not introduce separate model base classes,
    compilers, runtime backends, or execution routes.
  - [x] Prove that the final contract can naturally express every retained
    built-in: alpha/beta and steady-state/time-constant gates, passive and
    composed currents, unit-bearing defaults, aliases, derived parameters,
    Q10 and piecewise equations, custom initialization, concentration and pump
    states, dynamic reversal potentials, solver linearization/corrections,
    diagnostics, observables, and prepare/finalize step semantics.
  - [x] Keep the existing HH and auxiliary-state vocabulary as parts of the
    same canonical language, add Markov occupancies without a second model
    base/compiler/runtime route, and update compiler tests, inspection,
    generated-code tests, documentation, and the custom-authoring example.
    No replaced decorator or schema path remains to migrate.
  - [x] Add a compiler-level coupled-state transition representation and a
    stable generated JAX update that preserves non-negative occupancies and
    unit total probability; do not encode Nav-specific state logic in the
    solver or approximate NEURON's sparse kinetic solve with scalar Euler
    updates.
  - [x] Establish and optimize the local CPU baseline for per-node finite-state
    solves. The generated update now consumes transition rates directly,
    eliminates one state for probability-conserving blocks, applies a stable
    correction only when the reconstructed state would be negative, and never
    materializes `[N, S, S]` in the temporal path. At 204,800 kinetic nodes, a
    same-process Nav1.6 A/B measures about 15.0 ms for the retained dense 5x5
    reference versus 6.9 ms matrix-free, down from about 114 ms for the first
    generic route. The nine isoforms remain finite, non-negative, and within
    1.2e-7 of unit probability after 100,000 float32 updates over -120..60 mV.
    Retained evidence lives in
    `benchmark/results/p18_membrane_kinetics_local/`.
  - [x] Deduplicate uniform stationary initialization before device execution.
    At 204,800 identical Nav1.6 sites, one host solve plus broadcast takes about
    1.19 ms versus 485 ms for repeated identical solves, with identical values.
  - [x] Critically review
    `ideas/axonscope_markov_nav_runtime_agent_guidelines.md`. Retain its
    compiled-local-kinetics, active-site compaction, generated component update,
    validation, and evidence-gated custom-kernel principles. Do not add a
    parallel public `MarkovChannel` hierarchy, Markov-specific dispatch API,
    three selectable public Markov backends, explicit sparse Euler updates,
    separate cable-coupling policies, or Markov-specific solver observers.
    The existing `Model`/`@markov`/`Occupancy`, generated runtime, dispatcher,
    recording contract, and canonical result model remain the only paths.
  - [ ] Complete the exact generated Markov runtime evidence gate before any
    custom GPU kernel or approximate production update.
    - [x] Add integrated temporal profiles for uniform and section-localized
      kinetics in both single- and double-cable simulations. Cover CPU locally
      and P100 GPU, cold and warm execution, Naxon scaling, state memory, and
      the fraction of runtime spent in membrane update versus cable solve.
      `benchmark/membrane_temporal.py` now defines the canonical HH-only,
      Nav1.6, and mixed HH+Markov workload matrix through
      `AxonSimulation.run()`, plus a passive cable-floor ablation. The first
      P100 Naxon=1024/4096 campaign exposed and fixed generic double-cable GPU
      batching and optional generated-Triton capability bugs. The matched
      passive CPU/P100 ablation now estimates the integrated membrane fraction:
      at Naxon 4096 it is 54% for uniform Nav1.6 and 62% for uniform mixed
      HH+Nav1.6 double cable, falling to 24% and 40% when active membranes are
      limited to 21/221 compartments. That localized layout still carries
      90.5% inactive dense state bytes, making generic active-site compaction
      the next evidence-backed runtime target. Evidence and commands are in
      `benchmark/README.md`.
    - [ ] Validate the matrix-free conserved update against the retained dense
      solve for random valid transition graphs, all Nav1.x voltage-clamp
      protocols, spike waveform, threshold, velocity, and recruitment. Test
      float32/float64 and the supported `dt` range before removing the dense
      update from benchmark/reference use.
    - [ ] Inspect HLO and executable-cache replay for each retained shape. Keep
      topology and immutable parameter-derived constants compile-time static,
      but do not add rate tables while direct rate evaluation remains below 5%
      of the integrated update. Confirm fresh-process hits in the existing JAX
      persistent cache.
    - [ ] Formalize the full per-step operator before considering another
      solver. `Naxon` is an independent batch/direct-sum axis; each axon is a
      block-tridiagonal cable operator coupled to local finite-state blocks at
      its compartments. Demonstrate that with rates frozen at known `V`, the
      current matrix-free local update followed by Thomas is algebraically the
      block elimination of this global triangular system, so materializing one
      sparse `(Naxon * Nx * (1 + S))` matrix would add overhead without coupling
      information.
    - [ ] If temporal validation shows that stronger voltage-state coupling is
      needed, prototype it generically through generated local Jacobians and a
      per-compartment Schur complement. The local unknown block is the complete
      generated membrane program and may contain HH-like gates, one or more
      conserved Markov blocks, and auxiliary states in any supported
      composition; it must never assume a Markov-only cable. Eliminate all local
      membrane unknowns, contribute their effective diagonal/RHS terms to the
      existing scalar single-cable or 2x2 block double-cable system, run the
      retained CPU/Triton Thomas solver, then back-substitute local states. Do
      not couple independent axons or replace Thomas with a generic global
      sparse solve. Retain only with improved numerical behavior and a measured
      end-to-end benefit over the canonical split update.
    - [ ] Validate any global-operator/Schur implementation through the same
      generated entry point for HH-only, Markov-only, mixed HH+Markov, and
      multiple-kinetic-block membranes. Cover uniform and section-localized
      layouts in both single- and double-cable formulations, and compare Vm,
      every dynamic state, currents, threshold, and velocity against the
      canonical split path at multiple `dt` values.
  - [ ] Evaluate generic active-site membrane-state compaction before table or
    custom-kernel work.
    - [ ] Quantify wasted state bytes and updates in current heterogeneous and
      gated/leak layouts at realistic node/internode ratios and Naxon
      1024/4096. The current dense `[batch, Nx, n_gates_max]` route is the
      baseline, not a second path to preserve.
    - [ ] If the audit predicts a material end-to-end gain, replace dense
      inactive gate storage with a runtime-generated active-site projection for
      any section-localized membrane model, not a Nav- or MRG-specific backend.
      Gather active Vm, update compact state, and project conductance/current
      contributions back into the canonical cable arrays. A compartment may
      carry HH-like and Markov components together; compaction must preserve
      their generated composition and may use distinct active-site projections
      for distinct dynamic state blocks when that is measurably beneficial.
    - [ ] Benchmark state-last, node-first, and compact active-site layouts on
      CPU/P100 rather than prescribing state-major layout a priori. Validate
      homogeneous/unmyelinated no-regression and identical single-/double-cable
      results before retaining compaction.
  - [ ] Audit parameter batching for multiple isoforms and mutants sharing one
    generated source/topology. Reuse the existing membrane row plan and
    structural signatures; avoid one dispatch group or executable per parameter
    set when parameters can be carried as a numeric row axis. Test mixed Nav1.x
    populations, cache identity, compile count, and numerical equivalence.
  - [ ] Prototype a voltage-tabulated transition operator only as a benchmark
    candidate after the exact-path profiles above.
    - [ ] Generate `M(V, dt) = exp(dt Q(V))` and stationary states from the same
      compiled kinetic contract; key any artifact by source/topology,
      parameters, temperature, `dt`, dtype, voltage grid, and compiler version.
      Do not introduce SciPy or table configuration into the simulation API.
    - [ ] Compare nearest/linear interpolation and multiple voltage spacings
      against the exact matrix-free update over one-step states, long voltage
      trajectories, all clamp surfaces, spike waveform, threshold, velocity,
      and recruitment. Linear interpolation must preserve stochasticity and
      errors must be reported for states, open probability, and current.
    - [ ] Retain a generated table implementation only if it is numerically
      accepted and improves the integrated workload by at least 1.3x without
      pathological memory traffic or batch fragmentation. It must replace the
      exact temporal implementation selected for that runtime policy rather
      than create a broad fallback hierarchy; otherwise reject it explicitly.
  - [ ] Consider Pallas/Triton or CUDA only if the final integrated GPU profile
    still spends more than 20% in the generated Markov update and a same-shape
    A/B demonstrates at least 1.3x end-to-end improvement.
  - [x] Represent the shared six-state `C1/C2/O1/O2/I1/I2` topology once and
    provide the complete Nav1.1 through Nav1.9 parameter sets without nine
    duplicated execution paths.
  - [x] Reproduce the ModelDB voltage-clamp validation surface: I-V curves,
    normalized conductance-voltage curves, steady-state availability, and
    recovery from fast inactivation.
    - [x] Generate all four surfaces for all nine isoforms through the canonical
      generated membrane update.
    - [x] Validate I-V and normalized G-V against a fresh independent run of the
      official ModelDB MOD mechanisms.
    - [x] Validate availability and recovery against fresh independent ModelDB
      runs before treating those curves as numerical references.
  - [x] Keep selectable isoforms generic and composable through `Composite`,
    axon layouts, and the same generated runtime. Use NRV's Nav1.1/Nav1.6
    MRG-node substitution only as an external validation case; do not copy it
    into AxonScope as a special MRG preset or execution path.
- [ ] Finish the missing Gaines motor and sensory model families.
- [ ] Add focused numerical references and runnable advanced examples for each
  retained public model and each public Nav1.x validation workflow.

### P19 - Pre-V1 Cleanup And Public Surface

- [ ] Audit local and remote Git branches, preserve any unmerged work that is
  still relevant, then delete every branch except `main`.
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

### P18 Follow-Up - Candidate Membrane Families

Use Channelpedia as a discovery and validation index, not as the distributable
source of model equations. Its data/model license is non-commercial,
non-sublicensable, and revocable, so every retained implementation must be
reconstructed from the original publication or another redistribution-compatible
source with explicit provenance.

- [ ] Implement a native axonal Kv1.1/Kv1.2 heteromeric mechanism, validate its
  voltage-clamp behavior, and support canonical juxtaparanodal placement. Do
  not add two independent homomeric channels or another generic delayed
  rectifier that duplicates existing `AxNode`, Tigerholm, or Schild behavior.
- [ ] Implement a Kv7.2/Kv7.3 M-current mechanism for nodal and near-rest
  excitability, including voltage-clamp, threshold, repetitive-firing, and
  conduction validation.
- [ ] Evaluate isoform-specific HCN1/HCN2 only where it demonstrates behavior
  not represented by Tigerholm's existing fast/slow HCN mechanism.
- [ ] Defer sensory K2P channels (`TRESK`, `TREK1`, `TREK2`, and `TRAAK`) until
  the membrane contract can represent their required non-voltage inputs such
  as temperature, pH, mechanical drive, lipids, or intracellular signalling.
- [ ] Evaluate Cav2.2 and Cav3.2 only through an isoform-specific calcium
  validation campaign that justifies replacing or extending the N-type,
  T-type, and calcium-state machinery already present in Schild94/Schild97.
- [ ] Keep BK/SK, Kv3.4, and Kv4.3 as low-priority candidates until a concrete
  axonal workflow demonstrates behavior missing from the existing `KA` and
  `KCa` mechanisms.
- [ ] Keep terminal sensory-transduction channels such as TRPV1, TRPA1, and
  TRPM8 outside the core membrane roadmap unless AxonScope explicitly gains a
  validated receptor-terminal simulation scope.

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

## Unsorted

- [ ] Add didactic multi-electrode and multi-stimulus examples, including a
  multipolar cuff driving several externally defined fascicles through the
  canonical sampled-footprint stimulation path.

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
