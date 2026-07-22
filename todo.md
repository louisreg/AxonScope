# AxonScope TODO

Living execution plan for AxonScope. `GUIDELINES.md` owns architecture and
product boundaries; source, tests, runnable examples, and fresh benchmark
artifacts own current behavior. Historical detail before the 2026-07-12
cleanup remains in `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.
Completed performance evidence and commands live in `benchmark/README.md`.

## Snapshot

Updated on 2026-07-22 after closing P18 membrane completion and validation.

- P7, P11, P12, the VmRaster part of P13, and P14-P18 are closed.
- The production runtime is JAX. CPU double-cable uses Thomas; CUDA
  double-cable uses tiled Thomas in Triton. CPU single-cable uses JAX
  tridiagonal solve; CUDA single-cable uses exact tiled Thomas in Triton.
- Recruitment uses one source population plus a native numeric amplitude axis.
  It does not build `Namplitude x Naxon` Python simulation objects.
- Compact activation, latency, spike-count, and bounded spike-time states are
  available. VmRaster remains the temporal reference where history is needed.
- The active phase is P19 pre-v1 convergence.
- P20 tracks the future lazy runnable-plan and distributed-runner architecture.
  It is not an unfinished P14 optimization: current structural preparation is
  below 3% of realistic warm `run_pool` time.

Latest fast validation:

```text
python -m compileall -q src tests/unit
pytest -q tests/unit --tb=short
858 passed, 1 skipped
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
- [x] Implement the Balbi et al. ModelDB 230137 Nav1.x family through one
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
  - [x] Complete the exact generated Markov runtime evidence gate before any
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
    - [x] Complete matrix-free validation before removing the retained dense
      update from benchmark/reference use.
      - [x] Validate the conserved matrix-free update against an independently
        materialized full `(I - dt Q)` solve on random valid 2/3/6/9-state
        transition graphs. Cover float32/float64 and `dt = 0.001, 0.0125,
        0.05, 0.1 ms`, including state non-negativity and probability
        conservation.
      - [x] Validate all Nav1.1-Nav1.9 generated updates over the clamp voltage
        range and the same dtype/`dt` matrix. Compare complete states, open
        probability, and current against the independently materialized dense
        update. The exact unit surface is in
        `tests/unit/membranes/test_kinetics.py` and
        `tests/unit/membranes/test_nav_isoforms.py` (87 tests in 22.39 s on
        the 2026-07-22 local CPU run).
      - [x] Regenerate the independent ModelDB 230137 reference comparison for
        the current runtime. The complete clamp artifact is
        `benchmark/results/p18_nav_voltage_clamp_matrix_free_20260722/` and
        contains the fresh NEURON reference plus the AxonScope comparison.
        Across all isoforms, worst-case NRMSE is 0.272% for I-V, 0.223% for
        G-V, 0.215% for availability, and 0.190% for recovery. The official
        ModelDB archive SHA-256 is
        `cc05f481e5bf2698bce37aa30758f0ad4970e16edec58243fb994d26aab9234d`.
      - [x] Define a canonical excitable Nav + potassium + leak composition,
        then validate spike waveform, threshold, velocity, and recruitment on
        full single- and double-cable simulations. Keep this a runtime
        validation composition until it has its own physiological reference;
        a Nav sodium channel plus arbitrary leak is not such a reference. The
        benchmark-only Nav1.6 + Borg KDR + leak composition now runs through
        the canonical public simulation and protocol routes in
        `benchmark/curves/nav_cable_validation.py`. The retained 2026-07-22
        local CPU artifact passes stable-control, distal-propagation, velocity,
        bounded-threshold, and monotone-recruitment gates for both a full
        single cable and node-localized double cable; details and the artifact
        checksum are in `benchmark/README.md`.
    - [x] Inspect HLO and executable-cache replay for each retained shape. Keep
      topology and immutable parameter-derived constants compile-time static,
      but do not add rate tables while direct rate evaluation remains below 5%
      of the integrated update. Confirm fresh-process hits in the existing JAX
      persistent cache. `benchmark/membrane_temporal.py` now captures the exact
      stateful recording JIT and runs miss/exact/dynamic-value processes against
      one shared cache per shape. All 12 passive/HH/Nav1.6/mixed single,
      double-uniform, and double-node-localized CPU shapes preserve StableHLO,
      exact replay checksums, and create zero new XLA files on replay; compile
      speedups are 8.50x-15.44x. HLO confirms no gather in uniform layouts and
      exactly three in node-localized double layouts. CPU evidence and command
      are retained in `benchmark/README.md`; GPU HLO remains a separate
      backend-specific P100 evidence run, not a blocker for the generic cache
      contract.
    - [x] Formalize the full per-step operator before considering another
      solver. `Naxon` is an independent batch/direct-sum axis; each axon is a
      block-tridiagonal cable operator coupled to local finite-state blocks at
      its compartments. Demonstrate that with rates frozen at known `V`, the
      current matrix-free local update followed by Thomas is algebraically the
      block elimination of this global triangular system, so materializing one
      sparse `(Naxon * Nx * (1 + S))` matrix would add overhead without coupling
      information. The derivation is retained in
      `docs/architecture/p18_full_step_operator.md`; the executable single- and
      double-cable proof in `benchmark/analysis/full_step_operator.py` agrees
      with an assembled direct-sum solve to `3.33e-16` or better and finds zero
      off-axon nonzeros. At `Naxon=4096`, `Nx=201`, and `S=6`, conservative CSR
      storage is estimated at 304.58/376.75 MiB for single/double cable versus
      69.09/81.62 MiB for the production-oriented arrays.
    - [x] Do not prototype stronger voltage-state coupling without a numerical
      failure that requires it. Current temporal validation does not provide
      that trigger, and the frozen-operator proof shows that the canonical
      local update plus Thomas is already exact block elimination. A future
      coupled prototype, if justified by new evidence, must use generated local
      Jacobians and a per-compartment Schur complement for complete HH-like,
      Markov, mixed, and auxiliary-state programs; it must retain scalar/2x2
      Thomas, never couple independent axons, and replace the split path only
      after broader numerical validation and an end-to-end measured benefit.
  - [x] Evaluate and reject generic active-site membrane-state compaction before
    table or custom-kernel work. The realistic localized layout wastes 90.5% of
    dense dynamic-state bytes, but a canonical gather/update/scatter prototype
    changed P100 warm time by only 0.98x-1.02x for HH, Nav1.6, and mixed
    HH+Markov at Naxon 1024/4096 while usually increasing cold compilation.
    Per-step projection traffic cancels the smaller kinetic update, so commit
    `f67987d` was reverted by `ab89c42`; no parallel compact path remains.
  - [x] Audit parameter batching for multiple isoforms and mutants sharing one
    generated source/topology. Reuse the existing membrane row plan and
    structural signatures; avoid one dispatch group or executable per parameter
    set when parameters can be carried as a numeric row axis. Test mixed Nav1.x
    populations, cache identity, compile count, and numerical equivalence. The
    canonical single- and double-cable scans now carry only parameters that vary
    between rows; Nav1.1/Nav1.6 dense and VmRaster routes match scalar references,
    and all nine isoforms use one dispatch group and one generated program.
    Naxon=900 local and P100 evidence is retained under
    `benchmark/results/p18_parameter_batching_local_n900_reuse/` and Kaggle runs
    `axs-p18-parameter-batch-ef404fa` / `axs-p18-parameter-batch-long-ef404fa`.
    On the 100-step P100 double-cable A/B, dynamic isoform parameters cost
    34.9 ms warm versus 21.5 ms for homogeneous Nav1.6; this is execution work,
    not fragmentation (`runtime.prepare` remains about 0.1 ms warm).
  - [x] Evaluate and reject a voltage-tabulated transition operator as a
    benchmark-only candidate after the exact-path profiles above.
    - [x] Generate `M(V, dt) = exp(dt Q(V))` and stationary states from the same
      compiled kinetic contract; key any artifact by source/topology,
      parameters, temperature, `dt`, dtype, voltage grid, and compiler version.
      Do not introduce SciPy or table configuration into the simulation API.
      `benchmark/kinetic_transition_tables.py` keeps this benchmark-only and
      also generates the canonical implicit `(I - dt Q)^-1` operator so a
      temporal-scheme change is not confused with interpolation error.
    - [x] Compare nearest/linear interpolation and multiple voltage spacings
      against the exact matrix-free update over one-step states, long voltage
      trajectories, and cost. Stop before clamp, spike, threshold, velocity,
      and recruitment campaigns if the isolated candidate cannot meet the
      retention threshold. Linear interpolation must preserve stochasticity
      and errors must be reported for complete states.
      - [x] Complete the one-step, 300-step trajectory, stochasticity, and local
        CPU micro-cost gate at 204,800 Nav1.6 sites for 0.25/0.5/1.0 mV grids.
        Linear implicit lookup has `7.75e-7` to `1.16e-5` one-step error but is
        0.82x-0.86x as fast as the exact update; nearest lookup is roughly even
        but less accurate. Exponential lookup differs by about `1.35e-2` even
        on the finest grid because it changes the canonical time integrator.
      - [x] Run the paired micro gate on P100. Accurate linear implicit lookup
        is 0.82x-1.02x the exact update. Nearest reaches 1.24x-1.37x but has
        `2.91e-4` to `1.17e-3` one-step state error; even its best result implies
        only about 1.20x end-to-end at the measured 62% membrane fraction.
        Run `axs-p18-rate-table-883c15d` therefore fails the 1.3x integrated
        plausibility gate, so broader physiological validation is deliberately
        not run.
    - [x] Reject the generated table implementation: no numerically acceptable
      variant improves the isolated P100 workload, and the faster nearest
      variant cannot clear the integrated retention threshold. Keep the exact
      matrix-free generated update as the only temporal implementation; no
      runtime table path or fallback hierarchy was added.
  - [x] Retain a generated Triton local update after the final integrated GPU
    profile and same-shape A/B cleared the 1.3x end-to-end gate. The compiler
    emits it model-agnostically from the same HH/Markov contract; no Nav-specific
    solver code or public backend selector was added. On P100 at Naxon 4096,
    warm speedup is 1.549x for homogeneous Nav1.6, 1.345x for all nine
    row-parametric isoforms, and 1.555x for mixed HH+Markov. Corresponding
    maximum center-Vm differences versus the same-commit JAX route after 100
    float32 steps are 0.00375 mV or less for Nav and 0.00724 mV for mixed.
    Compatible uniform double-cable CUDA layouts use the generated kernel;
    single-cable and node-localized heterogeneous layouts retain their
    bit-identical canonical route. Do not add projection traffic to force
    Triton into the localized path.
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
- [x] Finish the missing Gaines motor and sensory model families.
  - [x] Express motor and sensory node/internode mechanisms through shared
    source-backed membrane topologies and the canonical generated runtime.
  - [x] Assemble public `GainesMotor` and `GainesSensory` axons on the retained
    MRG-like double-cable geometry without a model-specific runtime path.
  - [x] Validate rates, currents, defaults, velocity versus diameter,
    intracellular Vm, and extracellular Vm/Vext against fresh NRV runs, and
    add runnable Gaines and MRG-plus-Markov advanced examples.
- [x] Add focused numerical references and runnable advanced examples for each
  retained public model and each public Nav1.x validation workflow.
  - [x] Map Passive, HH, Rattay-Aberham, Sundt, Tigerholm, Schild94/97,
    AxNode/MRG, Gaines motor/sensory, and Nav1.1-Nav1.9 to their retained
    numerical references and executable examples in the P18 audit.
  - [x] Add public examples for the previously undocumented Sundt, Tigerholm,
    Schild94, Schild97, and Nav1.2-Nav1.5/Nav1.7-Nav1.9 families.
  - [x] Keep I-V, G-V, availability, and recovery in the runnable ModelDB
    validation runner until AxonScope has a designed public voltage-clamp API;
    public examples use only the canonical cable simulation path.

### P19 - Pre-V1 Cleanup And Public Surface

- [ ] Audit local and remote Git branches, preserve any unmerged work that is
  still relevant, then delete every branch except `main`.
- [ ] Reorganize `src/`, especially Python modules still at package root, now
  that P18 has settled the remaining ownership boundaries.
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

### NEURON Reference Validation And Full Membrane Recording

- [ ] Add an optional test-only NEURON validation mode that compiles and runs
  the canonical reference MOD mechanisms for every retained AxonScope membrane
  model. Pin each upstream source revision and mechanism checksum, then compare
  initialization, clamp responses, cable simulations, and every available
  internal trajectory. NEURON remains an independent validation oracle, not a
  public execution runtime or package dependency.
- [ ] Extend the canonical public `Recording` and result contracts so users can
  request every quantity declared by a membrane model: gates such as `m` and
  `h`, ionic and aggregate currents, conductances, auxiliary states, and all
  Markov occupancy states. Derive this generically from the membrane contract;
  do not add model-specific recording paths.
- [ ] Define explicit full, probe, downsampled, and bounded-retention policies
  for membrane internals before enabling population-scale recording. Report
  estimated host/device memory through `estimate()` and preserve the single
  canonical result model on CPU and GPU.

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
