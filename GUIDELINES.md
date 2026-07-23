# AxonFleet Architecture Guidelines

Snapshot: 2026-07-10.

This is the consolidated architecture reference for AxonFleet. It is normative
for project direction, refactors, public API shape, examples, and cleanup
decisions. The source tree, tests, examples, and benchmark reports remain the
implementation source of truth for current behavior.

Use this document to answer three questions:

1. What is AxonFleet?
2. Which concepts are public and stable?
3. Where must solver/runtime/backend code live?

Keep this file concise. Detailed experiment notes belong in `benchmark/`,
`docs/`, or `ideas/`; active work belongs in `todo.md`; operational agent notes
belong in `AGENTS.md`.

---

# 1. Current Status

AxonFleet is still pre-release and currently has one active user. Prefer the
clean final design over retrocompatibility, downstream migration paths,
deprecated wrappers, argument aliases, or prototype APIs. Delete superseded
paths rather than preserving shims.

Current focus after the P9 closeout:

- keep only retained solver routes in active runtime code;
- preserve the public/runtime/runtime boundary;
- prioritize a clean membrane/model/compiler surface and current JAX solver
  optimization before starting a new NumPy/SciPy solver runtime;
- converge simulation, estimation, inspection, results, analyses, and plots on
  one public workflow;
- flatten public examples and docs against the public API, then make every
  public option visible in examples or delete/archive it;
- make planning, batch/dispatch, preparation, lowering, execution, and result
  assembly inspectable;
- keep benchmark/profiling experiments out of public tutorials while making the
  supported benchmark surface documented and reproducible;
- keep membrane semantics, units, states, parameters, gates, channels, and
  observables owned by AxonFleet rather than by JAX-shaped backend
  implementations;
- keep membrane authoring user-facing: users define membrane models and
  equations in ordinary Python source, not intermediate representations or
  builder DSLs. Model IR is internal compiler vocabulary and must not become
  required user knowledge.
- make that single Python membrane language cover three composable forms of
  dynamics: independent HH-like gates expressed as alpha/beta or steady-state
  and time-constant equations; coupled Markov-like occupancies expressed as
  named states and transitions; and general auxiliary state updates for
  concentrations, pumps, buffers, dynamic reversal potentials, and other
  stateful mechanisms. One membrane may freely combine all three forms with
  passive terms, currents, observables, diagnostics, and custom initialization.
  These are source-level kinetic forms, not separate model classes, compilers,
  runtime backends, or execution paths.
- require the final membrane language to express every retained built-in model
  naturally, including unit-bearing and derived parameters, aliases,
  temperature scaling, piecewise equations, current linearization, explicit
  solver corrections, and prepare/finalize state semantics. When that
  authoring surface changes, migrate built-ins and public examples directly and
  remove the superseded vocabulary instead of retaining compatibility paths.
- treat Model IR as a compiler representation, not the long-term numerical
  runtime source. Runtime-specific generated modules are content-addressed per
  model and target, generated lazily, and own every model-specific runtime fact:
  numerical functions, canonical parameter metadata, state and gate policies,
  observables, diagnostics, initialization, and stateful step hooks. A cached
  runtime artifact must be reusable without regenerating an already valid
  target file or rebuilding its execution contract from Model IR; Model IR may
  remain beside it for validation, inspection, composition, and reference
  execution.
- keep built-in membrane model truth in `src/axonfleet/membranes/models/`: each
  model file owns its equations, unit-bearing defaults, public aliases, and
  derived parameter logic. Public constructors, `model_ir`, solvers, and
  backends must not duplicate model-family facts.
- keep built-in axon templates out of model-parameter ownership. They may keep
  ergonomic model-specific keyword arguments, but those arguments are forwarded
  only to the membrane descriptor/source compiler. Axon templates must not own
  model defaults, aliases, equations, derived parameter formulas, or unit
  conversion contracts.
- treat a future NumPy/SciPy runtime as a bonus reference/debug backend, not the
  next product priority. Do not expose `axs.runtime.numpy` until it is executable
  through the complete simulation, estimate, and inspection lifecycle.

## 1.1 Phase Snapshot

| Phase | Status | Current surface |
| --- | --- | --- |
| 0 - Guardrails and baselines | Done | Architecture tests, public API cleanup checks, import-boundary checks, scientific baselines. |
| 1 - Object model | Done | `AxonInstance`, root `AxonSimulation`, `AxonPopulation`, descriptive `Axon`. |
| 2 - Typed and extracellular contracts | Done | Typed signals, position selectors, cable formulation, identifiers, footprint/drive/stimulation objects. |
| 2.5 - Hotpath evidence | Done | Opt-in benchmark spans, hotpath workload catalog, Colab GPU workflow. |
| 3 - Planning and preparation | Done for current JAX path | Preparation signatures, prepared cohorts, footprint-oriented preparation. |
| 4 - JAX isolation | Done for current public execution path | Public entry points route through backend execution facades; dispatcher group execution does not import the JAX group runner directly. |
| 5 - Canonical simulation results | Done | `AxonSimulationResult`, `AxonResultView`, `RecordingManifest`, `RecordedSignal`, internal dense storage blocks. |
| 6 - Analyses | Done for current public layer | Real `axs.analysis`, definitions, requirements, statuses, reports, post-hoc helpers. |
| 7 - Performance evidence | Done for current evidence layer | Estimates, hotpath metadata, memory pressure reporting, footprint reuse evidence. |
| 7.5 - Generic solver-side observers | Superseded | Broad observer path removed from active direction; `PeakVoltage` remains post-hoc. |
| 7.6.1-7.6.2 - Hotpath/memory cleanup | Done for evidence layer | Sparse/zero inputs, compact observer outputs, runtime caches, chunking, profiler traces. |
| 7.6.3 - Exact cable GPU solvers | Closed for current evidence | CPU single-cable keeps JAX tridiagonal and CPU double-cable is Thomas-only; CUDA single- and double-cable execution use the retained exact Triton tiled-Thomas kernels behind typed execution policy. |
| 7.6.5 - Execution envelope and forcing | Done for current JAX lowering cleanup | Prepare/dispatch/probe-plan rebuilds are reduced, `Vext`/`Iinj` lowering is centralized, and retained dense forcing is explicit backend fallback behavior. |
| 7.6.6 - GPU dispatch scheduling | Planned | Memory-aware bucketing/coalescing before optional async scheduling. |
| 7.6.7 - VmRaster redesign | Done for current strict path | One threshold-style VmRaster primitive, packed in solver as `observations["vm_raster"]`, decoded post-hoc. |
| 7.7 - Solver surface stabilization | Done for current public surface | Archive standby candidates, align `solvers/` with runtime boundary, keep factorized Vext internal. |
| 7.8 - Runtime policy and inspection | Done for current JAX path | `ExecutionPolicy` controls JAX device/runtime and participates in cache identity; `AxonSimulation.inspect()` explains planning, dispatch, preparation, lowering, kernel, and result assembly through the runtime boundary. |
| 7.9 - Runtime-agnostic DSL and Model IR | Done for current P7 scope | Internally, `axonfleet.model_ir` owns covered built-in membrane semantics with NumPy interpreter tests and backend-neutral membrane programs; JAX consumes `JaxMembraneProgram` directly without per-model runtime adapters. Publicly, users see `axonfleet.membranes` membrane models and plain-Python equation sources under `membranes/models/`. The historical `icm/` and `channel_models/` packages have been removed from the active package. A future NumPy/SciPy reference solver remains deferred and has no public runtime target. |
| 8 - NumPy/SciPy reference solver runtime | Deferred bonus | Future scalar/tiny-simulation reference solver using Model IR semantics and tridiagonal Crank-Nicolson primitives. Not a JAX-backed compatibility route, and not the next implementation priority. |
| 9 - Cold-run/runtime benchmark closeout | Done | `cold_run_micro`, normalized scalar/batch spans, explicit hotpath chunk controls, and decisions to park larger optimization campaigns until realistic evidence exists. |
| 10 - Model/compiler surface cleanup | Active next | Flatten membrane helper/diagnostic/explain surfaces, harden compiler/cache identity, and prepare recording-aware pruning/fusion contracts before deeper solver work. |
| 11 - Realistic JAX solver benchmarking and optimization | Active next | Build realistic workflow evidence first, then optimize the current JAX preparation/lowering/kernel/result paths with validation gates. |
| 12 - Studies, serialization, integration | Not started | Callable studies, reuse policies, retention policies, schemas, HPC, FEM/NRV integration. |

## 1.2 Active Gaps

- Public execution uses one canonical workflow for one axon or 10,000 axons:
  construct `AxonSimulation(...)`, then call `.run()`.
- `Recording.to_plan()` now produces a runtime-neutral `RecordingPlan`; the JAX
  runtime lowers that plan to batch-kernel options. The remaining work is to
  broaden validation and move more result/observer boundaries out of solver
  modules.
- VmRaster output remains exposed as `observations["vm_raster"]`, with
  `VmRasterResult` and CPU unpacking under `results`. Solver/runtime code owns
  only plan lowering and packed bit updates.
- Fixed-step time-grid validation lives in backend-neutral
  `axonfleet.runtime.timebase`.
  Public planning, estimation, and inspection code must not import concrete
  `axonfleet.runtime.jax.*` helpers directly. Runtime-specific estimate and
  inspection facts route through `axonfleet.runtime.execution`, which delegates
  to runtime-owned support under `runtime/jax/benchmarking/`
  without forcing device transfers.
- `ExecutionPolicy` now resolves JAX device requests and validates uniform
  precision. Runtime, device, and precision policy participate in JAX batch
  runtime cache identity; runtime execution still rejects implicit casting
  instead of rebuilding models.
- Passive, HH, and Rattay-Aberham membrane semantics run through Model IR with
  a NumPy model-step oracle and JAX lowering, but a full NumPy/SciPy
  cable-solver runtime is a separate bonus phase documented in `todo.md`. Its
  public runtime target must appear only once that backend is executable.
- `estimate()` and `inspect()` must follow the same public workflow as
  execution. Prefer `AxonSimulation(...).estimate()` and
  `AxonSimulation(...).inspect()` over a separate root helper.
- Factorized `Vext`/`Iinj` is the active internal direction for compatible
  static-footprint or compact-source rows. Keep it behind equivalence tests and
  benchmark evidence, and remove dense internal routes when the factorized
  route covers the behavior.
- Runtime dispatch batches rows by compatible runtime shape and membrane
  structure plus the same temporal waveform compatibility signature. Diameters,
  sampled extracellular footprints, and row amplitude scales may vary inside a
  parameterized batch; rows with genuinely incompatible temporal waveform
  shapes must form separate dispatch groups.
- Benchmark modes, presets, flags, and names are documented in
  `benchmark/README.md`. Treat fresh benchmark outputs as evidence only when
  the command, machine metadata, git state, and validation context are recorded.
- Remaining post-P7 cleanup is now model/compiler and evidence oriented:
  flatten the public membrane authoring surface, harden compiler/cache identity
  and output pruning, then use realistic benchmarks to guide current JAX solver
  optimization before introducing a new runtime.

---

# 2. Core Principles

## 2.1 Pre-release Cleanup Policy

Because AxonFleet is not a stable deployed package and currently has one active
user, cleanup should optimize for a clean final architecture. Do not preserve
retrocompatibility, downstream migration paths, compatibility aliases,
deprecated wrappers, old argument names, or transition shims unless explicitly
requested.

Do:

- rename concepts directly;
- rewrite examples and tests;
- delete superseded modules;
- delete obsolete schemas and formats;
- keep one implementation path per concept;
- replace confusing interfaces directly;
- add guardrails before risky cleanup.

Do not accumulate:

- deprecated aliases;
- compatibility wrappers;
- forwarding modules;
- migration shims;
- downstream transition paths;
- `Legacy*` classes;
- duplicate scalar and population APIs;
- duplicate result models;
- obsolete benchmark readers;
- temporary modules that become permanent.

Cleanup is complete only when the replaced path is deleted and the examples,
tests, docs, and guardrails teach only the retained design.

## 2.2 Product Boundary

AxonFleet owns:

- one-dimensional axon models;
- myelinated and unmyelinated axons;
- membrane dynamics;
- single-cable and double-cable formulations;
- intracellular stimulation;
- extracellular potentials already evaluated along axons;
- one-axon and population execution;
- recording and numerical results;
- online and post-hoc scientific analyses;
- threshold, recruitment, and sweep workflows;
- validation, reproducibility, and performance evidence.

AxonFleet must not own:

- detailed nerve geometry;
- fascicle geometry;
- tissue segmentation;
- three-dimensional axon trajectories;
- anatomical coordinate systems;
- electrode CAD;
- finite-element meshing or field solving;
- surgical placement models.

External geometry or field packages should provide numerical extracellular
footprints sampled along each axon. AxonFleet combines those footprints with
temporal stimuli and runs the cable/membrane dynamics.

## 2.3 Intrinsic Versus World Geometry

AxonFleet needs intrinsic one-dimensional position along an axon:

```text
s = 0 ... axon length
```

Intrinsic position is required for discretization, section boundaries, node and
internode selectors, clamp placement, recordings, extracellular footprints, and
event locations.

Layout phase/shift parameters are still intrinsic geometry. `Layout.x_shift`
translates local compartment positions along the one-dimensional axon axis.
`Layout.sequence(..., phase_shift=...)` rotates a repeated motif before it is
cropped to the requested length. `MRG(..., x_shift=...)` is the MRG-specific
public hook for the same node-phase concept: it sets the intrinsic distance
from the axon start to the first node start, and is used when importing NRV
fractional `node_shift` values.

None of these parameters may be used as a synonym for anatomical placement,
electrode offset, or nerve world coordinates.

World position belongs outside the simulation core:

```text
x, y, z, orientation, trajectory, anatomical frame
```

`AxonInstance` does not carry world offsets. Analytical examples that need a
point-source in an external frame must sample the helper into an intrinsic
footprint/drive/stimulation for that axon, then attach the resulting sampled
stimulation to the instance. World geometry must not become a core solver
dependency or required public property.

## 2.4 One Concept, One Public Name

The target repository should read as:

```text
one concept
one public name
one execution path
one result model
```

If two names exist for the same public idea, choose one and delete the other.

---

# 3. Public Object Model

## 3.1 Main Public Concepts

```text
Axon
    descriptive biological, membrane, layout, and cable model

AxonInstance
    one concrete occurrence of an Axon plus local stimulation/state overrides

AxonPopulation
    ordered collection of AxonInstance objects

AxonSimulation
    executable definition for one axon or a population

Stimulus
    temporal waveform

ExtracellularFootprint
    static spatial extracellular transfer profile

ExtracellularDrive
    one footprint plus one stimulus

ExtracellularStimulation
    aggregate of drives

Recording
    public output request

AxonSimulationResult / AxonResultView
    canonical population result and one-row view

Analysis / AnalysisResult / AnalysisReport
    scientific interpretation of numerical outputs

AxonStudy / AxonStudyResult
    future callable simulation families
```

## 3.2 Axon

`Axon` is a reusable scientific description. It owns sections, layout,
membrane models, cable formulation, myelination structure, initial conditions,
temperature, and biophysical parameters.

It does not own simulation duration, backend selection, recording, world
geometry, electrode geometry, compiled arrays, or results.

`Axon`, `Layout`, sections, and membrane models are Python authoring and
inspection descriptions, not the population execution representation. Runtime
preparation lowers them once into typed numerical tables, unique parameter rows,
indices, masks, and contiguous arrays. Population cost should scale with unique
descriptions and numerical array size, not with repeated construction of Python
objects for every axon, amplitude, section occurrence, or compartment.

## 3.3 AxonInstance

`AxonInstance` is one concrete occurrence of an `Axon`. It may contain an id,
label, metadata, parameter overrides, initial-state overrides, and local
intracellular contexts plus an optional typed `ExtracellularStimulation`.

It should not contain trajectory, electrode definitions, field geometry, or
extracellular footprint generation logic.

## 3.4 AxonPopulation

`AxonPopulation` normalizes one or many public axon inputs while preserving
input order. Homogeneous and heterogeneous storage are internal optimization
details; public semantics are the same. Its canonical descriptive view is a
first-occurrence table of immutable axon templates plus one template index per
population row. Per-row `AxonInstance` stimulation and overrides remain
distinct even when rows share one template.

## 3.5 AxonSimulation

`AxonSimulation` is the root executable object and the target public workflow.
It carries axons, duration, time step, recording, solver options, batch options
where still needed, observers, execution policy, and progress settings.

Accepted inputs normalize to a collection of `AxonInstance` rows:

```text
Axon
AxonInstance
Sequence[Axon]
Sequence[AxonInstance]
AxonPopulation
```

One axon and many axons share the lifecycle:

```text
describe
validate
plan
prepare
compile
run
analyze
```

A single axon is the smallest population, not a separate product.

`simulate(...)`, `simulate_pool(...)`, and similar root helpers are not public
workflows. Do not reintroduce them as compatibility aliases.

---

# 4. Typed Public API

Prefer typed values to raw strings whenever the domain is known.

Use:

```text
closed set                  Enum
extensible scientific value  typed descriptor or registered object
structured selection         selector class
user-defined identity        opaque identifier type
display text / metadata      string
serialization boundary       primitive value
```

Do not turn every concept into an enum. Signals, analyses, positions, devices,
and precision policies are structured or extensible concepts.

## 4.1 Closed Domains

Closed domains may use enums:

- myelination;
- cable formulation;
- applicability policy;
- reuse policy;
- retention policy;
- analysis status;
- runtime;
- compartment role.

Raw serialized values such as `"myelinated"` are acceptable at file and
interchange boundaries, but they are not the preferred Python API.

## 4.2 Signals

Signals are extensible descriptors, not a closed enum.

Built-in examples:

```python
axs.signals.MEMBRANE_VOLTAGE
axs.signals.INTRACELLULAR_POTENTIAL
axs.signals.PERIAXONAL_POTENTIAL
axs.signals.IONIC_CURRENT
axs.signals.MEMBRANE_CONDUCTANCE
axs.signals.STATE_VARIABLES
```

Usage:

```python
recording = axs.Recording.center(axs.signals.MEMBRANE_VOLTAGE)
vm = result.signal(axs.signals.MEMBRANE_VOLTAGE)
```

## 4.3 Position Selectors

Do not prefer raw selector strings such as `"all"`, `"distal"`, `"nodes"`, or
`"recorded"`.

Use typed selector objects:

```python
axs.positions.ALL
axs.positions.PROXIMAL
axs.positions.DISTAL
axs.positions.RECORDED
axs.positions.At(2 * axs.mm)
axs.positions.Node(3)
axs.positions.Nodes()
axs.positions.Internodes()
axs.positions.SectionType(...)
axs.positions.Probes(count=16)
```

Selectors resolve against axon structure, layout, and recording metadata.

## 4.4 Opaque Identifiers

Do not use one interchangeable string type for every identity.

Recommended identity types:

```text
AxonId
DriveId
SignalId
CohortId
ModelId
```

This prevents accidental interchange between user-defined ids that serialize to
strings.

## 4.5 Runtime, Device, And Precision

Runtime targets are named public objects under `axs.runtime`. Device and
precision are structured values.

Use:

```python
axs.runtime.auto
axs.runtime.jax

axs.Device.auto()
axs.Device.cpu()
axs.Device.gpu(index=0)

axs.PrecisionPolicy.float32()
axs.PrecisionPolicy.float64()
axs.PrecisionPolicy.mixed(...)
```

Do not make `"gpu"` or `"float32"` the primary public API.

Executable policy:

```python
policy = axs.ExecutionPolicy(
    runtime=axs.runtime.jax,
    device=axs.Device.cpu(),
    precision=axs.PrecisionPolicy.float32(),
)

simulation = axs.AxonSimulation(
    axons=axon,
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
    execution_policy=policy,
)

result = simulation.run()
single = result.single
```

Current behavior:

- `axs.runtime.auto` and `axs.runtime.jax` are valid for execution;
- a future NumPy/SciPy reference runtime gets a public target only after it has
  executable behavior, docs, examples, estimates, inspection, and tests;
- `Device.cpu()` and `Device.gpu(index=...)` resolve through the JAX backend or
  fail clearly;
- uniform `float32` can execute when model dtypes match;
- `float64` requests account for `jax_enable_x64` and require matching model
  dtypes;
- mixed precision is estimate-only until casting/rebuild semantics are
  designed.

Precision must participate in prepared/compiled identity. Device placement
belongs to the backend execution context. Mutable global precision or device
state must not silently change compiled program identity.

---

# 5. Extracellular Model

## 5.1 Footprint/Drive Contract

The preferred extracellular representation is:

```text
ExtracellularFootprint
    static spatial transfer

Stimulus
    temporal waveform

ExtracellularDrive
    one footprint + one stimulus

ExtracellularStimulation
    sum of drives passed to a simulation
```

For one drive:

```text
Vdrive[axon, time, position] =
    footprint[axon, position] * stimulus[time]
```

For many drives:

```text
Vext[a, t, x] =
    sum_d footprint[d, a, x] * stimulus[d, t]
```

The solver or prepared forcing path performs the sum. The full tensor
`Vext[axon, time, position]` must not be materialized by default when the
footprint/drive structure is still available.

The public solver-facing attachment path is exactly
`ExtracellularFootprint` -> `ExtracellularDrive` ->
`ExtracellularStimulation`. `ExtracellularPotential` may stay public only as an
explicit dense imported/inspection/reference object; it is not an attachment
API and must not become the default runtime lowering path.

The old public context/electrode contract is not part of the active API.
`Electrode`, `AnalyticalElectrode`, `ExtracellularContext`,
`AnalyticalExtracellularContext`, `ExtracellularStimulationContext`, and
`NRVExtracellularContext` are historical names only. Analytical point-source
helpers may remain under `axs.analytical`, but they must produce sampled
footprints, drives, or stimulations before solver execution.

## 5.2 ExtracellularFootprint

`ExtracellularFootprint` is static and spatial. It contains no waveform and no
time dimension.

It should preserve:

- axon identifiers or shared flag;
- intrinsic position support;
- units;
- interpolation policy;
- sampling provenance;
- source identifier;
- reference convention;
- optional metadata.

It should not contain electrode CAD, world coordinates, nerve geometry, time
samples, or stimulus amplitude.

## 5.3 ExtracellularDrive

`ExtracellularDrive` groups one footprint, one stimulus, one id/name, and
metadata. The name `drive` is intentionally more general than `electrode`.

## 5.4 ExtracellularStimulation

`ExtracellularStimulation` aggregates drives and validates unique ids,
compatible units, footprint coverage, intrinsic support, stimulus sampling, and
duplicate/conflicting sources.

Immutable edits should return new objects:

```python
updated = extracellular.replace_drive(drive_id, stimulus=new_stimulus)
```

## 5.5 Dense Imported Potential

A dense `ExtracellularPotential` remains useful for non-separable fields,
imported external data, experimental potentials, and reference tests.

It is an explicit imported/inspection/reference object, not the default runtime
lowering representation. Dense internal `Vext` or `Iinj` preparation routes
should disappear once factorized equivalents cover the behavior. The planner or
estimate path should warn when dense input memory is large.

## 5.6 External Geometry Contract

External geometry packages may own nerve geometry, trajectories, electrodes,
conductivity, and FEM or analytical field solving.

Their contract with AxonFleet is numerical:

```text
axon identifiers
intrinsic position support
footprint values
units
provenance
```

AxonFleet may provide lightweight analytical helpers for examples and tests,
but those helpers should produce footprints and remain peripheral to solver
execution.

## 5.7 Study Reuse

The footprint/drive model is designed for amplitude sweeps, thresholds, and
recruitment. When only stimulus samples change, AxonFleet should reuse axon
preparation, footprints, spatial operators, dispatch groups, probe plans, and
compiled executables whenever signatures remain compatible.

Typed waveform sweeps describe a complete waveform factory per sampled value,
not necessarily one scalar multiplier. Proportional monophasic or balanced
waveforms may lower to `scaled_shared_waveform`; independently varying phases,
offsets, timings, or arbitrary samples must preserve their full waveform and
select/rebuild the compatible temporal plan when its shape signature changes.

---

# 6. Recording, Results, And Analyses

## 6.1 Recording

`Recording` is a public output request. It should describe semantic signals,
position selectors, temporal selection, and applicability policy.

Target dependency direction:

```text
Recording
RecordingPlan
axon-structure validation
cable-capability validation
backend lowering
```

`Recording` must not import solver-specific option classes in the final
architecture. Current JAX lowering from `RecordingPlan` to `BatchOptions` lives
under `runtime/jax`.

Unsupported combinations must be explicit. Do not create meaningless arrays or
silently fill unavailable rows with `NaN`.

## 6.2 Myelination And Cable Formulation

Do not conflate biological organization with numerical formulation:

```text
biological organization:
    myelinated
    unmyelinated

numerical formulation:
    single cable
    double cable
```

A myelinated single-cable model can still have nodes, internodes, and
saltatory propagation. A double-cable formulation may additionally expose
intracellular and periaxonal potentials. Semantic signals hide backend
equations from analyses.

## 6.3 Canonical Results

The public result is `AxonSimulationResult` for one axon and populations, with
`AxonResultView` for per-axon access. Internal scalar solver payloads are not
exported and must not appear in examples as a user path. Internal dense storage
blocks are implementation details; examples and public docs should teach
indexing, iteration, `.single`, `.signal(...)`, `recordings`, `recorded_axes`,
`final_states`, diagnostics, observations, and analysis/report helpers.

Use:

```python
simulation = axs.AxonSimulation(
    axons=axon,
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
)
run = simulation.run()
result = run.single

pool_simulation = axs.AxonSimulation(
    axons=pool,
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
)
pool_run = pool_simulation.run()
for row in pool_run:
    row.signal(axs.signals.Vm)
```

Results contain numerical outputs, final state, online observation payloads,
execution metadata, and diagnostics. Per-row data lives on `AxonResultView`;
population-wide aggregation lives on explicit plural properties such as
`recordings`, `recorded_axes`, `final_states`, and `diagnostics`. Post-hoc
analyses return separate objects or reports.

## 6.4 Analyses

Analyses produce events, metrics, statuses, and population summaries.

Examples:

```python
axs.analysis.Activation(...)
axs.analysis.ConductionVelocity(...)
axs.analysis.Latency(...)
axs.analysis.ConductionBlock(...)
axs.analysis.SpikeCount(...)
axs.analysis.PeakVoltage(...)
```

Each analysis declares requirements:

- semantic signals;
- supported myelination classes;
- required compartment roles;
- cable capabilities;
- positions;
- post-hoc and online support;
- algorithm version.

Population aggregates must expose their denominator.

`axs.analysis` is the real public analysis namespace. It must not be a
compatibility alias to `axs.results.analysis`.

## 6.5 VmRaster Observer Path

The active solver-side observer path is intentionally narrow:

```text
input per step: Vm[B, Nx]
state:          packed words[B, R, P, W] uint32
W:              ceil(Nt / 32)
output:         observations["vm_raster"]
```

It thresholds selected membrane-voltage probes at every solver `dt` and packs
bits. Activation, latency, conduction velocity, threshold, and recruitment
summaries are decoded after the solver.

Rules:

- keep public concepts analysis-oriented;
- do not expose backend/detail observer variants as public modes;
- lower fixed probes to static row-aware tables for padded/heterogeneous
  groups;
- keep `VmRasterResult` and CPU unpacking under `results`;
- protocols consume `axonfleet.results` VmRaster output, not solver observer
  internals;
- do not retain full Vm just to recover row-specific targets later;
- keep `PeakVoltage` and richer analyses post-hoc unless benchmark evidence
  justifies a dedicated solver-side implementation.

The old generic solver-side observer fallback should not re-enter active code.

---

# 7. Simulation Lifecycle And Inspection

## 7.1 Lifecycle

Target lifecycle:

```text
validate
plan
prepare
compile
execute
assemble results
analyze
```

Planning determines normalized instances, compatible cohorts, footprint and
drive structures, recording requirements, analysis requirements, expected
shapes, memory estimates, and candidate runtimes.

Planning also owns versioned structural cohort signatures. Compatible numeric
axes and backend shape padding derive those signatures compositionally;
runtime preparation must not re-hash every source row before a cache lookup.
Rebuilding a plan after a relevant structural change is the invalidation
boundary.

Planning must not create device arrays, compile code, import concrete kernels,
or execute solvers.

Preparation may compute flattened cable structures, membrane assignments,
intrinsic positions, cable operators, footprint resampling, compact
intracellular sources, recording indices, observer plans, and cohort metadata.

Compilation lowers prepared structures to a backend executable and dynamic
input schema.

Execution should reuse static structures across compatible stimulus-only
updates.

## 7.2 Inspection

Every major execution stage should be explainable without making diagnostics
part of the hot path.

Target public inspection:

```python
simulation = axs.AxonSimulation(
    axons=pool,
    duration=0.1 * axs.ms,
    dt=0.05 * axs.ms,
    execution_policy=policy,
)

report = simulation.inspect(print_summary=True)
report.print()
report.plot()
report.plot_details()
```

Root inspection helpers should not be public paths. Keep inspection on
`AxonSimulation.inspect()` so planning, estimate, lowering, and report objects
share the same simulation definition.

Current coverage:

- planning;
- dispatch/batch grouping;
- prepared cohort shapes;
- input lowering: compact/factorized/zero intracellular input and
  footprint/drive/factorized/explicit-dense extracellular input;
- observer/recording lowering: retained Vm width and VmRaster/post-hoc route;
- kernel routing: scalar or batch, cable formulation, block solver, chunks;
- result assembly: compact cohort records, row records, observations, retained
  recordings;
- detailed plotting for padding, estimated memory, VmRaster probes, and result
  assembly.
- progress reporting: structured dispatch/backend events for route choice,
  preparation, input lowering, kernel chunks, and result assembly.

Target coverage:

- planning: normalized axons, signatures, candidate runtimes, reuse keys;
- dispatch/batch: scalar/batch route, groups, padding, fallback reasons;
- preparation: flattened layouts, membrane/cable shapes, cache hits/misses,
  probe tables, prepared footprints;
- input lowering: waveform and footprint reuse/cache hits;
- observer/recording lowering: VmRaster probes, thresholds, masks, packed word
  shapes;
- kernel routing: device and dtype details from execution-time capture;
- result assembly: slicing, padded-row removal, and post-processing
  requirements.

Inspection must not allocate device arrays during planning, force device to
host transfers by default, expose private backend objects, change numerical
results, or replace benchmark spans for timing claims.

---

# 8. Solver And Runtime Boundary

## 8.1 Ownership Direction

Required dependency direction:

```text
core / units / identifiers / timebase
domain descriptions
planning
preparation
backend lowering
execution
results
analysis / visualization
studies
```

Forbidden dependencies:

- `axons/` imports JAX;
- `membranes/` imports JAX implementations;
- `extracellular/` imports geometry packages;
- analytical helpers enter solver execution directly;
- `recording/` imports solver options in the final architecture;
- `planning/` creates device arrays or calls backend lowering;
- `results/` depends on JAX array types;
- analyses encode backend equations;
- domain objects eagerly import visualization;
- internal modules import top-level `axonfleet`;
- mutable global precision silently changes compile identity.

## 8.2 Runtime Boundary

Public/descriptive layers should pass typed runtime requests and public
simulation definitions downward. Backend-specific modules own JAX imports,
device resolution, array placement, and kernel lowering.

The canonical materialization pipeline is:

```text
immutable Python descriptions
        -> backend-neutral NumPy lowering
        -> backend array placement/lowering
        -> execution
```

The NumPy stage is host-side materialization, not a future NumPy/SciPy
solver. It should produce compact structure-of-arrays representations, unique
parameter tables plus row/compartment indices, masks, and stable signatures.
The JAX stage consumes those numerical contracts and owns `jax.Array` creation,
device transfer, compilation, and kernels.

Pure intrinsic translations created through `Layout.with_x_shift()` share one
solver geometry/cable template plus a row-specific x shift. Motif phase
changes, topology changes, section-parameter changes, and simulation-level
cable overrides are distinct templates; never infer translation sharing from
approximately equal flattened arrays.

Lowering and runtime code must dispatch by typed capabilities, structure, shape,
and parameter signatures. It must not select optimized execution paths from
concrete authoring class names such as `MRG`, `RattayAberham`, or a particular
membrane family. A concrete model may be a benchmark and validation case, but
not a second production path.

Current boundary:

- public simulation entry points call `axonfleet.runtime.execution`, which
  resolves the currently supported concrete backend without importing JAX
  adapters from `simulation.py`;
- batch execution enters concrete JAX code only through backend execution
  facades, including one-row `B=1` simulations;
- dispatcher modules own planning, grouping, progress, and dispatch records,
  but must not import `axonfleet.runtime.jax` or call the JAX group runner
  directly;
- fixed-step time-grid validation and solver time-argument normalization live
  in `axonfleet.runtime.timebase`;
- `axonfleet.solvers` exports only the active batch execution contracts:
  `BatchOptions` and `BatchRecording`;
- JAX runtime preparation, stimulation compilation, batch kernels, low-level
  numerical helper kernels, observer packing, and backend input containers live
  under `axonfleet.runtime.jax`;
- backend-owned compiled-kernel caches live under
  `.axonfleet_cache/runtime/<backend>/` and remain internal runtime machinery;
- `ExecutionPolicy` resolves JAX device/runtime in the backend layer;
- `solvers/` retains only solver-facing contracts; executable solver routes
  live under concrete runtimes.

Persistent compiled-kernel artifacts must be content-addressed by every input
that can change generated device code or its launch contract. This includes the
kernel source, shapes and dtypes, compile metaparameters, precision, target
platform and compute capability, and compiler/runtime package versions.
Artifacts require a versioned manifest and checksum; unreadable, mismatched, or
unsupported artifacts are cache misses, never compatibility fallbacks. A cache
adapter that mirrors private compiler APIs must enable only explicitly reviewed
package versions and otherwise delegate to the package's normal lowering path.

## 8.3 Public Solver Surface

High-level execution policy is typed. Public code should choose runtime,
device, precision, and per-cable solver policy with:

```text
ExecutionPolicy
Device
PrecisionPolicy
SolverPolicy
runtime.jax.SingleCableSolver
runtime.jax.DoubleCableSolver
```

Policy:

- CPU double-cable is Thomas-only: `auto` resolves to `thomas`, and the only
  explicit CPU double-cable route is
  `axs.runtime.jax.cpu.DoubleCableSolver.thomas()`;
- CPU single-cable lowers to the portable JAX tridiagonal solve;
- CUDA single-cable lowers to one guarded internal exact scalar tiled-Thomas
  Triton route; it is not a separate public solver choice;
- GPU double-cable lowers to the retained looped jax-triton tiled-Thomas route;
- solver-specific options must live under typed solver policy values, not in
  `BatchOptions`;
- benchmark CLIs may keep string flags, but active benchmark workloads must
  translate them to typed policies at the benchmark boundary.

Do not expose approximate double-cable surrogate, split iterative, associative-transfer, Pallas,
static Triton, CUDA FFI, or other custom-kernel candidates as public solver
choices while they remain archived or standby evidence. Retained custom
kernels stay backend-internal unless a genuine user-facing policy distinction
is supported by fresh benchmark and validation evidence.

`BatchOptions` and `BatchRecording` are currently public advanced execution
knobs for batch-kernel retained Vm policy and time chunking. They are not the
solver-selection surface. Public examples should import them from the root
facade (`axs.BatchOptions`) rather than descending into `axs.solvers`. Longer
term, replace public tuning exposure with a clearer output/chunking surface and
make `BatchRecording` internal once `Recording` covers the needed cases.

`BatchOptions.none()` is the observer-only compact-output policy and defaults
to `axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS` to reduce cold JAX recompilation
across duration sweeps. Chunked observer kernels must assemble local VmRaster
chunk states back into the same public full-duration `VmRasterResult`; do not
introduce a second public observer result shape. Explicit
`time_chunk_steps=None` means unchunked.

## 8.4 Factorized Forcing

Factorized `Vext`/`Iinj` is an internal optimization direction, not a user
mode. The target extracellular lowering contract is shared by single-cable and
double-cable runtimes: prepared rows lower to static spatial footprints plus an
explicit temporal payload mode, then each cable solver consumes the modes it
declares as supported. The executable semantic contract lives in
`axonfleet.runtime.inputs.contracts`; this section owns its architecture-level
constraints.

The active compact extracellular path is deliberately narrow:

```text
current_mid_A[S, Nt] or base_current_mid_A[S, Nt] + row_scales[B, S]
footprint_mV_per_A[B, S, Nx]
Vstim[B, Nt, Nx] = sum_S current_mid_A * footprint_mV_per_A
```

Squeezed rank-1 forms are allowed internally for one-drive/shared-current
batches. The active compact path is used for static-footprint single-cable
observer-only and recorded-Vm batches to avoid dense `Vstim[B, Nt, Nx]`.
The next cleanup target is to separate temporal waveform shape from amplitude
scale, so threshold-style sweeps can stay in one `Nstim`-aware group with row
numeric scales instead of rebuilding row-local public stimuli. Multi-drive
stimulation should be represented as the same shared/scaled modes with
`Nstim > 1`; arbitrary temporal waveforms fall back to a current table or dense
route instead of a separate rank-K public concept.

Do not expose dense/factorized as public modes. Do not keep dense internal
preparation paths once the factorized route covers the same behavior. Do not
silently fall back to dense `Vstim[B, Nt, Nx]` inside observer-only compact
paths; reject unsupported stimulation shapes until a measured compact lowering
exists. Do not broaden rank-K factorized extracellular forcing to double-cable
without solver-equivalence tests and benchmark evidence.

## 8.5 Precision And Cache Identity

Prepared identity may include semantic identity, discretization, cohort shape,
resampled footprints, and transformed footprints.

Compiled identity may include prepared identity, backend, backend version,
device class, precision, static shapes, drive count, solver algorithm, and
optimization flags.

Stimulus samples should remain dynamic when shape-compatible.

---

# 9. Studies And Reuse

Studies are future public orchestration for related variants of a base
`AxonSimulation`.

Canonical update mechanism:

```python
def update(base_simulation, condition) -> axs.AxonSimulation:
    ...
```

The callable should avoid mutating the base, return a new simulation, avoid
hidden side effects, and make the condition explicit.

Target workflows:

- sweeps;
- threshold search;
- recruitment curves;
- retention policies;
- reuse policies.

Reuse policies:

```text
AUTO      reuse compatible plans, preparation, and compilation
REQUIRE   fail if a condition violates the reuse boundary
NONE      treat every condition independently
```

Lambdas are allowed, but AxonFleet should not claim every lambda is
serializable. Recommend named functions or frozen callable dataclasses for
strong reproducibility.

---

# 10. Serialization And Reproducibility

Typed Python values serialize to stable primitive values:

```json
{
  "myelination": "myelinated",
  "formulation": "double_cable",
  "signal": "membrane_voltage",
  "drive_id": "cathode"
}
```

Deserialization reconstructs typed values. Serialized representation must not
dictate an untyped in-memory API.

Potential final APIs:

```python
simulation.save("simulation.axs.json")
result.save("result.axs")
study_result.save("study.axs")
```

Only final schemas should receive readers and writers. Do not maintain readers
for prototype formats.

---

# 11. Source Organization Direction

Do not create empty packages without moving real responsibilities. Do not keep
forwarding modules for obsolete import paths.

Target ownership map:

```text
core/              units, errors, identifiers, enums, serialization helpers
axons/             descriptive axons, sections, layouts, templates
membranes/         runtime-independent membrane descriptions
model_ir/          internal runtime-independent membrane semantics, validation, hashes, fusion contracts
stimulation/       stimuli, intracellular clamps, and sampled stimulation objects
extracellular/     footprints, drives, dense potentials, validation
recording/         public recording specs and selectors
signals/           typed signal descriptors and registry
simulation/        AxonInstance, AxonPopulation, AxonSimulation
planning/          plans, cohorts, compatibility, signatures, inspection
preparation/       host-side prepared geometry/membrane/input structures
runtime/           public runtime policy, runtime registry, and concrete JAX/NumPy lowering
execution/         engine, progress, group runner, result assembly
results/           result containers, views, manifests, serialization
analysis/          scientific definitions and post-hoc algorithms
studies/           sweeps, thresholds, recruitment
observability/     benchmark spans, trace reports, profiler metadata
visualization/     plotting only
```

Current code may still differ. Use this map to guide moves, not to justify
empty directories.

Cleanup reminders:

- move `dispatcher/plan.py` responsibilities toward `planning/`;
- keep host-side axon, membrane, stimulation, and cohort row materialization in
  `preparation/` and backend array lowering under `runtime/jax`;
- keep fixed-step timebase rules out of JAX-heavy solver helper modules;
- keep runtime dataclasses, preparation helpers, numerical helper kernels, and
  batch kernels out of the `axonfleet.solvers` package facade;
- keep public `recording.py` independent from solver options;
- keep JAX membrane/solver implementation under `runtime/jax`;
- keep `solvers/` as a public facade for stable solver option contracts during
  cleanup, not as a permanent catch-all for backend internals;
- keep result-side VmRaster containers and CPU decoders out of solver modules.

---

# 12. Examples And Documentation

Examples must:

- use the public API directly;
- avoid importing solver/backend internals in public tutorials;
- document every public option, possibility, feature, runtime mode, inspection
  view, analysis workflow, solver-facing user concept, and advanced knob. If a
  feature is not worth describing in an executable example, remove or archive
  it rather than leaving it as hidden public surface;
- teach one concept per demo when possible;
- write examples as executable teaching material, not as terse smoke snippets;
- prefer a readable line-by-line flow with comments next to the relevant code;
- keep `examples/basic/` especially flat: avoid helper functions unless they
  make the script easier to read than the explicit sequence;
- include plots when they clarify Vm traces, activation, recruitment,
  velocity, recording retention, dispatch layouts, or observer checks;
- construct footprints separately from stimuli;
- keep world/anatomical placement out of `AxonInstance`; when a didactic
  point-source example needs external offsets, use `axs.analytical` helpers to
  build sampled footprints/drives before attaching stimulation;
- when an NRV example needs reusable handoff logic, use the two-bridge contract
  in `axonfleet.integrations.nrv`: `population_from_nrv(...)` creates the
  AxonFleet population from NRV fibers, then `footprints_from_nrv(...)` samples
  every NRV electrode footprint on that population. Do not redefine fiber-row
  extraction, LIFE/FEM footprint sampling, or NRV recruitment decoding inside
  examples, and do not add NRV geometry/population/electrode builders to the
  integration module;
- group one footprint and one stimulus into an `ExtracellularDrive`;
- pass `ExtracellularStimulation` to simulations when using the factorized
  extracellular model;
- use callable study updates once studies exist;
- avoid obsolete simulation entry points;
- keep benchmark/profiling and CPU/GPU measurement workflows under
  `benchmark/`, unless the example explicitly teaches a public benchmarking or
  inspection API;
- show runtime/device/precision only through `axs.runtime`, `axs.Device`,
  `axs.PrecisionPolicy`, and `axs.ExecutionPolicy`;
- show only retained public solver options;
- remain syntax-checked or import-checked in CI.

Current didactic example organization:

```text
examples/basic/
    compact first-pass scripts that show core AxonFleet capabilities
examples/advanced/simulation_workflow/
    AxonSimulation, AxonPopulation, one/many execution, estimate/inspect/run
examples/advanced/axon_models/
    layouts, custom axons, cable formulation
examples/advanced/stimulation/
    intracellular clamps, extracellular footprints, drives, stimulations
examples/advanced/recording_analysis/
    recording policies, typed signals/positions, analysis, VmRaster
examples/advanced/protocols/
    threshold and recruitment workflows
examples/advanced/runtime/
    ExecutionPolicy, preparation signatures, pipeline inspection
examples/with_nrv/
    optional NRV geometry/fiber-placement integration
examples/tutorials/
    notebook mini-courses, indexed like a teaching sequence
benchmark/
    profiling, CPU/GPU measurement, benchmark notebooks
```

The old `examples/advanced/object_model/` name was too abstract. The retained
folder is `examples/advanced/simulation_workflow/`; it teaches the one-path
simulation model rather than a historical object taxonomy.

When examples change, update together:

- `examples/README.md`;
- README commands;
- smoke/import tests;
- relevant docs pages;
- benchmark references if an example moved under `benchmark/`.

---

# 13. Benchmark Surface

Benchmark code is allowed to be richer and messier than tutorials while a
performance question is being explored, but the supported benchmark surface must
not be ambiguous.

Before treating benchmark results as product evidence, audit:

- CLI modes, presets, flags, and naming;
- which scripts are active, validation-only, experimental, archived, or
  generated output;
- whether each active mode uses the retained public simulation, estimate,
  inspection, recording, observer, and analysis APIs;
- whether benchmark-only imports or private helpers are clearly contained under
  `benchmark/`;
- whether generated outputs live outside architecture/docs decisions.

For every retained benchmark mode, document:

- purpose and workload;
- command line;
- expected machine metadata: CPU, GPU, RAM, VRAM, driver/CUDA, JAX backend and
  device, OS, Python, package versions, and important environment variables;
- memory measurement strategy, including host profiler data where possible,
  JAX/JAX-profiler data where relevant, and GPU VRAM reporting through
  `jax-smi` or an equivalent tool when available;
- JAX GPU memory behavior, including that JAX preallocates 75% of GPU VRAM by
  default unless configured otherwise;
- cold versus warm timing policy, cache hits/misses, `runtime.prepare`,
  `kernel.dispatch_jax`, enqueue time, result assembly, and retained output
  size when those claims matter.

Remove or archive unclear benchmark modes instead of documenting around them.
Do not make speed or memory claims from stale benchmark outputs.

---

# 14. Testing And Acceptance Criteria

## 14.1 Critical Test Classes

Guardrails should cover:

- no JAX imports in public/descriptive layers;
- no direct `dispatcher` imports from `axonfleet.runtime.jax`;
- no geometry package dependency in core AxonFleet;
- no raw strings as preferred public API for closed/structured domains;
- no legacy compatibility aliases, migration shims, deprecated wrappers, or old
  argument-name aliases;
- public exports stay intentional;
- standby solver candidates do not re-enter public options or active runtime
  helpers;
- examples import or execute through public APIs;
- every public option/feature/workflow has an executable example or is removed
  or archived;
- factorized paths avoid dense `Vext`/`Iinj` when promised;
- observer-only paths do not retain full Vm unless requested;
- online and post-hoc analyses are cross-validated;
- stimulus-only updates reuse compatible static structures where implemented.

## 14.2 Architecture Acceptance

Typed API:

- closed domains use enums;
- extensible signals use typed descriptors;
- selectors use dedicated classes;
- identities use opaque identifier types;
- devices and precision use structured objects;
- raw strings remain limited to labels, metadata, and serialization boundaries.

Product boundary:

- AxonFleet owns intrinsic axon geometry only;
- `AxonInstance` has no required world position;
- external geometry packages provide numerical footprints;
- core AxonFleet does not depend on nerve geometry packages.

Extracellular model:

- footprints contain spatial transfer only;
- stimuli contain temporal waveform only;
- drives pair one footprint with one stimulus;
- stimulation aggregates drives;
- solver/prepared execution sums drive contributions;
- footprint/drive paths do not materialize full `Vext` by default;
- stimulus-only updates can reuse prepared footprints.

Simulation model:

- one and many axons use the same lifecycle;
- public result semantics are coherent across cardinalities;
- myelination and cable formulation remain separate metadata;
- shared and per-axon footprints are supported.

Results and analyses:

- unsupported signals are explicit;
- analyses remain separate from numerical outputs;
- population summaries expose denominators;
- VmRaster remains a strict packed runtime primitive surfaced through
  `VmRasterResult` and decoded into public analysis semantics.

Cleanup:

- obsolete modules are deleted;
- forwarding compatibility aliases and transition shims are removed;
- examples use public APIs directly;
- scientific reference tests pass.

---

# 15. Non-goals

Do not:

- make AxonFleet own nerve geometry;
- make world position mandatory on `AxonInstance`;
- keep electrode geometry in the solver core;
- conflate footprint and stimulus;
- pre-sum all drives before execution;
- materialize `Vext[axon, time, position]` by default when footprint/drive
  structure is still available;
- keep dense `Vext`/`Iinj` internal routes once factorized routes cover the
  behavior;
- create separate one-axon and population products;
- conflate myelination and cable formulation;
- merge analyses into raw numerical results;
- expose backend arrays as public contracts;
- preserve prototype compatibility, aliases, deprecated wrappers, or downstream
  migration paths;
- keep forwarding modules;
- introduce a generic kernel IR before a real second backend;
- create empty packages without responsibilities;
- expose archived solver experiments as public choices.

---

# 16. Final Target API Sketch

Build footprints outside AxonFleet or with lightweight helpers:

```python
cathode_footprint = external_package.compute_footprint(
    source="cathode",
    axons=population.ids,
    intrinsic_positions=population.compartment_positions,
)
```

Build drives:

```python
cathode = axs.ExtracellularDrive(
    id=axs.DriveId("cathode"),
    footprint=cathode_footprint,
    stimulus=cathode_stimulus,
)
```

Aggregate drives:

```python
extracellular = axs.ExtracellularStimulation([cathode])
```

Run:

```python
simulation = axs.AxonSimulation(
    axons=population,
    extracellular=extracellular,
    duration=20 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.MEMBRANE_VOLTAGE),
    execution_policy=axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.float32(),
    ),
)

simulation.inspect(print_summary=True)
result = simulation.run()
```

Analyze:

```python
activation = result.analyze(
    axs.analysis.Activation(...)
)
```

Study:

```python
study = simulation.sweep(
    values=amplitudes,
    update=update_amplitude,
    reuse=axs.ReusePolicy.AUTO,
)
```

Defining principles:

```text
AxonFleet knows axons in intrinsic one-dimensional space, not nerve world space.
External geometry packages provide spatial extracellular footprints.
One ExtracellularDrive combines one footprint with one stimulus.
ExtracellularStimulation aggregates all drives.
The solver/prepared backend performs forcing without dense `Vext`/`Iinj` by
default whenever the factorized representation covers the behavior.
Prototype APIs should be deleted once replaced.
```
