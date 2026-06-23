# AxonScope Architecture Guidelines

Snapshot: 2026-06-23.

This is the consolidated architecture reference for AxonScope. It is normative
for project direction, refactors, public API shape, examples, and cleanup
decisions. The source tree, tests, examples, and benchmark reports remain the
implementation source of truth for current behavior.

Use this document to answer three questions:

1. What product is AxonScope?
2. Which concepts are public and stable?
3. Where must solver/runtime/backend code live?

Keep this file concise. Detailed experiment notes belong in `benchmark/`,
`docs/`, or `ideas/`; active work belongs in `todo.md`; operational agent notes
belong in `agent.md`.

---

# 1. Current Status

AxonScope is still pre-release. Prefer a clean final design over compatibility
with prototype APIs. Delete superseded paths rather than preserving aliases.

Current focus:

- keep only retained solver routes in active runtime code;
- preserve the public/runtime/backend boundary;
- flatten public examples against the public API;
- make runtime/device/precision policy executable;
- make planning, batch/dispatch, preparation, lowering, execution, and result
  assembly inspectable;
- keep benchmark/profiling experiments out of public tutorials.

## 1.1 Phase Snapshot

| Phase | Status | Current surface |
| --- | --- | --- |
| 0 - Guardrails and baselines | Done | Architecture tests, public API cleanup checks, import-boundary checks, scientific baselines. |
| 1 - Object model | Done | `AxonInstance`, root `AxonSimulation`, `AxonPopulation`, descriptive `Axon`. |
| 2 - Typed and extracellular contracts | Done | Typed signals, position selectors, cable formulation, identifiers, footprint/drive/stimulation objects. |
| 2.5 - Hotpath evidence | Done | Opt-in benchmark spans, hotpath workload catalog, Colab GPU workflow. |
| 3 - Planning and preparation | Done for current JAX path | Preparation signatures, prepared cohorts, footprint-oriented preparation. |
| 4 - JAX isolation | Done for current boundary | Scalar and batch execution enter through `axonscope.backends.jax`; low-level kernels still need later ownership cleanup. |
| 5 - Canonical simulation results | Done | `AxonSimulationResult`, `AxonResultView`, `RecordingManifest`, `RecordedSignal`, internal dense storage blocks. |
| 6 - Analyses | Done for current public layer | Real `axs.analysis`, definitions, requirements, statuses, reports, post-hoc helpers. |
| 7 - Performance evidence | Done for current evidence layer | Estimates, hotpath metadata, memory pressure reporting, footprint reuse evidence. |
| 7.5 - Generic solver-side observers | Superseded | Broad observer path removed from active direction; `PeakVoltage` remains post-hoc. |
| 7.6.1-7.6.2 - Hotpath/memory cleanup | Done for evidence layer | Sparse/zero inputs, compact observer outputs, runtime caches, chunking, profiler traces. |
| 7.6.3 - Exact double-cable GPU solver | Closed | Retained choices: `auto`, `thomas`, `pcr`, `pcr_soa`, `pcr_adaptive`. |
| 7.6.4 - Pseudo-double validation | Standby | Harness exists under `benchmark/pseudo_double/`; not a public solver replacement. |
| 7.6.5 - Execution envelope and Vext | In progress | Reduce prepare/dispatch/probe-plan rebuilds, dense/factorized Vext materialization, enqueue/result overhead. |
| 7.6.6 - GPU dispatch scheduling | Planned | Memory-aware bucketing/coalescing before optional async scheduling. |
| 7.6.7 - VmRaster redesign | In progress | One strict threshold raster primitive, packed in solver, decoded post-hoc. |
| 7.7 - Solver surface stabilization | In progress | Archive standby candidates, align `solvers/` with backend boundary, keep factorized Vext internal. |
| 7.8 - Runtime policy and inspection | In progress | `ExecutionPolicy` controls JAX device/runtime; `inspect_simulation()` prints planning, dispatch, prepare, lowering, kernel, and result assembly. |
| 8 - Studies | Not started | Callable studies, reuse policies, retention policies, study results. |
| 9 - Serialization and reference backend | Not started | Final schemas, NumPy reference backend, cross-backend validation. |

## 1.2 Active Gaps

- Public execution returns one result model: `simulate(...)`, `simulate_pool(...)`,
  and `AxonSimulation.run()` return `AxonSimulationResult`. One-axon access is
  through `.single` or `[0]`.
- `Recording.to_plan()` now produces a backend-neutral `RecordingPlan`; the JAX
  backend lowers that plan to batch-kernel options. The remaining work is to
  broaden validation and move more result/observer boundaries out of solver
  modules.
- VmRaster output remains exposed as `observations["vm_raster"]`, with
  `VmRasterResult` and CPU unpacking under `results`. Solver/backend code owns
  only plan lowering and packed bit updates.
- Fixed-step time-grid validation lives in backend-neutral `axonscope.timebase`.
  Public planning, estimation, and inspection code must not import JAX-heavy
  solver numerical helpers for this contract.
- `ExecutionPolicy` now resolves JAX device requests and validates uniform
  precision. Precision participates in membrane/runtime cache identity; runtime
  execution still rejects implicit casting instead of rebuilding models.
- `inspect_simulation(...)` covers host-side planning, dispatch/batch,
  preparation, input/observer/recording lowering, kernel routing, result
  assembly, and detailed plots for padding, memory, probes, and assembly.
- Factorized Vext is active for compatible static-footprint rows and remains an
  internal lowering choice. Keep it behind dense-equivalence tests and
  benchmark evidence.

---

# 2. Core Principles

## 2.1 Pre-release Cleanup Policy

Because AxonScope is not a stable deployed package, migrations should optimize
for a clean final architecture.

Do:

- rename concepts directly;
- rewrite examples and tests;
- delete superseded modules;
- delete obsolete schemas and formats;
- keep one implementation path per concept;
- add guardrails before risky cleanup.

Do not accumulate:

- deprecated aliases;
- compatibility wrappers;
- forwarding modules;
- `Legacy*` classes;
- duplicate scalar and population APIs;
- duplicate result models;
- obsolete benchmark readers;
- temporary modules that become permanent.

A migration is complete only when the replaced path is deleted.

## 2.2 Product Boundary

AxonScope owns:

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

AxonScope must not own:

- detailed nerve geometry;
- fascicle geometry;
- tissue segmentation;
- three-dimensional axon trajectories;
- anatomical coordinate systems;
- electrode CAD;
- finite-element meshing or field solving;
- surgical placement models.

External geometry or field packages should provide numerical extracellular
footprints sampled along each axon. AxonScope combines those footprints with
temporal stimuli and runs the cable/membrane dynamics.

## 2.3 Intrinsic Versus World Geometry

AxonScope needs intrinsic one-dimensional position along an axon:

```text
s = 0 ... axon length
```

Intrinsic position is required for discretization, section boundaries, node and
internode selectors, clamp placement, recordings, extracellular footprints, and
event locations.

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

## 3.3 AxonInstance

`AxonInstance` is one concrete occurrence of an `Axon`. It may contain an id,
label, metadata, parameter overrides, initial-state overrides, and local
intracellular or extracellular contexts.

It should not contain trajectory, electrode definitions, field geometry, or
extracellular footprint generation logic.

## 3.4 AxonPopulation

`AxonPopulation` normalizes one or many public axon inputs while preserving
input order. Homogeneous and heterogeneous storage are internal optimization
details; public semantics are the same.

## 3.5 AxonSimulation

`AxonSimulation` is the root executable object. It carries axons, duration,
time step, recording, solver options, batch options, observers, execution
policy, and progress settings.

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

Runtime is a closed enum. Device and precision are structured values.

Use:

```python
axs.Runtime.AUTO
axs.Runtime.JAX
axs.Runtime.NUMPY

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
    runtime=axs.Runtime.JAX,
    device=axs.Device.cpu(),
    precision=axs.PrecisionPolicy.float32(),
)

result = axs.simulate(
    simulation,
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
    execution_policy=policy,
)
single = result.single
```

Current behavior:

- `Runtime.AUTO` and `Runtime.JAX` are valid for execution;
- `Runtime.NUMPY` is reserved for a future reference backend;
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

## 5.5 Dense Fallback

A dense `ExtracellularPotential` remains useful for non-separable fields,
imported external data, experimental potentials, and reference tests.

It is a fallback, not the default performance representation. The planner or
estimate path should warn when dense input memory is large.

## 5.6 External Geometry Contract

External geometry packages may own nerve geometry, trajectories, electrodes,
conductivity, and FEM or analytical field solving.

Their contract with AxonScope is numerical:

```text
axon identifiers
intrinsic position support
footprint values
units
provenance
```

AxonScope may provide lightweight analytical helpers for examples and tests,
but those helpers should produce footprints and remain peripheral to solver
execution.

## 5.7 Study Reuse

The footprint/drive model is designed for amplitude sweeps, thresholds, and
recruitment. When only stimulus samples change, AxonScope should reuse axon
preparation, footprints, spatial operators, dispatch groups, probe plans, and
compiled executables whenever signatures remain compatible.

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
under `backends/jax`.

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
run = axs.simulate(sim, duration=5 * axs.ms, dt=0.01 * axs.ms)
result = run.single

pool_run = axs.simulate_pool(pool, duration=5 * axs.ms, dt=0.01 * axs.ms)
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
- protocols consume `axonscope.results` VmRaster output, not solver observer
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

Current public inspection:

```python
report = axs.inspect_simulation(
    pool,
    duration=0.1 * axs.ms,
    dt=0.05 * axs.ms,
    execution_policy=policy,
)

report.print()
report.plot()
report.plot_details()

simulation.inspect(print_summary=True)
```

Current coverage:

- planning;
- dispatch/batch grouping;
- prepared cohort shapes;
- input lowering: dense/sparse/zero intracellular input, dense/factorized/zero
  extracellular input;
- observer/recording lowering: retained Vm width and VmRaster/post-hoc route;
- kernel routing: scalar or batch, cable formulation, block solver, chunks;
- result assembly: compact cohort records, row records, observations, retained
  recordings;
- detailed plotting for padding, estimated memory, VmRaster probes, and result
  assembly.

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

# 8. Solver And Backend Boundary

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
- internal modules import top-level `axonscope`;
- mutable global precision silently changes compile identity.

## 8.2 Runtime Boundary

Public/descriptive layers should pass typed runtime requests and public
simulation definitions downward. Backend-specific modules own JAX imports,
device resolution, array placement, and kernel lowering.

Current boundary:

- public simulation entry points call `axonscope.backends.execution`, which
  resolves the currently supported concrete backend without importing JAX
  adapters from `simulation.py`;
- scalar and batch execution enter through `axonscope.backends.jax`;
- fixed-step time-grid validation and solver time-argument normalization live
  in `axonscope.timebase`;
- `axonscope.solvers` exports only the stable solver facade: solver base,
  `CrankNicholson`, solver options, batch options, and block-solver resolution;
- JAX runtime preparation, stimulation compilation, scalar kernels, batch
  kernels, observer packing, and backend input containers live under
  `axonscope.backends.jax`;
- `ExecutionPolicy` resolves JAX device/runtime in the backend layer;
- `solvers/` retains only solver-facing contracts and the public solver class.

## 8.3 Public Solver Surface

Retained exact double-cable block-solver choices:

```text
auto
thomas
pcr
pcr_soa
pcr_adaptive
```

Policy:

- `auto` resolves from the effective execution device;
- CPU/default backends resolve to `thomas`;
- GPU-like backends resolve to `pcr_adaptive`;
- `pcr_adaptive` uses `pcr_soa` through `B <= 4096` and `pcr` above that;
- forced choices are diagnostic unless benchmark evidence updates defaults.

Do not expose pseudo-double, split iterative, associative-transfer, Pallas,
Triton, JAX-Triton, CUDA FFI, or other custom-kernel candidates as public
solver choices while they remain archived or standby evidence.

`BatchOptions` and `BatchRecording` are currently public execution knobs for
batch-kernel recording/chunking and retained solver selection. Public examples
should import them from the root facade (`axs.BatchOptions`) rather than
descending into `axs.solvers`.

## 8.4 Factorized Vext

Factorized Vext is an internal optimization. The active path is deliberately
narrow:

```text
current_mid_A[Nt]
footprint_mV_per_A[B, Nx]
```

It is used for static-footprint single-cable VmRaster observer-only batches to
avoid dense `Vstim[B, Nt, Nx]`.

Do not expose factorized Vext as a public mode. Do not broaden it to
double-cable without solver-equivalence tests and benchmark evidence.

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

Lambdas are allowed, but AxonScope should not claim every lambda is
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
stimulation/       stimuli and intracellular contexts
extracellular/     footprints, drives, dense potentials, validation
recording/         public recording specs and selectors
signals/           typed signal descriptors and registry
simulation/        AxonInstance, AxonPopulation, AxonSimulation
planning/          plans, cohorts, compatibility, signatures, inspection
preparation/       host-side prepared geometry/membrane/input structures
backends/          backend registry and concrete JAX/NumPy lowering
execution/         engine, progress, group runner, result assembly
results/           result containers, views, manifests, serialization
analysis/          scientific definitions and post-hoc algorithms
studies/           sweeps, thresholds, recruitment
observability/     benchmark spans, trace reports, profiler metadata
visualization/     plotting only
```

Current code may still differ. Use this map to guide moves, not to justify
empty directories.

Migration reminders:

- move `dispatcher/plan.py` responsibilities toward `planning/`;
- keep host-side runtime-batch row helpers in `preparation/runtime_batches.py`
  and backend array lowering under `backends/jax`;
- keep fixed-step timebase rules out of JAX-heavy solver helper modules;
- keep runtime dataclasses, preparation helpers, scalar kernels, and batch
  kernels out of the `axonscope.solvers` package facade;
- keep public `recording.py` independent from solver options;
- keep JAX membrane/solver implementation under `backends/jax`;
- keep `solvers/` as a public facade for stable solver classes/options during
  migration, not as a permanent catch-all for backend internals;
- keep result-side VmRaster containers and CPU decoders out of solver modules.

---

# 12. Examples And Documentation

Examples must:

- use the public API directly;
- avoid importing solver/backend internals in public tutorials;
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
- group one footprint and one stimulus into an `ExtracellularDrive`;
- pass `ExtracellularStimulation` to simulations when using the factorized
  extracellular model;
- use callable study updates once studies exist;
- avoid obsolete simulation entry points;
- keep benchmark/profiling and CPU/GPU measurement workflows under
  `benchmark/`, unless the example explicitly teaches a public benchmarking or
  inspection API;
- show runtime/device/precision only through `axs.Runtime`, `axs.Device`,
  `axs.PrecisionPolicy`, and `axs.ExecutionPolicy`;
- show only retained public solver options;
- remain syntax-checked or import-checked in CI.

Current didactic example organization:

```text
examples/basic/
    compact first-pass scripts that show core AxonScope capabilities
examples/advanced/object_model/
    AxonSimulation, AxonPopulation, canonical pool results
examples/advanced/axon_models/
    layouts, custom axons, cable formulation
examples/advanced/stimulation/
    stimulation contexts, extracellular footprints, drives
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

When examples change, update together:

- `examples/README.md`;
- README commands;
- smoke/import tests;
- relevant docs pages;
- benchmark references if an example moved under `benchmark/`.

---

# 13. Testing And Acceptance Criteria

## 13.1 Critical Test Classes

Guardrails should cover:

- no JAX imports in public/descriptive layers;
- no geometry package dependency in core AxonScope;
- no raw strings as preferred public API for closed/structured domains;
- no legacy compatibility aliases;
- public exports stay intentional;
- standby solver candidates do not re-enter public options or active runtime
  helpers;
- examples import or execute through public APIs;
- factorized paths avoid dense `Vext` when promised;
- observer-only paths do not retain full Vm unless requested;
- online and post-hoc analyses are cross-validated;
- stimulus-only updates reuse compatible static structures where implemented.

## 13.2 Architecture Acceptance

Typed API:

- closed domains use enums;
- extensible signals use typed descriptors;
- selectors use dedicated classes;
- identities use opaque identifier types;
- devices and precision use structured objects;
- raw strings remain limited to labels, metadata, and serialization boundaries.

Product boundary:

- AxonScope owns intrinsic axon geometry only;
- `AxonInstance` has no required world position;
- external geometry packages provide numerical footprints;
- core AxonScope does not depend on nerve geometry packages.

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

Migration:

- obsolete modules are deleted;
- forwarding compatibility aliases are removed;
- examples use public APIs directly;
- scientific reference tests pass.

---

# 14. Non-goals

Do not:

- make AxonScope own nerve geometry;
- make world position mandatory on `AxonInstance`;
- keep electrode geometry in the solver core;
- conflate footprint and stimulus;
- pre-sum all drives before execution;
- materialize `Vext[axon, time, position]` by default when footprint/drive
  structure is still available;
- create separate one-axon and population products;
- conflate myelination and cable formulation;
- merge analyses into raw numerical results;
- expose backend arrays as public contracts;
- preserve prototype compatibility;
- keep forwarding modules;
- introduce a generic kernel IR before a real second backend;
- create empty packages without responsibilities;
- expose archived solver experiments as public choices.

---

# 15. Final Target API Sketch

Build footprints outside AxonScope or with lightweight helpers:

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
        runtime=axs.Runtime.JAX,
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
AxonScope knows axons in intrinsic one-dimensional space, not nerve world space.
External geometry packages provide spatial extracellular footprints.
One ExtracellularDrive combines one footprint with one stimulus.
ExtracellularStimulation aggregates all drives.
The solver/prepared backend performs the drive sum without dense Vext by default.
Prototype APIs should be deleted once replaced.
```
