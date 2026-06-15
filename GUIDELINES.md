# AxonScope Master Architecture and Reorganization Plan

## Status

This document is the consolidated architecture reference for AxonScope.

It defines:

- the product boundary of AxonScope;
- the public object model;
- the API for one axon and large populations;
- intracellular and extracellular stimulation;
- the representation of extracellular footprints;
- planning, preparation, compilation, and execution;
- recording and numerical results;
- myelinated versus unmyelinated analysis semantics;
- single-cable and double-cable signal capabilities;
- callable-based sweeps, thresholds, and recruitment;
- the target source organization;
- the migration and deletion strategy;
- the tests and acceptance criteria.

The source tree, runnable examples, and tests are the implementation sources of
truth.

README and documentation prose must not override the behavior expressed by the
code.

## Current implementation status

Snapshot updated on 2026-06-15.

This document defines the target architecture. The codebase has implemented the
roadmap through Phase 7.5 for the current scalar and single-cable batch observer
surface. Phase 7.6 is the current evidence pass before the Phase 8 study APIs.
Phases 8-9 are still roadmap work.

| Phase | Status | Implemented surface | Didactic example |
| --- | --- | --- | --- |
| Phase 0 — Guardrails and baselines | Done | Architecture guardrails, public API cleanup checks, import-boundary checks, non-NRV baseline. | None; this is test/build infrastructure. |
| Phase 1 — Object model | Done | `AxonInstance`, root `AxonSimulation`, `AxonPopulation`, direct public diameter inspection. | `examples/advanced/example_08_root_axon_simulation.py`, `examples/advanced/example_09_axon_population.py` |
| Phase 2 — Typed and extracellular contracts | Done | Typed recording signals, position selectors, cable formulation, opaque identifiers, `ExtracellularFootprint`, `ExtracellularDrive`, `ExtracellularStimulation`, analytical footprint builders. | `examples/advanced/example_10_typed_recording_signals.py`, `examples/advanced/example_11_typed_position_selectors.py`, `examples/advanced/example_12_cable_formulation.py`, `examples/advanced/example_13_extracellular_footprint_drive.py` |
| Phase 2.5 — Hotpath evidence | Done | Opt-in benchmark spans, hotpath workload catalog, Colab GPU workflow. | `examples/advanced/example_14_hotpath_benchmarking.py` |
| Phase 3 — Planning and preparation | Done | Preparation signatures, internal prepared cohorts, lower planning/input overhead, footprint-oriented preparation path. | `examples/advanced/example_15_preparation_signatures.py` |
| Phase 4 — JAX isolation | Done for the current boundary | JAX batch and scalar execution now enter through `axonscope.backends.jax`; public/descriptive layers are guarded against direct JAX imports. Low-level numerical kernels still live under `solvers/` until a later kernel ownership cleanup. | None; this is an internal backend boundary. |
| Phase 5 — Canonical pool results | Done | `CohortResult`, `AxonSimulationResult`, `AxonResultView`, extensible `Signal` descriptors, `SignalId`, `RecordingManifest`, `RecordedSignal`, no public `list[SimResult]` pool result. | `examples/advanced/example_16_canonical_pool_results.py` |
| Phase 6 — Analyses | Done for the current public layer | Real `axs.analysis` package, analysis definitions, low-level post-hoc helpers, structured input requirements, per-axon statuses, population denominators, `AnalysisReport`, `result.analyze(...)` / `result.report(...)`, and online Vm observers for activation/peak-voltage cross-validation. | `examples/advanced/example_17_analysis_layer.py` |
| Phase 7 — Performance | Done for the current evidence layer | `axs.performance`, `AxonSimulation.estimate()`, simulation memory estimates, typed runtime/device/precision planning values, hotpath memory metadata, and `footprint_reuse_sweep`. Estimates surface dense `Vstim` and retained-`Vm` pressure so observer-only runs can be chosen deliberately. | `examples/advanced/example_14_hotpath_benchmarking.py` |
| Phase 7.5 — Solver-side observers | Done for scalar + single-cable batch observer-only runs | Public `axs.analysis.PeakVoltage` and `axs.analysis.Activation` definitions lower to compact solver observer state; scalar kernels and single-cable batch kernels update that state at every `dt`; `Recording.none()` returns trace-free `result.observations`. Double-cable batch observer-only execution currently falls back to scalar for correctness. | `examples/advanced/example_18_solver_side_observers.py` |
| Phase 7.6 — Realistic hotpath evidence | In progress | `realistic_mixed_population`, `hotpath_matrix`, and richer hotpath manifest metadata for model/formulation mix, diameter and compartment distributions, recording policy, observers, and per-simulation memory estimates. | Benchmark workloads documented in `benchmark/hotpaths/README.md`; no new public concept example required. |
| Phase 8 — Studies | Not started | Target: callable studies, reuse policies, retention policies, study result containers. | To add when callable study APIs land. |
| Phase 9 — Serialization and reference backend | Not started | Target: final schemas, typed serialization, NumPy reference backend validation. | To add after schemas are stable. |

Known implementation gaps against the final target:

- Public scalar `simulate(...)` still returns `SimResult`; pool runs return
  `AxonSimulationResult`. Decide before final docs/serialization whether scalar
  public runs also become `AxonSimulationResult`.
- `Recording.to_batch_options()` still lowers directly to solver batch options.
  The final boundary should be `Recording -> RecordingPlan -> validation ->
  backend lowering`.
- `axs.analysis` is now a real package, not a forwarding compatibility alias.
  Low-level post-hoc helpers live under `axs.analysis`, not under
  `axs.results.analysis`.
- Backend-neutral axon structure descriptors, cable capabilities, and richer
  semantic signals remain future work.
- Solver-side observer execution exists for scalar kernels and homogeneous
  single-cable batch kernels. Double-cable batch observer-only execution should
  only move back to the batch path once its compact observer state has the same
  per-`dt` guarantees and correctness tests.

---

# 1. Development-stage breaking-change policy

AxonScope has not yet been deployed as a stable public package.

The architecture should therefore optimize for a clean final design rather than
backward compatibility with prototype APIs.

The migration policy is:

```text
rename concepts directly
rewrite examples and tests
delete superseded modules
delete obsolete schemas and formats
keep one implementation path
```

Do not accumulate:

- deprecated aliases;
- compatibility wrappers;
- forwarding modules;
- `Legacy*` classes;
- duplicate scalar and population APIs;
- duplicate result models;
- old and new meanings for the same class name;
- obsolete benchmark readers;
- obsolete serialization readers;
- temporary modules that remain permanently.

Scientific behavior must remain protected by:

- reference tests;
- convergence tests;
- numerical regression tests;
- analysis-equivalence tests;
- cross-backend validation.

The following may change deliberately:

- import paths;
- class names;
- constructors;
- function signatures;
- internal array layouts;
- result shapes;
- benchmark formats;
- prototype serialization formats.

A migration is complete only when the replaced code path is deleted.

The target repository should contain:

```text
one concept
one public name
one execution path
one result model
```

---

# 2. Product boundary

AxonScope is a simulation package for axons and axon populations.

It should focus on:

- one-dimensional axon models;
- myelinated and unmyelinated axons;
- membrane dynamics;
- single-cable and double-cable formulations;
- intracellular electrical stimulation;
- extracellular potentials applied along axons;
- one-axon educational and debugging workflows;
- large population execution;
- recording;
- online and post-hoc scientific analyses;
- threshold, recruitment, and sweep studies;
- validation, reproducibility, and performance.

AxonScope should not own:

- detailed nerve geometry;
- fascicle geometry;
- tissue segmentation;
- three-dimensional axon trajectories;
- anatomical coordinate systems;
- electrode CAD;
- full volume-conductor modeling;
- finite-element meshing;
- finite-element field solving;
- surgical placement models.

Another package may own:

```text
nerve geometry
axon trajectories in world coordinates
electrode geometry
conductivity models
FEM or analytical field calculations
```

That package should provide AxonScope with an already-resolved extracellular
representation along each axon.

The core boundary is:

```text
external geometry / field package
    computes spatial extracellular footprints

AxonScope
    combines footprints with temporal stimuli
    runs membrane and cable dynamics
    records numerical outputs
    performs scientific analyses
```

---

# 3. Intrinsic axon space versus world geometry

AxonScope must know the intrinsic one-dimensional coordinate of each axon.

AxonScope does not need to know the axon's world position.

## 3.1 Intrinsic position

Intrinsic position describes location along the axon:

```text
s = 0 ... axon length
```

It is required for:

- compartment discretization;
- section boundaries;
- nodes and internodes;
- intracellular clamp placement;
- recording selectors;
- conduction velocity;
- applying extracellular footprints;
- reporting events and block locations.

Examples:

```python
axs.positions.At(2 * axs.mm)
axs.positions.Node(3)
axs.positions.Nodes()
axs.positions.DISTAL
axs.positions.Probes(count=16)
```

## 3.2 World position

World position includes:

```text
x
y
z
orientation
trajectory in a nerve
anatomical frame
```

These concepts belong outside the AxonScope simulation core.

`AxonInstance` should not require:

```text
position
orientation
trajectory
world coordinates
```

The only geometry entering AxonScope from the outside is the numerical
extracellular coupling already evaluated along the axon.

---

# 4. Public object model

The target public hierarchy is:

```text
Axon
    descriptive biological, membrane, layout, and cable model

AxonInstance
    one concrete parameterized occurrence of an Axon

AxonPopulation
    compact or heterogeneous collection of AxonInstance objects

Stimulus
    one temporal waveform

ExtracellularFootprint
    one static spatial extracellular transfer profile

ExtracellularDrive
    one footprint + one stimulus

ExtracellularStimulation
    aggregate of several ExtracellularDrive objects

AxonSimulation
    complete executable definition for one or many axons

AxonSimulationPlan
    backend-neutral execution plan

PreparedAxonSimulation
    reusable prepared host-side structures

CompiledAxonSimulation
    backend-specific executable and dynamic input schema

AxonSimulationResult
    canonical batch-backed numerical result

AxonResultView
    one-axon view

Analysis
    versioned scientific interpretation algorithm

AnalysisResult
    per-axon metrics, events, statuses, and population summaries

AxonStudy
    related simulations generated through callable updates

AxonStudyResult
    condition-indexed results and summaries
```

---

# 5. Typed public API and autocomplete policy

The public Python API should use typed values instead of raw strings whenever
the domain is known.

Goals:

- IDE autocomplete;
- static type checking;
- discoverable documentation;
- safer refactoring;
- early validation;
- fewer spelling errors;
- separation between identifiers that serialize to the same primitive type.

General rule:

```text
closed set
    Enum

extensible scientific concept
    typed object or registered descriptor

structured selection
    dedicated selector class

user-defined identity
    opaque identifier type

display text and free metadata
    string
```

Do not turn every concept into an enum.

Signals, analyses, position selectors, devices, and precision policies are
extensible or structured concepts and should remain objects.

## 5.1 Closed enums

Recommended enums:

```python
class Myelination(Enum):
    MYELINATED = "myelinated"
    UNMYELINATED = "unmyelinated"


class CableFormulation(Enum):
    SINGLE = "single_cable"
    DOUBLE = "double_cable"


class ApplicabilityPolicy(Enum):
    REQUIRE_ALL = "require_all"
    COMPATIBLE_ONLY = "compatible_only"
    MARK_NOT_APPLICABLE = "mark_not_applicable"


class ReusePolicy(Enum):
    AUTO = "auto"
    REQUIRE = "require"
    NONE = "none"


class RetentionPolicy(Enum):
    ALL = "all"
    RECORDINGS = "recordings"
    ANALYSES = "analyses"
    SUMMARY = "summary"


class AnalysisStatus(Enum):
    VALID = "valid"
    NOT_APPLICABLE = "not_applicable"
    MISSING_INPUT = "missing_input"
    NUMERICAL_FAILURE = "numerical_failure"
    UNDETERMINED = "undetermined"


class Runtime(Enum):
    AUTO = "auto"
    NUMPY = "numpy"
    JAX = "jax"


class CompartmentRole(Enum):
    NODE = "node"
    MYSA = "MYSA"
    FLUT = "FLUT"
    STIN = "STIN"
    CONTINUOUS_CABLE = "continuous_cable"
```

Preferred usage:

```python
result.select(
    myelination=axs.Myelination.MYELINATED,
)

result.select(
    formulation=axs.CableFormulation.DOUBLE,
)
```

Raw serialized values such as `"myelinated"` remain acceptable at file and
interoperability boundaries, but are not the preferred Python API.

## 5.2 Typed signals

Signals are extensible and should not be a closed enum.

```python
@dataclass(frozen=True)
class Signal[T]:
    id: SignalId
    quantity_type: type[T]
    unit: Unit
    description: str
```

Built-in descriptors:

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
recording = axs.Recording.center(
    axs.signals.MEMBRANE_VOLTAGE,
)

vm = result.signal(
    axs.signals.MEMBRANE_VOLTAGE,
)
```

Custom signals remain possible:

```python
MY_CUSTOM_CURRENT = axs.Signal[CurrentDensity](
    id=axs.SignalId("my_custom_current"),
    quantity_type=CurrentDensity,
    unit=axs.A_per_m2,
    description="Custom transmembrane current density",
)
```

## 5.3 Typed position selectors

Do not use raw selector strings such as:

```text
"all"
"distal"
"nodes"
"recorded"
```

Use typed selector objects:

```python
axs.positions.ALL
axs.positions.PROXIMAL
axs.positions.DISTAL
axs.positions.RECORDED

axs.positions.At(2 * axs.mm)
axs.positions.Node(3)
axs.positions.Node(-1)
axs.positions.Nodes()
axs.positions.Internodes()
axs.positions.SectionType(axs.CompartmentRole.FLUT)
axs.positions.Probes(count=16)
```

Selectors implement a common protocol:

```python
class PositionSelector(Protocol):
    def resolve(
        self,
        structure: AxonStructure,
        layout: AxonLayout,
    ) -> ResolvedPositions:
        ...
```

## 5.4 Opaque identifiers

Do not use one interchangeable string type for every identity.

Recommended types:

```python
@dataclass(frozen=True, order=True)
class AxonId:
    value: str


@dataclass(frozen=True, order=True)
class DriveId:
    value: str


@dataclass(frozen=True, order=True)
class SignalId:
    value: str


@dataclass(frozen=True, order=True)
class CohortId:
    value: str


@dataclass(frozen=True, order=True)
class ModelId:
    value: str
```

Example:

```python
CATHODE = axs.DriveId("cathode")

drive = axs.ExtracellularDrive(
    id=CATHODE,
    footprint=footprint,
    stimulus=stimulus,
)
```

This prevents accidental interchange between `AxonId`, `DriveId`, `SignalId`,
and other identifiers.

## 5.5 Typed runtime, device, and precision

Runtime is a closed enum.

Device selection and precision are structured objects:

```python
compiled = prepared.compile(
    runtime=axs.Runtime.JAX,
    device=axs.Device.gpu(index=0),
    precision=axs.PrecisionPolicy.float32(),
)
```

Preferred constructors:

```python
axs.Device.auto()
axs.Device.cpu()
axs.Device.gpu(index=0)

axs.PrecisionPolicy.float32()
axs.PrecisionPolicy.float64()
axs.PrecisionPolicy.mixed(...)
```

Do not make `"gpu"` or `"float32"` the primary public API.

## 5.6 Strings that remain appropriate

Strings remain appropriate for:

- labels;
- display names;
- descriptions;
- free-form user metadata;
- paths;
- serialized values;
- external interchange formats.

```python
instance = axs.AxonInstance(
    axon=axon,
    label="patient-A-fiber-42",
    metadata={
        "group": "large-diameter",
    },
)
```

## 5.7 Serialization

Typed Python values serialize to stable primitive values:

```json
{
  "myelination": "myelinated",
  "formulation": "double_cable",
  "signal": "membrane_voltage",
  "drive_id": "cathode"
}
```

Deserialization reconstructs typed values.

The serialized representation must not dictate an untyped in-memory API.

---

# 6. Axon, AxonInstance, AxonPopulation, and AxonSimulation

## 5.1 Axon

`Axon` is a reusable scientific description.

It owns:

- sections;
- layout;
- compartment roles;
- membrane models;
- diameter;
- cable formulation;
- myelination structure;
- initial conditions;
- temperature;
- biophysical parameters.

It does not own:

- simulation duration;
- backend selection;
- recording;
- world geometry;
- extracellular electrode geometry;
- compiled arrays;
- results.

## 5.2 AxonInstance

`AxonInstance` represents one concrete occurrence of an `Axon`.

```python
instance = axs.AxonInstance(
    axon=axon,
    label="axon-42",
    parameters={
        "diameter_scale": 1.05,
    },
    initial_state={
        "membrane_voltage": -70 * axs.mV,
    },
    intracellular=[local_clamp],
    metadata={
        "group": "large-fibers",
    },
)
```

It may contain:

- a stable identifier;
- a label;
- user metadata;
- per-axon parameter overrides;
- initial-state overrides;
- local intracellular stimulation.

It should not contain:

- world position;
- trajectory;
- electrode definitions;
- field geometry;
- extracellular footprint generation logic.

The current prototype object named `AxonSimulation` should be renamed directly
to `AxonInstance`.

No compatibility alias should remain.

## 5.3 AxonPopulation

`AxonPopulation` supports compact homogeneous and heterogeneous forms.

Homogeneous:

```python
population = axs.AxonPopulation.broadcast(
    axon=axon,
    count=10_000,
    parameters={
        "diameter": diameters,
    },
    labels=labels,
)
```

Heterogeneous:

```python
population = axs.AxonPopulation([
    axs.AxonInstance(axon_a, label="a"),
    axs.AxonInstance(axon_b, label="b"),
    axs.AxonInstance(
        axon_c,
        label="c",
        intracellular=[local_clamp],
    ),
])
```

The public semantics are identical.

Storage optimization is internal.

## 5.4 AxonSimulation

`AxonSimulation` is the complete executable definition.

```python
simulation = axs.AxonSimulation(
    axons=axon_or_population,
    intracellular=intracellular_stimulation,
    extracellular=extracellular_stimulation,
    duration=20 * axs.ms,
    dt=0.01 * axs.ms,
    recording=recording,
)
```

Accepted `axons` forms:

```text
Axon
AxonInstance
Sequence[Axon]
Sequence[AxonInstance]
AxonPopulation
```

All forms normalize to:

```text
AxonInstanceCollection[B]
```

---

# 7. Unified one-axon and population API

One axon and many axons use the same lifecycle:

```text
describe
→ validate
→ plan
→ prepare
→ compile
→ run
→ analyze
```

## 6.1 One axon

```python
simulation = axs.AxonSimulation(
    axons=axon,
    intracellular=[clamp],
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.full(axs.signals.MEMBRANE_VOLTAGE),
)

result = simulation.run()

result.single.plot.vm()
```

## 6.2 Population

```python
population = axs.AxonPopulation.broadcast(
    axon=axon,
    count=1000,
    parameters={
        "diameter": diameters,
    },
)

simulation = axs.AxonSimulation(
    axons=population,
    extracellular=extracellular_stimulation,
    duration=20 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.MEMBRANE_VOLTAGE),
)

result = simulation.run()
```

## 6.3 Internal execution

The backend may choose:

```text
scalar debug execution
batch execution with B = 1
batch execution with B > 1
several homogeneous cohorts
padded cohorts
scalar fallback
```

This must not change:

- scientific semantics;
- recording semantics;
- result semantics;
- units;
- events;
- error behavior.

A single axon is the smallest population, not a separate product.

---

# 8. Intracellular stimulation

Intracellular stimulation remains explicit and local to the cable model.

## 7.1 Shared or targeted stimulation

```python
simulation = axs.AxonSimulation(
    axons=population,
    intracellular=[
        axs.IntracellularCurrentClamp(
            target=axs.AxonSelector.indices([0, 3, 7]),
            position=50 * axs.um,
            stimulus=pulse,
        )
    ],
    ...
)
```

## 7.2 Instance-local stimulation

```python
instance = axs.AxonInstance(
    axon=axon,
    intracellular=[local_clamp],
)
```

## 7.3 Combination rule

```text
simulation-level intracellular stimulation
    shared or explicitly targeted

instance-level intracellular stimulation
    local to one instance

combination
    additive by default
```

Replacement or masking must require an explicit override object.

---

# 9. Extracellular architecture

The extracellular API is built from four concepts:

```text
ExtracellularFootprint
ExtracellularDrive
ExtracellularStimulation
ExtracellularPotential
```

The recommended path is factorized:

```text
one spatial footprint
+
one temporal stimulus
=
one ExtracellularDrive
```

Several drives are aggregated into:

```text
ExtracellularStimulation
```

The solver performs the sum during time integration.

---

# 10. ExtracellularFootprint

`ExtracellularFootprint` is a static spatial transfer profile.

It contains no time dimension and no waveform.

Conceptually:

```text
footprint[axon, intrinsic_position]
```

or, when shared:

```text
footprint[intrinsic_position]
```

## 9.1 Physical meaning

A footprint represents extracellular potential per unit command.

Typical unit:

```text
V / A
```

or an equivalent consistent unit.

For one drive:

```text
Vext_drive[a, t, x]
    = footprint[a, x] × stimulus[t]
```

## 9.2 One axon

```python
footprint = axs.ExtracellularFootprint(
    values=values,
    axon_ids=("axon-0",),
    positions=axon.compartment_positions,
)
```

## 9.3 Population

```python
footprint = axs.ExtracellularFootprint(
    values=values,
    axon_ids=population.ids,
    positions=position_support,
)
```

Logical shape:

```text
[axon, position]
```

## 9.4 Shared footprint

For simplified tests:

```python
footprint = axs.ExtracellularFootprint.shared(
    values=values,
    positions=position_support,
)
```

The planner should preserve the shared representation rather than eagerly
copying it across all axons.

## 9.5 Required metadata

A footprint should preserve:

```text
axon identifiers or shared flag
intrinsic position support
units
interpolation policy
sampling provenance
source identifier
reference convention
optional metadata
```

It should not contain:

```text
electrode CAD
world coordinates
nerve geometry
time samples
stimulus amplitude
```

---

# 11. ExtracellularDrive

`ExtracellularDrive` represents one independent extracellular contribution.

```python
CATHODE = axs.DriveId("cathode")

drive = axs.ExtracellularDrive(
    id=CATHODE,
    footprint=cathode_footprint,
    stimulus=cathode_stimulus,
)
```

The object groups:

```text
one footprint
one stimulus
one name
optional metadata
```

Its physical contribution is:

```text
Vdrive[a, t, x]
    = footprint[a, x] × stimulus[t]
```

This object is the user-facing unit corresponding to:

- one electrode contact;
- one source;
- one field basis component;
- one multipolar mode;
- one externally computed spatial mode.

The name `drive` is intentionally more general than `electrode`.

---

# 12. ExtracellularStimulation

`ExtracellularStimulation` aggregates multiple drives.

```python
extracellular = axs.ExtracellularStimulation([
    cathode_drive,
    anode_left_drive,
    anode_right_drive,
])
```

It is the object passed to `AxonSimulation`:

```python
simulation = axs.AxonSimulation(
    axons=population,
    extracellular=extracellular,
    duration=5 * axs.ms,
    dt=0.01 * axs.ms,
    recording=recording,
)
```

## 11.1 Mathematical model

For drives `d = 1 ... D`:

```text
Vext[a, t, x]
    = Σd footprint[d, a, x] × stimulus[d, t]
```

The sum is performed inside the solver or in a prepared linear forcing stage.

The full tensor:

```text
Vext[axon, time, position]
```

must not be materialized by default.

## 11.2 Collection API

```python
extracellular.drives
extracellular.names
extracellular["cathode"]
```

Immutable modifications:

```python
updated = extracellular.replace_drive(
    "cathode",
    stimulus=new_stimulus,
)
```

```python
updated = extracellular.add(
    axs.ExtracellularDrive(...)
)
```

```python
updated = extracellular.remove("anode-right")
```

Each operation returns a new object.

## 11.3 Validation

The collection validates:

- unique names;
- compatible physical units;
- axon identifier coverage;
- compatible intrinsic supports;
- stimulus duration and sampling;
- interpolation policy;
- duplicate or conflicting sources;
- missing footprint rows.

Example error:

```text
ExtracellularDriveValidationError

Drive:
  cathode

Population axons:
  axon-0
  axon-1
  axon-2

Footprint available for:
  axon-0
  axon-2

Missing:
  axon-1
```

---

# 13. Dense extracellular fallback

A fully precomputed potential remains useful for general cases.

```python
potential = axs.ExtracellularPotential(
    values=values,
    dims=("axon", "time", "position"),
    axon_ids=population.ids,
    positions=position_support,
)
```

This supports:

- time-dependent non-separable fields;
- imported simulation data;
- experimental potentials;
- reference tests;
- arbitrary external solvers.

It is a fallback, not the preferred representation.

The planner should warn when a dense representation is large:

```text
Extracellular input
  representation: dense potential
  estimated size: 18.4 GB

Recommendation
  provide ExtracellularStimulation with factorized drives
```

---

# 14. External geometry package contract

A separate package may own:

```text
nerve geometry
fiber trajectories
electrode geometry
conductivity
FEM or analytical field solving
```

It should output one footprint per independent source:

```python
footprints = geometry_package.compute_footprints(
    fibers=fiber_trajectories,
    sources=electrodes,
    target_axons=population,
)
```

It may then construct AxonScope objects:

```python
drives = [
    axs.ExtracellularDrive(
        name=source.name,
        footprint=footprints[source.name],
        stimulus=stimuli[source.name],
    )
    for source in sources
]

extracellular = axs.ExtracellularStimulation(drives)
```

AxonScope must not depend on the geometry package.

The contract between packages is numerical:

```text
axon identifiers
intrinsic position support
footprint values
units
provenance
```

---

# 15. Analytical field helpers

AxonScope may provide lightweight analytical helpers for:

- examples;
- validation;
- tests;
- educational use;
- simple standalone simulations.

These helpers should be peripheral.

Recommended namespace:

```text
axonscope.analytical_fields
```

Potential modules:

```text
analytical_fields/
├── point_source.py
├── line_source.py
├── uniform_field.py
└── builders.py
```

The helpers produce footprints only.

They do not enter the solver.

## 14.1 One point source

```python
footprint = axs.analytical_fields.point_source(
    axon=axon,
    axial_position=2 * axs.mm,
    radial_distance=500 * axs.um,
    conductivity=0.3 * axs.S_per_m,
)
```

## 14.2 Population point-source helper

```python
footprint = axs.analytical_fields.point_source_population(
    axons=population,
    axial_positions=axial_positions,
    radial_distances=radial_distances,
    conductivity=0.3 * axs.S_per_m,
)
```

This helper may accept simplified geometric parameters.

It must not make world position a required property of `AxonInstance`.

## 14.3 Multiple sources

```python
cathode_footprint = axs.analytical_fields.point_source(...)
anode_left_footprint = axs.analytical_fields.point_source(...)
anode_right_footprint = axs.analytical_fields.point_source(...)
```

Then create drives explicitly:

```python
extracellular = axs.ExtracellularStimulation([
    axs.ExtracellularDrive(
        name="cathode",
        footprint=cathode_footprint,
        stimulus=cathode_stimulus,
    ),
    axs.ExtracellularDrive(
        name="anode-left",
        footprint=anode_left_footprint,
        stimulus=anode_left_stimulus,
    ),
    axs.ExtracellularDrive(
        name="anode-right",
        footprint=anode_right_footprint,
        stimulus=anode_right_stimulus,
    ),
])
```

The analytical helper owns only spatial calculation.

`Stimulus` owns time.

The solver combines both.

---

# 16. Preparation of extracellular drives

The public object model is object-oriented:

```text
ExtracellularStimulation
    tuple[ExtracellularDrive, ...]
```

Preparation converts it into efficient arrays.

Possible prepared layout:

```text
footprints[drive, axon, position]
stimuli[time, drive]
```

or:

```text
footprints[axon, drive, position]
stimuli[time, drive]
```

At each time step:

```python
stimulus_t = stimuli[t]

vext_t = einsum(
    "dax,d->ax",
    footprints,
    stimulus_t,
)
```

This produces:

```text
vext_t[axon, position]
```

without storing:

```text
Vext[axon, time, position]
```

## 15.1 Linear operator preapplication

When the cable forcing is linear:

```text
forcing_ext = Lext(Vext)
```

preparation may compute per-drive transformed footprints:

```text
prepared_footprint[d, a, x]
    = Lext(footprint[d, a, x])
```

Then:

```text
forcing_ext[a, t, x]
    = Σd prepared_footprint[d, a, x]
         × stimulus[d, t]
```

This preserves the critical separation:

```text
spatial footprint
    prepared once

temporal stimulus
    dynamic and replaceable
```

## 15.2 Shared footprints

A shared footprint should remain logically:

```text
footprint[drive, position]
```

when possible.

The backend should broadcast without eager duplication.

## 15.3 Dynamic stimulus inputs

Compiled execution should accept sampled stimuli as dynamic inputs:

```python
compiled.run(
    extracellular_stimuli=sampled_stimuli,
)
```

Prepared footprints should remain reusable when only waveforms change.

---

# 17. Sweeps with extracellular drives

The drive model is designed for efficient callable-based studies.

## 16.1 Replace one stimulus

```python
def update_amplitude(simulation, amplitude):
    extracellular = simulation.extracellular.replace_drive(
        axs.DriveId("cathode"),
        stimulus=build_cathodic_stimulus(amplitude),
    )

    return simulation.replace(
        extracellular=extracellular,
    )
```

Reusable artifacts:

```text
axon preparation
footprints
spatial operators
cohort structure
compiled executable
```

Only temporal stimulus samples change.

## 16.2 Replace several stimuli

```python
def update_configuration(simulation, amplitude):
    extracellular = (
        simulation.extracellular
        .replace_drive(
            axs.DriveId("cathode"),
            stimulus=build_stimulus(-amplitude),
        )
        .replace_drive(
            axs.DriveId("anode-left"),
            stimulus=build_stimulus(0.5 * amplitude),
        )
        .replace_drive(
            axs.DriveId("anode-right"),
            stimulus=build_stimulus(0.5 * amplitude),
        )
    )

    return simulation.replace(
        extracellular=extracellular,
    )
```

## 16.3 Replace a footprint

```python
def update_spatial_configuration(simulation, condition):
    return simulation.replace(
        extracellular=condition.extracellular_stimulation,
    )
```

The planner compares signatures.

```text
same drive count and array shapes
    dynamic update may be possible

same compile shape but changed footprint values
    reuse compilation, update prepared/device buffers

changed drive count or static shape
    reprepare or recompile affected cohorts
```

---

# 18. Lifecycle

## 17.1 Validation

```python
report = simulation.validate()
```

Validation covers:

- units;
- axon structure;
- population identities;
- stimulation targeting;
- drive names;
- footprint coverage;
- intrinsic position support;
- recording applicability;
- memory feasibility;
- backend capabilities.

## 17.2 Planning

```python
plan = simulation.plan()
```

Planning determines:

- normalized instances;
- myelination classes;
- cable formulations;
- compatible cohorts;
- footprint structure;
- drive structure;
- recording requirements;
- analysis requirements;
- expected shapes;
- memory estimates;
- candidate runtimes.

Planning must not:

- create device arrays;
- compile code;
- import concrete JAX kernels;
- execute solvers.

## 17.3 Preparation

```python
prepared = simulation.prepare()
```

Preparation may compute and cache:

- flattened cable structures;
- section and compartment roles;
- membrane assignments;
- intrinsic positions;
- cable operators;
- footprint resampling;
- transformed footprints;
- compact intracellular sources;
- recording indices;
- observer plans;
- cohort metadata.

## 17.4 Compilation

```python
compiled = prepared.compile(
    runtime=axs.Runtime.JAX,
    device=axs.Device.gpu(index=0),
    precision=axs.PrecisionPolicy.float32(),
)
```

Compilation may perform:

- backend lowering;
- static-shape specialization;
- solver specialization;
- JIT or AOT compilation;
- cache lookup;
- device memory planning.

## 17.5 Execution

```python
result = compiled.run(
    extracellular_stimuli=dynamic_stimuli,
)
```

Execution reuses all compatible static structures.

---

# 19. Planning and explainability

```python
print(simulation.plan().explain())
```

Example:

```text
AxonSimulation
  1000 axons
  600 unmyelinated
  400 myelinated
  3 execution cohorts

Extracellular stimulation
  3 drives
  cathode
  anode-left
  anode-right

Footprints
  representation: factorized
  shape: [drive, axon, position]
  spatial preparation: cached
  full Vext tensor: not materialized

Stimuli
  dynamic inputs
  sampled shape: [time, drive]

Runtime
  JAX GPU

Memory
  footprints: 48 MB
  stimuli: 0.2 MB
  runtime state: 110 MB
  output: 12 MB
```

The plan should explain automatic decisions.

---

# 20. Myelination and cable formulation

The architecture must not conflate:

```text
biological organization
    myelinated
    unmyelinated

numerical formulation
    single cable
    double cable
```

The primary scientific distinction for recording and analysis is:

```text
myelinated versus unmyelinated
```

Cable formulation determines numerical state and additional signals.

Examples:

```text
myelinated + single cable
myelinated + double cable
unmyelinated + single cable
```

A myelinated single-cable model still has:

- nodes;
- internodes;
- saltatory propagation;
- node-based recording selectors.

A double-cable formulation may additionally expose:

- intracellular potential;
- periaxonal potential;
- periaxonal current.

---

# 21. Axon structure descriptors

Each axon should expose a backend-neutral structure descriptor.

Myelinated:

```python
AxonStructure(
    myelination=axs.Myelination.MYELINATED,
    compartment_roles={
        "node",
        "MYSA",
        "FLUT",
        "STIN",
    },
)
```

Unmyelinated:

```python
AxonStructure(
    myelination=axs.Myelination.UNMYELINATED,
    compartment_roles={
        "continuous_cable",
    },
)
```

The descriptor supports:

- semantic position selectors;
- recording validation;
- analysis applicability;
- result metadata;
- cohort inspection.

---

# 22. Cable capabilities and semantic signals

Each formulation declares available signals.

Single cable:

```python
CableCapabilities(
    formulation="single_cable",
    native_signals={
        "membrane_voltage",
    },
)
```

Double cable:

```python
CableCapabilities(
    formulation=axs.CableFormulation.DOUBLE,
    native_signals={
        "intracellular_potential",
        "periaxonal_potential",
    },
    derived_signals={
        "membrane_voltage",
    },
)
```

For double cable:

```text
membrane_voltage =
    intracellular_potential - periaxonal_potential
```

The backend owns this reconstruction.

Analyses request semantic signals.

They do not encode backend equations.

---

# 23. Recording

The recording contract combines:

```text
semantic signal
axon structure
compartment role
intrinsic position selector
temporal selector
cable capability
applicability policy
```

## 22.1 Myelinated selectors

```python
axs.positions.Nodes()
axs.positions.Node(index)
axs.positions.Internodes()
axs.positions.SectionType(axs.CompartmentRole.MYSA)
axs.positions.SectionType(axs.CompartmentRole.FLUT)
axs.positions.SectionType(axs.CompartmentRole.STIN)
```

## 22.2 Unmyelinated selectors

```python
axs.positions.At(2 * axs.mm)
axs.positions.PROXIMAL
axs.positions.DISTAL
axs.positions.Probes(count=16)
```

## 22.3 Semantic signals

```text
membrane_voltage
ionic_current
membrane_conductance
state_variables
intracellular_potential
periaxonal_potential
```

Short aliases may exist, but semantic names are canonical.

## 22.4 Applicability

```python
axs.Recording.signals(
    "membrane_voltage",
    positions=axs.positions.Nodes(),
    applicability=axs.ApplicabilityPolicy.COMPATIBLE_ONLY,
)
```

Unmyelinated axons are explicitly not applicable for node selectors.

A periaxonal request may require:

```text
myelinated structure with requested roles
double-cable signal capability
```

## 22.5 No artificial completion

Do not create meaningless arrays for unsupported combinations.

Do not fill unavailable rows silently with `NaN`.

Use explicit applicability and dense compatible storage.

## 22.6 Dependency direction

```text
Recording
→ RecordingPlan
→ axon-structure validation
→ cable-capability validation
→ backend lowering
```

`Recording` must not import solver-specific option classes.

Migration note: if the current implementation still lowers `Recording` directly
to solver or batch options, that is a transitional convenience. The target
boundary is `Recording -> RecordingPlan -> validation -> backend lowering`.

---

# 24. Canonical numerical result

Use one public result type:

```text
AxonSimulationResult
```

for:

- one axon;
- homogeneous populations;
- heterogeneous populations;
- mixed myelination;
- mixed cable formulations;
- multiple execution cohorts.

Migration note: the implementation may temporarily keep a scalar `SimResult`
while pool results move first. Before serialization or final docs, decide
whether scalar public runs also return `AxonSimulationResult` or whether
`SimResult` remains an explicitly scoped single-run convenience.

## 23.1 Logical schema

```text
AxonSimulationResult
├── axons: AxonMetadataTable
├── cohorts: tuple[CohortResult, ...]
├── recording: RecordingManifest
├── online_analyses: AnalysisCollection
├── execution: ExecutionMetadata
└── diagnostics: DiagnosticCollection
```

## 23.2 Axon metadata

```text
axon_id
input_index
label
myelination
formulation
compartment_roles
cohort_id
model_id
geometry_id
user metadata
```

No world position is required.

## 23.3 Dense cohort storage

Example:

```text
cohort 0
    600 unmyelinated single-cable axons
    membrane_voltage

cohort 1
    250 myelinated single-cable axons
    membrane_voltage
    node/internode metadata

cohort 2
    150 myelinated double-cable axons
    membrane_voltage
    intracellular_potential
    periaxonal_potential
    node/internode metadata
```

Each cohort stores dense arrays.

The global result preserves input order logically.

## 23.4 Signal access

```python
vm = result.signal(axs.signals.MEMBRANE_VOLTAGE)
```

Logical shape:

```text
(axon, time, recorded_position)
```

Partially available signal:

```python
double_result = result.select(
    formulation=axs.CableFormulation.DOUBLE,
)

ve = double_result.signal(
    axs.signals.PERIAXONAL_POTENTIAL,
)
```

Partial availability must never be silent.

## 23.5 One-axon view

```python
result.single
result.axon(index)
```

return:

```text
AxonResultView
```

For one axon:

```text
result.single == result.axon(0)
```

## 23.6 Numerical result immutability

The numerical result contains:

- recorded signals;
- final states;
- online observer outputs;
- execution metadata;
- diagnostics.

Post-hoc analyses return separate objects.

---

# 25. Scientific analyses

The solver produces signals and states.

Analyses produce:

- events;
- metrics;
- statuses;
- population summaries.

Recommended objects:

```python
axs.analysis.Activation(...)
axs.analysis.ConductionVelocity(...)
axs.analysis.Latency(...)
axs.analysis.ConductionBlock(...)
axs.analysis.SpikeCount(...)
axs.analysis.PeakVoltage(...)
axs.analysis.PeriaxonalDepolarization(...)
```

Namespace note: `axs.analysis` is the real public analysis namespace. It must
not be implemented as a forwarding compatibility alias to
`axs.results.analysis`.

## 24.1 Requirements

Each analysis declares:

```text
required semantic signals
supported myelination classes
required compartment roles
required cable capabilities
required positions
post-hoc support
online-observer support
algorithm version
```

## 24.2 Activation

Unmyelinated:

```python
axs.analysis.Activation(
    signal=axs.signals.MEMBRANE_VOLTAGE,
    positions=axs.positions.DISTAL,
    threshold=0 * axs.mV,
)
```

Myelinated:

```python
axs.analysis.Activation(
    signal=axs.signals.MEMBRANE_VOLTAGE,
    positions=axs.positions.Node(-1),
    threshold=0 * axs.mV,
)
```

## 24.3 Conduction velocity

Unmyelinated strategy:

```text
crossing times at two physical positions
distance / time difference
```

Myelinated strategy:

```text
crossing times at two nodes
node-center distance / time difference
```

## 24.4 Myelinated-specific analyses

Examples:

```text
node activation sequence
saltatory velocity
failed-node detection
nodal block
internodal delay
node-to-node jitter
```

## 24.5 Unmyelinated-specific analyses

Examples:

```text
continuous propagation profile
spatial spike width
local velocity versus position
continuous block location
```

## 24.6 Cable-specific analyses

```python
axs.analysis.PeriaxonalDepolarization(...)
```

requires:

```text
periaxonal_potential
double-cable capability
```

Its applicability comes from signal capability, not myelination alone.

## 24.7 Statuses

Per-axon metrics contain:

```text
value[axon]
status[axon]
```

Statuses:

```text
VALID
NOT_APPLICABLE
MISSING_INPUT
NUMERICAL_FAILURE
UNDETERMINED
```

## 24.8 Events

Variable-length events use columnar storage:

```text
event_type[event]
axon_index[event]
time[event]
position[event]
value[event]
direction[event]
```

## 24.9 Population summaries

```python
analysis.population.n_total
analysis.population.n_applicable
analysis.population.n_valid
analysis.population.n_failed
```

Every aggregate records its denominator.

## 24.10 Post-hoc and online

Post-hoc:

```python
activation = result.analyze(
    axs.analysis.Activation(...)
)
```

Online:

```python
simulation = axs.AxonSimulation(
    ...,
    recording=axs.Recording(
        traces=[],
        analyses=[activation_definition],
    ),
)
```

Online and post-hoc definitions must be cross-validated.

## 24.11 Missing input

Post-hoc analysis must not rerun a simulation silently.

It raises a structured error describing the required recording.

---

# 26. Reports

To associate a numerical result with analyses:

```text
AxonSimulationReport
├── simulation_result
└── analyses
```

Example:

```python
report = result.report(
    axs.analysis.Activation(...),
    axs.analysis.ConductionVelocity(...),
)
```

The original numerical result remains immutable.

---

# 27. Studies and callable updates

`AxonStudy` orchestrates related variants of a base `AxonSimulation`.

The canonical update mechanism is a callable.

## 26.1 Update contract

```python
update(
    base_simulation: AxonSimulation,
    condition: Condition,
) -> AxonSimulation
```

The callable should:

- avoid mutating the base;
- return a new simulation;
- avoid hidden side effects;
- make the condition explicit.

## 26.2 Sweep

```python
study = simulation.sweep(
    values=conditions,
    update=update_condition,
    reuse=axs.ReusePolicy.AUTO,
)

study_result = study.run()
```

## 26.3 Threshold

```python
threshold = simulation.find_threshold(
    bounds=(0 * axs.mA, 5 * axs.mA),
    update=update_amplitude,
    criterion=activation_definition,
    reuse=axs.ReusePolicy.AUTO,
)
```

## 26.4 Recruitment

```python
study = simulation.recruitment_sweep(
    values=amplitudes,
    update=update_amplitude,
    criterion=activation_definition,
    reuse=axs.ReusePolicy.REQUIRE,
    retain=axs.RetentionPolicy.ANALYSES,
)
```

## 26.5 Reuse policies

```text
reuse=axs.ReusePolicy.AUTO
    reuse compatible plans, preparation, and compilation

reuse=axs.ReusePolicy.REQUIRE
    fail if a condition violates the required reuse boundary

reuse=axs.ReusePolicy.NONE
    treat every condition independently
```

## 26.6 Structural signatures

May include:

```text
axon count
myelination classes
membrane programs
cable formulations
compartment roles
number of extracellular drives
footprint shapes
recording contract
online observer contract
dtype
solver algorithm
```

Dynamic values may include:

```text
stimulus samples
amplitudes
delays
waveform parameters
dynamic membrane parameters
initial states
```

## 26.7 Callable reproducibility

Lambdas are allowed.

AxonScope should not claim that every lambda is serializable.

For stronger reproducibility, recommend named functions or frozen callable
dataclasses.

---

# 28. Study results

```text
AxonStudyResult
├── conditions
├── simulation identities
├── per-condition analyses
├── aggregate metrics
├── execution metadata
└── optional retained simulation outputs
```

Retention:

```text
retain="all"
retain="recordings"
retain=axs.RetentionPolicy.ANALYSES
retain="summary"
```

Threshold and recruitment should not retain every trace by default.

---

# 29. Resource estimation and diagnostics

```python
estimate = simulation.estimate()
```

Report:

```text
axon count
myelination classes
cohorts
number of extracellular drives
footprint memory
stimulus memory
solver-state memory
recording memory
observer memory
peak memory
compile count
recommended device
recommended chunking
```

Numerical failures preserve:

```text
time
axon
compartment
compartment role
node index when applicable
variable
last finite value
cohort
backend
possible causes
```

---

# 30. Reproducibility and serialization

Potential APIs:

```python
simulation.save("simulation.axs.json")
result.save("result.axs")
study_result.save("study.axs")
```

A bundle may contain:

- simulation definition;
- axon model identities;
- myelination metadata;
- cable formulations;
- footprint descriptors;
- drive definitions;
- stimulus definitions;
- recording manifest;
- analysis definitions;
- backend and device;
- precision;
- environment information.

Only final schemas should receive readers and writers.

Do not maintain readers for prototype formats.

---

# 31. Observability

Measure separately:

```text
planning
preparation
footprint resampling
footprint operator application
compilation
stimulus sampling
execution
recording
online analysis
result assembly
post-hoc analysis
study orchestration
```

Suggested events:

```text
planning.build
preparation.geometry
preparation.membrane
preparation.extracellular_footprints
preparation.intracellular
preparation.recording
compilation.lower
compilation.compile
execution.enqueue
execution.wait
recording.finalize
analysis.online.finalize
results.assemble
analysis.posthoc
study.condition
```

---

# 32. Target source tree

```text
src/axonscope/
├── __init__.py
│
├── core/
│   ├── units.py
│   ├── validation.py
│   ├── errors.py
│   ├── precision.py
│   ├── identifiers.py
│   ├── enums.py
│   ├── device.py
│   └── serialization.py
│
├── axons/
│   ├── axon.py
│   ├── section.py
│   ├── layout.py
│   ├── myelinated.py
│   └── unmyelinated.py
│
├── axon_structure/
│   ├── myelination.py
│   ├── compartment_roles.py
│   └── selectors.py
│
├── membranes/
│   ├── model.py
│   ├── builtins.py
│   └── section_layout.py
│
├── stimulation/
│   ├── stimuli.py
│   ├── intracellular.py
│   └── targeting.py
│
├── extracellular/
│   ├── footprint.py
│   ├── drive.py
│   ├── stimulation.py
│   ├── potential.py
│   ├── validation.py
│   └── serialization.py
│
├── analytical_fields/
│   ├── point_source.py
│   ├── line_source.py
│   ├── uniform_field.py
│   └── builders.py
│
├── recording/
│   ├── spec.py
│   ├── selectors.py
│   ├── events.py
│   └── summaries.py
│
├── cable/
│   ├── formulation.py
│   ├── capabilities.py
│   └── signal_definitions.py
│
├── signals/
│   ├── signal.py
│   ├── identifiers.py
│   ├── registry.py
│   └── builtins.py
│
├── simulations/
│   ├── instance.py
│   ├── population.py
│   ├── simulation.py
│   ├── normalization.py
│   ├── prepared.py
│   ├── compiled.py
│   └── estimate.py
│
├── planning/
│   ├── plan.py
│   ├── cohorts.py
│   ├── compatibility.py
│   ├── signatures.py
│   ├── inspection.py
│   └── estimates.py
│
├── preparation/
│   ├── geometry.py
│   ├── axon.py
│   ├── membrane.py
│   ├── intracellular.py
│   ├── extracellular.py
│   ├── recording.py
│   ├── cohort.py
│   └── prepared.py
│
├── compilation/
│   ├── contracts.py
│   ├── artifacts.py
│   ├── cache.py
│   └── keys.py
│
├── backends/
│   ├── base.py
│   ├── registry.py
│   ├── capabilities.py
│   ├── numpy/
│   └── jax/
│       ├── runtime.py
│       ├── lowering.py
│       ├── intracellular.py
│       ├── extracellular.py
│       ├── recording.py
│       ├── profiler.py
│       ├── membrane/
│       │   ├── contracts.py
│       │   ├── state.py
│       │   ├── update_plans.py
│       │   ├── batching.py
│       │   ├── channels/
│       │   └── models/
│       └── solver/
│           ├── runtime.py
│           ├── scan.py
│           ├── linear.py
│           ├── forcing.py
│           ├── single_cable.py
│           ├── double_cable.py
│           └── observers.py
│
├── execution/
│   ├── engine.py
│   ├── local.py
│   ├── group_runner.py
│   ├── progress.py
│   └── result_assembly.py
│
├── results/
│   ├── simulation.py
│   ├── cohort.py
│   ├── recordings.py
│   ├── signals.py
│   ├── states.py
│   ├── events.py
│   ├── metrics.py
│   ├── status.py
│   ├── views.py
│   ├── report.py
│   ├── study.py
│   ├── validation.py
│   └── serialization.py
│
├── analysis/
│   ├── base.py
│   ├── requirements.py
│   ├── applicability.py
│   ├── activation.py
│   ├── velocity.py
│   ├── latency.py
│   ├── block.py
│   ├── peaks.py
│   ├── spikes.py
│   ├── myelinated.py
│   ├── unmyelinated.py
│   └── periaxonal.py
│
├── studies/
│   ├── study.py
│   ├── updates.py
│   ├── threshold.py
│   ├── recruitment.py
│   └── sweeps.py
│
├── observability/
│   ├── session.py
│   ├── events.py
│   ├── report.py
│   ├── metadata.py
│   └── profiler.py
│
└── visualization/
    ├── axons.py
    ├── results.py
    ├── activation.py
    └── planning.py
```

Do not create empty packages without moving real responsibilities.

Do not retain forwarding modules for obsolete import paths.

---

# 33. Dependency rules

Required direction:

```text
core
  ↑
domain descriptions
  ↑
planning
  ↑
preparation
  ↑
backend lowering
  ↑
execution
```

Results form a backend-neutral branch:

```text
core + domain metadata
          ↑
        results
          ↑
 analysis / visualization
```

Studies orchestrate:

```text
simulations + execution + analysis
                 ↑
              studies
```

Forbidden dependencies:

- `axons/` imports JAX;
- `membranes/` imports JAX implementations;
- `extracellular/` imports geometry packages;
- `analytical_fields/` enters solver execution directly;
- `recording/` imports solver options;
- `planning/` creates device arrays;
- `planning/` calls backend lowering;
- `results/` depends on JAX array types;
- analyses encode backend equations;
- domain objects eagerly import visualization;
- internal modules import top-level `axonscope`;
- mutable global precision silently changes compile identity.

---

# 34. Current-to-target migration map

## 33.1 Current AxonSimulation

```text
axon_simulation.py
    rename semantics to simulations/instance.py
    class becomes AxonInstance
    delete old module
```

Then introduce:

```text
simulations/simulation.py
    new root AxonSimulation
```

## 33.2 Axons

```text
axons/axon.py
    keep descriptive

axons/section.py
    keep descriptive

axons/layout.py
    keep intrinsic layout responsibilities

axons/flattened.py
    move to preparation/geometry.py

axons/plotting.py
    move to visualization/axons.py
```

Remove world-position responsibility from instances.

## 33.3 Existing extracellular contexts

Current descriptive electrode/context objects should be split.

Target:

```text
core simulation path
    ExtracellularFootprint
    ExtracellularDrive
    ExtracellularStimulation
    ExtracellularPotential

analytical helper path
    PointSource
    LineSource
    UniformField
```

Electrode geometry must not remain a core solver dependency.

Existing analytical contexts should be rewritten as footprint builders.

## 33.4 Stimulation

```text
Stimulus
    remains temporal

intracellular stimulation
    remains under stimulation/

extracellular spatial transfer
    moves to extracellular/
```

## 33.5 Recording

```text
recording.py
    split into recording/spec.py and selectors.py

Recording → solver options
    remove

backend lowering
    move to preparation/recording.py and backends/jax/recording.py
```

## 33.6 Dispatcher

```text
dispatcher/plan.py
    split into planning/

dispatcher/inspection.py
    move to planning/inspection.py

dispatcher/progress.py
    move to execution/progress.py

dispatcher/runtime_batches.py
    split into preparation/ and backend/

dispatcher/execution.py
    split into execution/
```

Delete `dispatcher/` when empty.

## 33.7 Membranes and solvers

```text
channel_models/
    move to backends/jax/membrane/

icm/
    split between preparation/membrane.py
    and backends/jax/membrane/batching.py

solvers/*
    move implementation to backends/jax/solver/
```

Delete old modules after migration.

## 33.8 Results and analyses

```text
SimResult
    replace with AxonSimulationResult and AxonResultView

list[SimResult]
    delete

results/activation.py
    move algorithm to analysis/activation.py

results/analysis.py
    split into analysis modules

results/visualization.py
    move to visualization/
```

---

# 35. Precision and cache identity

Introduce explicit precision:

```python
PrecisionPolicy(
    state_dtype="float32",
    solver_dtype="float32",
    accumulation_dtype="float32",
)
```

Semantic identity may include:

```text
axon model hash
myelination structure hash
layout hash
footprint identity
drive topology
recording contract
observer contract
```

Prepared identity may include:

```text
semantic identity
discretization
cohort shape
resampled footprints
transformed footprints
```

Compiled identity may include:

```text
prepared identity
backend
backend version
device class
precision
static shapes
drive count
solver algorithm
optimization flags
```

Stimulus samples should remain dynamic when shape-compatible.

---

# 36. Test architecture

```text
tests/
├── architecture/
│   ├── test_import_boundaries.py
│   ├── test_no_jax_in_domain.py
│   ├── test_no_geometry_dependency.py
│   ├── test_no_raw_string_public_api.py
│   ├── test_no_legacy_paths.py
│   └── test_public_exports.py
├── unit/
│   ├── axons/
│   ├── axon_structure/
│   ├── stimulation/
│   ├── extracellular/
│   ├── analytical_fields/
│   ├── recording/
│   ├── simulations/
│   ├── planning/
│   ├── preparation/
│   ├── backends/
│   ├── execution/
│   ├── results/
│   ├── analysis/
│   └── studies/
├── integration/
│   ├── test_single_axon.py
│   ├── test_population.py
│   ├── test_multiple_extracellular_drives.py
│   ├── test_shared_footprint.py
│   ├── test_dense_extracellular_potential.py
│   ├── test_mixed_myelination.py
│   ├── test_prepare_compile_run.py
│   └── test_recording_applicability.py
├── scientific/
│   ├── passive_cable/
│   ├── myelinated/
│   ├── unmyelinated/
│   ├── point_source_reference/
│   ├── activation/
│   ├── velocity/
│   └── convergence/
└── performance/
    ├── test_no_recompile_on_stimulus_change.py
    ├── test_footprint_reuse.py
    ├── test_no_dense_vext_materialization.py
    ├── test_memory_contracts.py
    └── test_batch_scaling.py
```

Critical tests:

```text
closed public domains use enums
built-in signals use typed descriptors
position selectors are typed objects
DriveId cannot be passed where AxonId is expected
serialization reconstructs typed values
AxonInstance has no required world position
core extracellular objects contain no electrode geometry
one drive equals footprint × stimulus
multiple drives sum inside execution
stimulus changes do not rebuild footprints
full Vext tensor is not materialized on factorized paths
analytical helpers output footprints only
pool execution uses the canonical result model
final scalar execution either uses the canonical result model or is documented
as an explicit single-run convenience
```

---

# 37. Normative examples

Examples should include:

```text
basic/
    stimulus waveforms
    point-source footprint generation
    one intracellular axon
    one extracellular drive
    multiple extracellular drives
    homogeneous population
    velocity analysis

advanced/
    heterogeneous population
    custom axon
    shared footprint
    per-axon footprints
    mixed myelination
    recording applicability
    online activation
    callable recruitment sweep
```

Current didactic advanced examples:

```text
example_01_pool_dispatch_nrv.py
    heterogeneous pool dispatch inspection with optional NRV-generated fibers
example_02_layout_options.py
    advanced axon layout options
example_03_custom_axon_from_scratch.py
    custom axon construction
example_04_stimulation_contexts.py
    stimulation context variants
example_05_recording_options.py
    recording policy options
example_06_activation_criterion.py
    post-hoc activation criterion
example_07_recruitment_curve.py
    recruitment curve workflow
example_08_root_axon_simulation.py
    root executable AxonSimulation
example_09_axon_population.py
    AxonPopulation as first-class cohort
example_10_typed_recording_signals.py
    typed/extensible recording signal descriptors
example_11_typed_position_selectors.py
    typed position selectors
example_12_cable_formulation.py
    typed cable formulation selection
example_13_extracellular_footprint_drive.py
    extracellular footprints, drives, and stimulation
example_14_hotpath_benchmarking.py
    memory estimates plus opt-in hotpath benchmarking and traces
example_15_preparation_signatures.py
    deterministic preparation signatures
example_16_canonical_pool_results.py
    canonical pool results, per-axon views, and recording manifests
example_17_analysis_layer.py
    structured analyses, missing-input requirements, and online Vm observers
example_18_solver_side_observers.py
    solver-side observers with trace-free Recording.none() results
```

Examples must:

- use the final API directly;
- include a clear didactic demo in `examples/advanced/` whenever a new
  advanced concept or non-trivial user workflow is introduced;
- teach one concept per demo when possible, with runnable, verbose,
  line-by-line code and comments focused on guiding the user through the
  workflow rather than hiding it behind helper scaffolding;
- include plots whenever they make the feature easier to understand: Vm traces,
  activation markers, peak-voltage markers, recruitment curves, velocity
  metrics, dispatch layouts, recording/retention comparisons, or
  observer-versus-recorded checks;
- never require world position on `AxonInstance`;
- construct footprints separately from stimuli;
- group each footprint and stimulus into an `ExtracellularDrive`;
- pass `ExtracellularStimulation` to the simulation;
- show that the solver performs the drive sum;
- use callable study updates;
- avoid obsolete simulation entry points;
- remain syntax-checked in CI.

---

# 38. Implementation roadmap

## Phase 0 — Guardrails and baselines

Implementation status: done.

- add architecture tests;
- establish CPU/GPU baselines;
- preserve scientific reference cases;
- remove obsolete benchmark formats.

## Phase 1 — Object model

Implementation status: done.

- rename current `AxonSimulation` to `AxonInstance`;
- remove world-position requirements;
- add `AxonPopulation`;
- add root `AxonSimulation`;
- rewrite examples and tests.

## Phase 2 — Typed and extracellular contracts

Implementation status: done. Runtime/device/precision planning values were
added in Phase 7; wiring them into execution remains future work.

Typed public API:

- add enums for closed scientific domains and policies;
- add `Signal[T]` and the built-in signal registry;
- add typed position selectors;
- add opaque identifiers;
- add structured devices and precision policies;
- rewrite public examples and tests without raw-string APIs.

Extracellular API:

- add `ExtracellularFootprint`;
- add `ExtracellularDrive`;
- add `ExtracellularStimulation`;
- add dense `ExtracellularPotential`;
- rewrite analytical contexts as footprint builders;
- delete core electrode/context coupling.

## Phase 2.5 — Hotpath evidence

Implementation status: done.

- add opt-in benchmark spans;
- catalog representative hotpath workloads;
- add a manual Colab GPU workflow;
- use CPU/GPU results to guide planning and backend-boundary work.

## Phase 3 — Planning and preparation

Implementation status: done for the current JAX backend path.

- make planning backend-neutral;
- add drive and footprint signatures;
- add footprint validation and resampling;
- add transformed-footprint preparation;
- create reusable prepared cohorts.

## Phase 4 — JAX isolation

Implementation status: done for the current backend boundary. Low-level kernels
can move later if that reduces real coupling.

- move JAX membrane runtime;
- move JAX solver runtime;
- move extracellular drive lowering;
- implement in-scan drive summation;
- delete old duplicate solver and dispatcher execution paths when empty.

## Phase 5 — Canonical results

Implementation status: done for pool results and result manifests. Scalar
`simulate(...)` still returns `SimResult`; decide before final docs whether to
unify scalar public output as `AxonSimulationResult`.

- add dense `CohortResult`;
- add signal descriptors;
- add recording manifests;
- add `AxonSimulationResult`;
- add `AxonResultView`;
- delete public eager `list[SimResult]` pool results.

## Phase 6 — Analyses

Implementation status: done for the current public layer. The public
`axs.analysis` namespace, definition objects, low-level post-hoc helpers,
structured input requirements, requirement/capability metadata, statuses,
population denominators, reports, online Vm observers, and the didactic example
exist. Phase 7.5 adds the first solver-side observer execution path for
activation and peak-voltage definitions.

- move activation and velocity into `analysis/`;
- add requirements and applicability;
- add statuses and provenance;
- add online observers;
- cross-validate online and post-hoc modes.

## Phase 7 — Performance

Implementation status: done for the current evidence layer.

Phase 7 adds public simulation memory estimates, typed runtime/device/precision
planning values, hotpath manifest memory metadata, and a footprint/stimulus-only
reuse workload. It does not pretend that dense extracellular time-space arrays
are gone: estimates explicitly surface current dense `Vstim[B,Nt,Nx]` memory
risk and compare it with factorized footprint/stimulus sizes. Phase 7.5 owns
the solver-side kernel changes that remove unnecessary trace/input retention.

- verify whether dense `Vext`/`Vstim` is currently materialized;
- keep hotpath traces as the CPU/GPU evidence loop;
- add footprint/stimulus-only sweep diagnostics;
- add memory estimates and warning thresholds;
- add typed runtime/device/precision planning values;
- integrate memory estimates with observability manifests.

## Phase 7.5 — Solver-side observers

Implementation status: done for the first public observer-only workflow.

Purpose: connect the public observer/analysis specifications to backend
execution so compact observer state is updated at every solver `dt` inside the
kernel or scan loop. The memory goal is to avoid retaining full
`Vm[time, position]` traces, and to avoid GPU-to-CPU transfer of those traces,
when the user only needs compact outputs such as activation, latency, peak
voltage, spike counts, or block summaries.

- done: lower public `axs.analysis.PeakVoltage` and `axs.analysis.Activation`
  specs into compact backend observer state;
- done: call observer updates inside scalar kernels and homogeneous single-cable
  batch kernels at each `dt`;
- done: keep single-cable batch observer state static-shaped and vectorized over
  batch rows;
- done: support observer-only execution with `Recording.none()` and trace-free
  `result.observations`;
- done: cross-validate solver-side peak voltage and activation against post-hoc
  traces in unit tests;
- done: add local hotpath/memory evidence showing no retained Vm output for an
  observer-only run;
- future optimization: move double-cable batch observer-only execution off
  scalar fallback once the compact state is wired and validated;
- future analysis design: decide whether latency/block-style analyses become
  direct solver observers or thin views over activation observer state.

## Phase 8 — Studies

Implementation status: not started.

- add callable sweeps;
- add threshold search;
- add recruitment;
- add reuse policies;
- add retention policies;
- add `AxonStudyResult`.

## Phase 9 — Serialization and reference backend

Implementation status: not started.

- serialize only final schemas;
- add NumPy reference backend;
- add cross-backend validation.

---

# 39. Suggested pull-request sequence

1. Add architecture and no-legacy tests.
2. Rename current `AxonSimulation` to `AxonInstance`.
3. Remove required world position from instances.
4. Add `AxonPopulation` and root `AxonSimulation`.
5. Add enums, typed selectors, signal descriptors, and opaque identifiers.
6. Rewrite public examples and tests without raw-string APIs.
7. Add `ExtracellularFootprint`.
8. Add `ExtracellularDrive`.
9. Add `ExtracellularStimulation`.
10. Rewrite point-source tools as footprint builders.
11. Add factorized preparation and in-solver summation.
12. Add footprint reuse and stimulus-only update tests.
13. Split planning from preparation.
14. Isolate JAX membrane and solver code.
15. Add canonical cohort-backed results.
16. Add analysis requirements and statuses.
17. Add online observers.
18. Add memory/performance estimates for recording and observer workloads.
19. Wire solver-side observers into scalar and batch kernels as per-`dt`
    compact reductions.
20. Add callable studies.
21. Add final serialization.
22. Delete every superseded module and format.

The development branch may break temporarily.

The merged target must not contain old and new architectures in parallel.
---

# 40. Non-goals

Do not:

- use raw strings as the preferred API for closed scientific domains;
- make built-in signals a closed enum that prevents extension;
- use one untyped string class for every identifier;
- make AxonScope own nerve geometry;
- make world position mandatory on `AxonInstance`;
- keep electrode geometry in the solver core;
- conflate footprint and stimulus;
- pre-sum all drives before execution;
- materialize `Vext[axon, time, position]` by default;
- create separate one-axon and population APIs;
- conflate myelination and cable formulation;
- merge analyses into raw numerical results;
- expose backend arrays as public contracts;
- preserve prototype compatibility;
- keep forwarding modules;
- introduce a generic Kernel IR before a real second backend;
- create empty packages without responsibilities.

---

# 41. Architecture acceptance criteria

## Typed public API

- closed domains use enums;
- extensible signals use typed descriptors;
- selectors use dedicated classes;
- identities use opaque identifier types;
- devices and precision use structured objects;
- raw strings remain limited to labels, metadata, and serialization boundaries;
- built-in values are discoverable through IDE autocomplete.

## Product boundary

- AxonScope owns intrinsic axon geometry only.
- `AxonInstance` has no required world position.
- external geometry packages provide numerical footprints.
- AxonScope does not depend on nerve geometry packages.

## Extracellular model

- `ExtracellularFootprint` contains spatial transfer only.
- `Stimulus` contains temporal waveform only.
- `ExtracellularDrive` contains exactly one footprint and one stimulus.
- `ExtracellularStimulation` aggregates several drives.
- the solver sums drive contributions during execution.
- factorized paths do not materialize full `Vext`.
- stimulus-only updates reuse prepared footprints.

## Simulation model

- one and many axons use the same lifecycle.
- one result model is used for all cardinalities.
- myelination and cable formulation remain separate metadata.
- shared and per-axon footprints are supported.

## Results and analyses

- results are batch-backed and cohort-dense.
- unsupported signals are explicit.
- analyses remain separate from numerical outputs.
- population summaries expose their denominator.
- online and post-hoc analyses are cross-validated.

## Migration

- obsolete modules are deleted.
- no compatibility aliases remain.
- examples use the final API directly.
- scientific reference tests pass.

---

# 42. Final target API

## 41.1 Build footprints

```python
cathode_footprint = external_package.compute_footprint(
    source="cathode",
    axons=population.ids,
    intrinsic_positions=population.compartment_positions,
)

anode_footprint = external_package.compute_footprint(
    source="anode",
    axons=population.ids,
    intrinsic_positions=population.compartment_positions,
)
```

Or use the lightweight helper:

```python
cathode_footprint = axs.analytical_fields.point_source_population(
    axons=population,
    axial_positions=axial_positions,
    radial_distances=radial_distances,
    conductivity=0.3 * axs.S_per_m,
)
```

## 41.2 Build drives

```python
CATHODE = axs.DriveId("cathode")

cathode = axs.ExtracellularDrive(
    id=CATHODE,
    footprint=cathode_footprint,
    stimulus=cathode_stimulus,
)

ANODE = axs.DriveId("anode")

anode = axs.ExtracellularDrive(
    id=ANODE,
    footprint=anode_footprint,
    stimulus=anode_stimulus,
)
```

## 41.3 Aggregate drives

```python
extracellular = axs.ExtracellularStimulation([
    cathode,
    anode,
])
```

## 41.4 Run simulation

```python
simulation = axs.AxonSimulation(
    axons=population,
    extracellular=extracellular,
    duration=20 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.MEMBRANE_VOLTAGE),
)

result = simulation.run()
```

## 41.5 Analyze

```python
activation = result.analyze(
    axs.analysis.Activation(...)
)
```

## 41.6 Sweep

```python
study = simulation.sweep(
    values=amplitudes,
    update=update_amplitude,
    reuse=axs.ReusePolicy.AUTO,
)
```

Internal path:

```text
normalize axons
→ validate drives and footprints
→ plan compatible cohorts
→ prepare footprints and cable operators
→ compile backend executable
→ sample stimuli
→ sum footprint × stimulus contributions inside the solver
→ assemble numerical result
→ run scientific analyses
```

The defining principles are:

> AxonScope knows the axon in intrinsic one-dimensional space, not its world
> position in a nerve.

> External geometry and field packages provide spatial extracellular
> footprints.

> One `ExtracellularDrive` combines one footprint with one stimulus.

> `ExtracellularStimulation` aggregates all drives passed to a simulation.

> The solver performs the sum during execution without materializing the full
> extracellular potential tensor.

> Because AxonScope is not deployed, obsolete prototype APIs should be deleted
> rather than preserved.
