# Public API Draft

Status on 2026-07-02: this is a proposal/roadmap document with a small
current-API snapshot near the top. It is not the canonical public API
reference. Treat the sections after **Implemented Today** as historical design
intent unless the surrounding text explicitly says they are implemented today.
For current runnable examples, prefer `README.md`, `examples/basic/`,
`examples/advanced/`, and the non-draft pages under `docs/`.

This document is a working proposal for the public AxonScope API before adding
new features. The goal is to make the package didactic and modular while keeping
the current solver/runtime internals free to evolve.

## Implemented Today

The currently runnable public surface is exposed from `axonscope.__init__` and
the examples. In short:

- `axs.axons`, `axs.membranes`, `axs.stimulation`, `axs.results`,
  `axs.analysis`, `axs.performance`, `axs.protocols`, `axs.dispatcher`,
  `axs.solvers`, `axs.signals`, `axs.positions`, and `axs.identifiers` are
  importable public namespaces.
- Public construction examples should use Pint quantities at boundaries:
  `length=500 * axs.um`, `diameter=0.5 * axs.um`,
  `duration=5 * axs.ms`, and `dt=0.01 * axs.ms`.
- `axs.AxonInstance`, `axs.AxonPopulation`, and `axs.AxonSimulation` are the
  current execution entry points. Older snippets below that use
  `axs.simulate(...)` or `axs.simulate_pool(...)` are historical draft material,
  not canonical API.
- `axs.Recording.full()`, `axs.Recording.center(...)`, `axs.Recording.none()`,
  `axs.signals`, one-axon result views, `recording_manifest`, and compact
  `observations` are current result/recording concepts.
- `axs.analysis.Activation`, `axs.analysis.Latency`, and
  `axs.analysis.ConductionBlock` can feed the strict VmRaster observer-only
  route. `axs.analysis.PeakVoltage` remains post-hoc on recorded Vm until a
  dedicated benchmarked solver-side design is accepted.
- `axs.SolverOptions` exposes current solver choices. Pseudo-double-cable
  options are standby research artifacts, not recommended public API.

Legacy roadmap snippets below may still use older unit-suffix constructor
names. Do not copy roadmap snippets into new examples without checking current
source and examples first.

## Design Intent

AxonScope should expose the construction path explicitly:

```text
membrane model -> axon cable model -> stimulation context -> simulation
                                                     \-> pool/batch simulation
```

The public API should make it natural to:

- build an axon from a membrane model or a membrane composition;
- choose a family whose cable formulation is explicit in its type;
- distinguish unmyelinated and myelinated axons at the modeling level;
- attach intracellular and extracellular stimulation contexts through an
  `AxonInstance` after axon construction;
- run either one axon or a pool/batch of related simulation conditions;
- keep backend internals inspectable for development without presenting
  low-level JAX runtimes and kernels as user-facing APIs.

## Namespaces

Recommended stable namespaces:

```text
axonscope.membranes
axonscope.axons
axonscope.stimulation
axonscope.results
axonscope.solvers
```

Advanced or semi-internal namespaces:

```text
axonscope.dispatcher
axonscope.runtime.jax.runtime
axonscope.runtime.jax.batch_kernels
axonscope.runtime.jax.kernels
axonscope.benchmarking
```

These may be useful while debugging internals, but they should not be presented
as the default API in tutorials.

## Membranes

Membrane models are the first public building block. Current built-in membranes
should be importable with biological names, not implementation names:

```python
import axonscope as axs

membrane = axs.membranes.HodgkinHuxley(celsius=6.3)
```

Proposed public membrane namespace:

```text
axs.membranes.Passive
axs.membranes.HodgkinHuxley
axs.membranes.RattayAberham
axs.membranes.Sundt
axs.membranes.Tigerholm
axs.membranes.Schild94
axs.membranes.Schild97
axs.membranes.AxNode
axs.membranes.Composite
axs.membranes.SectionLayout
```

There is no public advanced base class for membrane semantics. Public docs
should call these objects "membrane models"; the custom-model DSL produces
objects that satisfy the same membrane-model contract through the internal
compiler/runtime path.

`Composite` and `SectionLayout` should mean different things:

- `Composite` combines several membrane/channel mechanisms into one membrane
  model used by one compartment type or anatomical section.
- `SectionLayout` assigns one membrane model or composite membrane model to each
  cable section, for example `node`, `mysa`, `flut`, and `stin` in a myelinated
  axon.

## Axons

The public axon vocabulary should center on cable formulation and biological
family:

```text
axs.axons.Unmyelinated
axs.axons.Myelinated
```

Recommended decision: do not expose empty single/double-cable formulation
objects. The formulation is implicit in the biological family:
`Unmyelinated` is a single-cable monocompartment family and `Myelinated` is a
double-cable family. The structural base classes remain useful for typing and
internal organization:

```text
axs.axons.Axon + one Section
axs.axons.Axon + multi-section Layout
axs.axons.Axon(formulation=axs.axons.CableFormulation.SINGLE_CABLE)
axs.axons.Axon(formulation=axs.axons.CableFormulation.DOUBLE_CABLE)
```

Geometry should not be a major public namespace for now. Length, diameter,
compartment count, node count, and per-compartment arrays can be constructor
parameters or advanced options on axon classes. Dedicated geometry objects can
be introduced later if they remove real complexity.

Geometry still needs to anticipate two advanced cases:

- non-uniform fibers may contain several anatomical sections inside one
  numerical compartment, so section assignment cannot always be a simple
  one-section-per-compartment lookup;
- myelinated fibers can phase Ranvier node positions along the intrinsic
  one-dimensional layout axis, for example to align or de-align nodes across
  fibers in a pool.

Current public hook:

```python
axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5, x_shift=25.0 * axs.um)
stimulation = axs.analytical.point_source_stimulation(
    electrode,
    axon.layout.position_values(unit=axs.um) * axs.um,
    sigma=0.3 * axs.S_per_m,
    axon_y=20.0 * axs.um,
    axon_z=30.0 * axs.um,
)
sim = axs.AxonInstance(axon)
sim.add_extracellular_stimulation(stimulation=stimulation)
```

Do not store placement offsets on `AxonInstance`; they are external geometry
inputs used to build sampled footprints/stimulation.

### Granular Construction

The didactic construction path should look like this:

```python
membrane = axs.membranes.HodgkinHuxley(celsius=6.3)

axon = axs.axons.Unmyelinated(
    membrane=membrane,
    length_um=500.0,
    diameter_um=0.5,
    compartments=41,
)
```

For double-cable models:

```python
section_membranes = axs.membranes.SectionLayout(
    node=axs.membranes.AxNode(),
    mysa=axs.membranes.Passive(...),
    flut=axs.membranes.Passive(...),
    stin=axs.membranes.Passive(...),
)

template = axs.axons.MRGLikeDoubleCableTemplate(
    diameter=10.0 * axs.um,
    nodes=5,
)

axon = axs.axons.Myelinated(
    layout=template.layout(membranes=section_membranes),
)
```

The important public contract is that every named section of the cable receives
an explicit membrane model or composite membrane model. The implementation can
initially translate this into the current per-compartment heterogeneous layout.
The first implementation uses the existing MRG-like section geometry as the
double-cable geometry template; other myelinated geometries can be added later
without changing the section-membrane contract.

### Templates

Templates should be convenience constructors on the modeling families, not
separate architectural categories:

```python
axon = axs.axons.HodgkinHuxley(
    length_um=500.0,
    diameter_um=0.5,
    compartments=41,
    celsius=6.3,
)

axon = axs.axons.RattayAberham(...)
axon = axs.axons.Schild97(...)

axon = axs.axons.MRG(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
```

`MRG` is the concrete myelinated model. The reusable structure underneath it is
the MRG-like double-cable layout template.

Initial implementation:

```text
HodgkinHuxley -> current HodgkinHuxley axon
RattayAberham -> current RattayAberham axon
MRG -> Myelinated(...) with the default MRG membrane layout
```

## Stimulation

Stimuli and stimulation contexts should be attached to an `AxonInstance`
after axon construction. The public API should read as a protocol attached to a
descriptive axon.

```python
current = axs.stimulation.Stimulus.pulse(
    start=1.0,
    duration=0.5,
    amplitude=2.0,
)

sim = axs.AxonInstance(axon)
sim.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position=250.0 * axs.um,
        current=current,
    )
)
```

Public Python docs should prefer `AxonInstance.add_intracellular_context`.
`add_current_clamp(...)` remains a compact wrapper for scripts.

Extracellular:

```python
current = axs.stimulation.Stimulus.biphasic(
    start=0.5,
    cathodic_amplitude=80.0,
    cathodic_duration=0.05,
    interphase=0.02,
)

electrode = axs.analytical.PointSourceElectrode(
    x=250.0 * axs.um,
    y=0.0 * axs.um,
    z=500.0 * axs.um,
)
stimulation = axs.analytical.point_source_stimulation(
    electrode,
    axon.layout.position_values(unit=axs.um) * axs.um,
    sigma=0.3 * axs.S_per_m,
    stimulus=current,
)
sim.add_extracellular_stimulation(stimulation=stimulation)
```

`AxonInstance.add_extracellular_stimulation` is the canonical explicit
extracellular attachment method for sampled fields.

## Simulation

Single-axon simulation should have one simple entry point:

```python
run = axs.AxonSimulation(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
).run()
result = run.single
```

Low-level solver objects remain an internal/advanced validation route. The
canonical public execution path is `AxonSimulation(...).run()` with
`solver_options` for supported tuning.

Compatibility aliases can keep `tsim` and `dt` while public docs prefer
unit-explicit names.

## Recording Policy

Recording should be an explicit public concept separate from analysis observers.
Recording answers: what full traces do we store and return? Observers answer:
what compact quantities do we compute during the solve?

Default behavior:

- record `Vm` by default;
- do not record every current, conductance, gate, and auxiliary state variable
  unless the user asks for it;
- keep current full-observable support, but expose it through a clearer public
  recording policy.

`state_variables` means dynamic membrane variables that are not classical gates:
ion pools, pump states, buffer states, or any auxiliary variables declared by a
membrane model. This is the public-friendly name for the current internal
`MembraneStateSpec` mechanism.

Target API:

```python
run = axs.AxonSimulation(
    axon,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording(
        voltage=True,
        gates=True,
        currents=True,
        conductances=False,
        state_variables=False,
    ),
).run()
result = run.single
```

Convenience constructors:

```python
axs.Recording.voltage()
axs.Recording.full()
axs.Recording.none()
axs.Recording.only(axs.signals.Vm, axs.signals.GATES, axs.signals.CURRENTS)
```

Spatial/time filtering should be supported by the same policy:

```python
recording = axs.Recording(
    signals=[axs.signals.Vm, axs.signals.GATES, axs.signals.CURRENTS],
    positions=[0.0 * axs.um, 250.0 * axs.um, 500.0 * axs.um],
    sample_dt=0.1 * axs.ms,
)
```

`sample_dt` and `every_n_steps` should be mutually exclusive. `sample_dt`
is friendlier for users; `every_n_steps` is useful when exact solver-step
alignment matters.

Equivalent step-based form:

```python
recording = axs.Recording(
    signals=axs.signals.Vm,
    every_n_steps=10,
)
```

For batch and pool runs, this same concept should map onto the existing
batch recording machinery:

```python
results = axs.AxonSimulation(
    pool,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
)
```

Current implementation notes:

- `record_observables=True` already records gates, currents, conductances, and
  state variables in the single-axon solver path;
- `BatchOptions` and `BatchRecording` already cover part of spatial filtering
  for batch `Vm`;
- the public API should unify these into one user-facing `Recording` object
  before expanding the feature further.

Initial public-wrapper scope:

- single-axon `Recording` supports `Vm` plus observable groups and filters the
  returned `recordings` groups;
- if any observable group is requested, the current solver still computes all
  observable groups internally, so this is an output contract first, not yet a
  memory-minimizing storage contract;
- pool `Recording` supports `Vm` spatial policies (`full`, `center`,
  `probes`) through existing batch machinery;
- position-based recording, temporal subsampling, observer-only simulation, and
  pool gates/currents are intentionally explicit future work.

## Pool And Batch Simulation

Pools should cover two use cases:

1. biological groups, such as fibers in a nerve or fascicle;
2. experimental batches, such as testing different stimulation amplitudes,
   electrode positions, diameters, or membrane models.

Proposed public shape:

```python
sim_a = axs.AxonInstance(axon_a)
sim_a.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position=250.0 * axs.um,
        current=stimulus_a,
    )
)
sim_a.add_extracellular_stimulation(
    stimulation=axs.analytical.point_source_stimulation(
        electrode_a,
        axon_a.layout.position_values(unit=axs.um) * axs.um,
        sigma=0.3 * axs.S_per_m,
        stimulus=extra_a,
    )
)

sim_b = axs.AxonInstance(axon_b)
sim_b.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position=250.0 * axs.um,
        current=stimulus_b,
    )
)
sim_b.add_extracellular_stimulation(
    stimulation=axs.analytical.point_source_stimulation(
        electrode_b,
        axon_b.layout.position_values(unit=axs.um) * axs.um,
        sigma=0.3 * axs.S_per_m,
        stimulus=extra_b,
    )
)

results = axs.AxonSimulation(
    [sim_a, sim_b],
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
).run()
```

`AxonSimulation.run()` should return an `AxonSimulationResult`, with dense
cohorts internally and one `AxonResultView` per axon in input order. Dispatch
metadata can live in each view's `diagnostics`, while the selected `Recording`
policy lives on the pool result and views.

For now, each axon should carry its own intracellular and extracellular
contexts through its axon object. Pool-level drive helpers should stay out
of the public API unless a future batching layer genuinely needs them.

The current `run_pool` can remain an advanced implementation piece. It may
expose private dispatch result tuples for debugging, but public docs should
present `AxonSimulation([axon_a, axon_b], ...).run()` first.

```text
AxonSimulation([axons], ...).run() -> public pool entry point
```

## Visualization And Analysis

The current public plotting surface is deliberately small. One-axon result
views provide the voltage trace/map helpers directly. Analysis-derived plots
such as spike rasters live under `axs.analysis.views`.

```text
axs.results.views
axs.analysis.views
axs.analysis
```

### Result Visualization

```python
run = axs.AxonSimulation(sim, duration=5.0 * axs.ms, dt=0.01 * axs.ms).run()
result = run.single

result.plot_trace(position=250.0 * axs.um)
result.plot_map()
axs.analysis.views.plot_spike_raster(result, threshold_mV=-10.0)
```

Population results expose one-axon views through indexing, iteration, and
`.single` for one-row runs. Plot population summaries explicitly from those
views until a dedicated population plotting API exists.

### Geometry Visualization

Geometry plotting is currently owned by the model/layout objects, for example
`axon.layout.plot(...)`. It is not part of result or analysis views.

### Analysis Functions

Post-processing functions should be available both after a simulation and,
eventually, inside the solver loop.

Initial post-hoc API:

```python
spikes = axs.analysis.rasterize(result, threshold_mV=-10.0)
velocity = axs.analysis.conduction_velocity(result)
peaks = axs.analysis.peak_voltage(result)
```

Long-term solver-side API:

```python
run = axs.AxonSimulation(
    axon,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[
        axs.analysis.Activation(threshold=-10.0 * axs.mV),
    ],
).run()
result = run.single
```

The solver-side mechanism should be called `observers` rather than
visualization. The current supported observer-only runtime lowers
threshold-style membrane-voltage definitions to packed VmRaster output, avoiding
full `Vm[Nt, Nx]` materialization when users only need activation-style events.

Design constraints for solver-side observers:

- observers must be JAX-compatible when used inside JIT-compiled solvers;
- observers should have a clear distinction between streaming state and final
  output;
- post-hoc analysis functions and solver-side observers should share naming and
  semantics where possible;
- `PeakVoltage` remains post-hoc on recorded Vm until a dedicated benchmarked
  solver-side implementation exists;
- full trace recording remains the default teaching mode, while observer-only
  simulation becomes the large-scale/pool mode.

## Units

Current convention:

- public axon geometry and simulation time names should be physical nouns with
  Pint quantities: `length`, `diameter`, `duration`, `dt`;
- stimulation and point-source coordinates use quantity-oriented names such as
  `position`, `x`, `y`, `z`, and `min_distance`; canonical internal fields can
  still use suffixes such as `position_um` or `x_um`;
- Pint quantities are supported at public boundaries;
- quantities are normalized once during construction/runtime preparation;
- keep solvers operating on plain numeric arrays with canonical internal units.

Current target usage:

```python
import axonscope as axs

axon = axs.axons.HodgkinHuxley(
    length=500 * axs.um,
    diameter=0.5 * axs.um,
    compartments=41,
)

run = axs.AxonSimulation(sim, duration=5 * axs.ms, dt=0.01 * axs.ms).run()
result = run.single
```

## Migration Strategy

Phase 1: public wrappers and aliases.

- Add `axonscope.membranes` as the public membrane-description surface.
- Add `axonscope.membranes.SectionLayout` as a friendly wrapper over current
  heterogeneous membrane layout internals.
- Add `axonscope.axons.Unmyelinated` and `axonscope.axons.Myelinated` as
  instantiable modeling families, with template constructors over current axon
  classes.
- Add `AxonSimulation` as the public execution root.
- Add a public `Recording` object that wraps current `record_observables` and
  batch recording options.
- Prefer clean quantity-oriented public names such as `diameter`, `duration`,
  and `AxonInstance.add_intracellular_context`.

Phase 2: examples.

- Rewrite basic examples against public API.
- Keep advanced examples for runtime and batch kernels.
- Add smoke tests that execute small examples.
- Add at least one result visualization example and one geometry visualization
  example.

Phase 3: docs.

- Shrink README to installation and two quickstarts.
- Move concepts, tutorials, validation, and benchmarking to `docs/`.

Phase 4: cleanup policy.

- Keep tests and examples on the public API.
- Remove compatibility shims when they duplicate the public vocabulary.

## Open Questions

- Should `SectionLayout` gain validation helpers for required sections such as
  `node`, `mysa`, `flut`, and `stin`, or should constructors own that check?
- How should `SectionLayout` represent several anatomical sections inside one
  numerical compartment for non-uniform fibers?
- Should Ranvier node shifting live on the axon (`axon.shift_nodes`) or on a
  reusable placement/geometry helper?
- Should pool result objects expose only per-axon traces, or also grouped
  batch diagnostics by default?
- What should the public grammar be for selecting recorded variables: boolean
  groups, string paths such as `gates.m`, or both?
- Should solver-side post-processing be called `observers`, `processors`, or
  something closer to signal-processing terminology?
