# Public API Draft

This document is a working proposal for the public AxonScope API before adding
new features. The goal is to make the package didactic and modular while keeping
the current solver/runtime internals free to evolve.

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
  `AxonSimulation` after axon construction;
- run either one axon or a pool/batch of related simulation conditions;
- keep low-level JAX runtimes and kernels available for advanced users without
  making them the first user experience.

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
axonscope.channel_models
axonscope.icm
axonscope.dispatcher
axonscope.solvers.runtime
axonscope.solvers.batch_kernels
axonscope.solvers.kernels
axonscope.benchmarking
```

These can remain importable, but they should not be presented as the default
API in tutorials.

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

The current `IonChannelModelBase` can remain the advanced base class. Public
docs should call these objects "membrane models". Later, the custom-model DSL
can produce objects that satisfy the same membrane-model contract.

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
axs.axons.Axon(formulation="single-cable")
axs.axons.Axon(formulation="double-cable")
```

Geometry should not be a major public namespace for now. Length, diameter,
compartment count, node count, and per-compartment arrays can be constructor
parameters or advanced options on axon classes. Dedicated geometry objects can
be introduced later if they remove real complexity.

Geometry still needs to anticipate two advanced cases:

- non-uniform fibers may contain several anatomical sections inside one
  numerical compartment, so section assignment cannot always be a simple
  one-section-per-compartment lookup;
- myelinated fibers need an explicit way to shift Ranvier node positions in
  space, for example to phase-align or de-align nodes across fibers in a
  pool.

Candidate public hooks:

```python
axon.shift_nodes(delta_um=25.0)
sim = axs.AxonSimulation(axon, x_offset_um=0.0, y_um=20.0, z_um=30.0)
```

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

Stimuli and stimulation contexts should be attached to an `AxonSimulation`
after axon construction. The public API should read as a protocol attached to a
descriptive axon.

```python
current = axs.stimulation.Stimulus.pulse(
    start=1.0,
    duration=0.5,
    amplitude=2.0,
)

sim = axs.AxonSimulation(axon)
sim.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position_um=250.0,
        current=current,
    )
)
```

Public Python docs should prefer `AxonSimulation.add_intracellular_context`.
`add_current_clamp(...)` remains a compact wrapper for scripts.

Extracellular:

```python
current = axs.stimulation.Stimulus.biphasic(
    start=0.5,
    cathodic_amplitude=80.0,
    cathodic_duration=0.05,
    interphase=0.02,
)

electrode = axs.stimulation.PointSourceElectrode(
    x_um=250.0,
    y_um=0.0,
    z_um=500.0,
)

sim.add_extracellular_context(
    context=axs.stimulation.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(current)],
        sigma=0.3,
    ),
)
```

`AxonSimulation.add_extracellular_context` stays as the canonical explicit
extracellular attachment method.

## Simulation

Single-axon simulation should have one simple entry point:

```python
result = axs.simulate(
    sim,
    duration_ms=5.0,
    dt_ms=0.01,
)
```

Advanced users can still choose a solver:

```python
solver = axs.solvers.CrankNicholson()
result = solver.solve(sim, duration_ms=5.0, dt_ms=0.01)
```

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
result = axs.simulate(
    axon,
    duration_ms=5.0,
    dt_ms=0.01,
    recording=axs.Recording(
        voltage=True,
        gates=True,
        currents=True,
        conductances=False,
        state_variables=False,
    ),
)
```

Convenience constructors:

```python
axs.Recording.voltage()
axs.Recording.full()
axs.Recording.none()
axs.Recording.only("Vm", "gates.m", "currents.I_Na")
```

Spatial/time filtering should be supported by the same policy:

```python
recording = axs.Recording(
    variables=["Vm", "gates", "currents"],
    positions_um=[0.0, 250.0, 500.0],
    sample_dt_ms=0.1,
)
```

`sample_dt_ms` and `every_n_steps` should be mutually exclusive. `sample_dt_ms`
is friendlier for users; `every_n_steps` is useful when exact solver-step
alignment matters.

Equivalent step-based form:

```python
recording = axs.Recording(
    variables=["Vm"],
    every_n_steps=10,
)
```

For batch and pool runs, this same concept should map onto the existing
batch recording machinery:

```python
result = axs.simulate_pool(
    pool,
    duration_ms=5.0,
    dt_ms=0.01,
    recording=axs.Recording.center(["Vm"]),
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
sim_a = axs.AxonSimulation(axon_a)
sim_a.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position_um=250.0,
        current=stimulus_a,
    )
)
sim_a.add_extracellular_context(
    context=axs.stimulation.AnalyticalExtracellularContext(
        electrodes=[electrode_a.with_stimulus(extra_a)],
        sigma=0.3,
    )
)

sim_b = axs.AxonSimulation(axon_b)
sim_b.add_intracellular_context(
    context=axs.stimulation.IntracellularCurrentClamp(
        position_um=250.0,
        current=stimulus_b,
    )
)
sim_b.add_extracellular_context(
    context=axs.stimulation.AnalyticalExtracellularContext(
        electrodes=[electrode_b.with_stimulus(extra_b)],
        sigma=0.3,
    )
)

result = axs.simulate_pool(
    [sim_a, sim_b],
    duration_ms=5.0,
    dt_ms=0.01,
)
```

`simulate_pool` should return a plain `list[SimResult]`, one result per
axon in input order. Dispatch metadata can live in `result.diagnostics`, while
the selected `Recording` policy lives directly on
each `SimResult`.

For now, each axon should carry its own intracellular and extracellular
contexts through its axon object. Pool-level drive helpers should stay out
of the public API unless a future batching layer genuinely needs them.

The current `run_pool` can remain an advanced implementation piece. It may
expose private dispatch result tuples for debugging, but public docs should
present `simulate_pool([axon_a, axon_b])` first.

```text
simulate_pool([axons]) -> public pool entry point
```

## Visualization And Analysis

Visualization should become a first-class public layer, but it should stay
separate from solver internals. Two use cases matter:

1. plotting and inspecting simulation results;
2. plotting and inspecting axon/fiber geometry before simulation.

Proposed public namespaces:

```text
axs.results.visualization
axs.results.analysis
```

### Result Visualization

Result plotting should work from `SimResult` and pool results without
requiring users to know the internal array layout.

Target API:

```python
result = axs.simulate(sim, duration_ms=5.0, dt_ms=0.01)

axs.results.visualization.plot_voltage_trace(
    result,
    position_um=250.0,
)

axs.results.visualization.plot_voltage_map(result)
axs.results.visualization.plot_raster(result, threshold_mV=-10.0)
```

Pool result visualization:

```python
pool_result = axs.simulate_pool(pool, duration_ms=5.0, dt_ms=0.01)

axs.results.visualization.plot_pool_peaks(pool_result)
axs.results.visualization.plot_pool_raster(pool_result)
```

`SimResult` should stay a data container. Plotting and post-processing should
live under `axs.results.visualization` and `axs.results.analysis` so analysis code can explicitly
check whether `Vm` is full, center-only, probe-only, or otherwise filtered.

### Geometry Visualization

Geometry visualization should let users inspect the model before running an
expensive simulation.

Target API:

```python
axs.results.visualization.plot_axon_geometry(axon)
axs.results.visualization.plot_membrane_sections(axon)
axs.results.visualization.plot_pool_geometry(pool)
```

For myelinated axons, the visualization should show section labels such as
`node`, `mysa`, `flut`, and `stin`, and optionally color by membrane model.

For extracellular stimulation, it should eventually be possible to overlay
electrodes and fiber positions:

```python
axs.results.visualization.plot_stimulation_geometry(pool)
```

### Analysis Functions

Post-processing functions should be available both after a simulation and,
eventually, inside the solver loop.

Initial post-hoc API:

```python
spikes = axs.results.analysis.rasterize(result, threshold_mV=-10.0)
velocity = axs.results.analysis.conduction_velocity(result)
peaks = axs.results.analysis.peak_voltage(result)
```

Long-term solver-side API:

```python
result = axs.simulate(
    axon,
    duration_ms=5.0,
    dt_ms=0.01,
    observers=[
        axs.results.analysis.RasterObserver(threshold_mV=-10.0),
        axs.results.analysis.PeakVoltageObserver(),
    ],
)
```

The solver-side mechanism should be called `observers` or `processors` rather
than visualization. These functions should consume solver state during the time
loop and return compact derived outputs, avoiding full `Vm[Nt, Nx]`
materialization when users only need spikes, peaks, thresholds, or summary
statistics.

Design constraints for solver-side observers:

- observers must be JAX-compatible when used inside JIT-compiled solvers;
- observers should have a clear distinction between streaming state and final
  output;
- post-hoc analysis functions and solver-side observers should share naming and
  semantics where possible;
- full trace recording remains the default teaching mode, while observer-only
  simulation becomes the large-scale/pool mode.

## Units

Current convention:

- public geometry/time names should be explicit: `length_um`, `diameter_um`,
  `duration_ms`, `dt_ms`, `position_um`;
- Pint quantities are supported at public boundaries;
- quantities are normalized once during construction/runtime preparation;
- keep solvers operating on plain numeric arrays with canonical internal units.

Current target usage:

```python
import axonscope as axs

axon = axs.axons.HodgkinHuxley(
    length_um=500 * axs.um,
    diameter_um=0.5 * axs.um,
    compartments=41,
)

result = axs.simulate(sim, duration_ms=5 * axs.ms, dt_ms=0.01 * axs.ms)
```

## Migration Strategy

Phase 1: public wrappers and aliases.

- Add `axonscope.membranes` as a friendly wrapper over `channel_models`.
- Add `axonscope.membranes.SectionLayout` as a friendly wrapper over current
  heterogeneous membrane layout internals.
- Add `axonscope.axons.Unmyelinated` and `axonscope.axons.Myelinated` as
  instantiable modeling families, with template constructors over current axon
  classes.
- Add `simulate` and `simulate_pool`.
- Add a public `Recording` object that wraps current `record_observables` and
  batch recording options.
- Prefer unit-explicit public names such as `diameter_um`, `duration_ms`, and
  `AxonSimulation.add_intracellular_context`.

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
