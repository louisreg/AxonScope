# Axon Model Organization

AxonFleet models an axon as a descriptive object first. Solver arrays are
derived only when a solver asks for them.

```text
axs.membranes.Model -> Section -> Layout -> Axon -> AxonInstance/AxonPopulation -> AxonSimulation -> solver runtime
```

The important split is:

- `Section` describes a local membrane/material prototype.
- `Layout` places sections in one-dimensional space and assigns compartment
  counts.
- The solver bridge privately materializes per-compartment arrays from the
  layout. Those arrays are not part of the modeling API.

## Core Idea

A mono-section axon is one section placed once with several compartments. The
statement "N copies of the same section" is numerically equivalent to one
section with N compartments when all local properties are identical. Repeating
the section can still be useful when users want explicit section boundaries for
inspection, templates, or later heterogeneity.

A multi-section axon is an ordered sequence of placed sections. Each placed
section has:

- one `Section` prototype;
- one physical `length`;
- one `compartments` count.

This keeps the public model close to the biology: node, MYSA, FLUT, STIN, or
plain axon pieces are just sections placed by a layout.

## Units

New section/layout interfaces use short physical names and require units:

- `Section(diameter=..., Ra=..., Cm=...)`
- `Layout.single_uniform(..., length=...)`
- `Layout.single_non_uniform(..., x=...)`
- `Layout.sequence(..., section_lengths=..., lengths=..., phase_shift=...)`
- `PeriaxonalLayer(radial_conductance=..., radial_capacitance=..., axial_resistance=...)`

Internally, AxonFleet stores explicit canonical floats such as `diameter_um`,
`length_um`, `Ra_ohm_cm`, and `Cm_uF_cm2`. The unit-bearing public boundary is
there to prevent ambiguous calls before publication.

```python
import axonfleet as axs

section = axs.axons.Section(
    "axon",
    membrane=axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
    diameter=0.5 * axs.um,
    Ra=100.0 * axs.ohm_cm,
    Cm=1.0 * axs.uF_per_cm2,
)

layout = axs.axons.Layout.single_uniform(
    section,
    length=1.0 * axs.mm,
    compartments=101,
)
```

## Public Objects

`axs.membranes.Model`

A runtime-independent public membrane model from `axonfleet.membranes`. It can
be a built-in membrane class, a user-authored class-based membrane model, or a
composite membrane. It does not know about solvers, batches, JAX, or time
stepping. The internal `MembraneModel` descriptor is compiler/runtime plumbing,
not a user-facing modeling concept.

`Section`

A conceptual local piece of cable. It owns the membrane model, diameter, axial
resistivity, membrane capacitance density, optional periaxonal layer, and tags.
It does not own length or discretization.

`PeriaxonalLayer`

Local double-cable material properties around one section. This is part of the
axon description, not an extracellular stimulation context.

`LayoutElement`

One placed section: `section + length + compartments`.

`Layout`

An ordered sequence of `LayoutElement` objects, plus helper constructors for
common cases:

```python
layout = axs.axons.Layout.single_uniform(section, length=1.0 * axs.mm, compartments=101)

layout = axs.axons.Layout.single_non_uniform(
    section,
    x=axs.units.Q_([0.0, 20.0, 60.0, 140.0], "micrometer"),
)

layout = axs.axons.Layout.sequence(
    [node, internode],
    section_lengths=axs.units.Q_([1.0, 199.0], "micrometer"),
    compartments=[1, 8],
    lengths=1.0 * axs.mm,
)

phased = axs.axons.Layout.sequence(
    [node, internode],
    section_lengths=axs.units.Q_([1.0, 199.0], "micrometer"),
    compartments=[1, 8],
    lengths=1.0 * axs.mm,
    phase_shift=80.0 * axs.um,
)

layout.plot(position_unit=axs.um, compartment_labels="auto")
```

Runtime materialization

The runtime privately derives canonical per-compartment arrays such as
positions, lengths, diameters, membrane models, section names, and section
indices. Public inspection stays on `Layout.position_values(...)`,
`diameter_values(...)`, `compartment_length_values(...)`, and `plot(...)`.

`Axon`

The descriptive model object. It owns the layout, cable formulation, initial
state, and descriptive aggregate properties such as `length` and
`diameter`. Per-compartment diameter values are available through
`axon.diameter_values(unit=...)` when a custom layout is heterogeneous. Solver
arrays are still derived at the solver boundary. `Axon` does not own
stimulation protocols or solver runtime arrays.

`AxonInstance`

The simulation protocol wrapped around one descriptive `Axon`. It owns
intracellular clamps and sampled extracellular stimulation in the axon's
intrinsic coordinate system. Anatomical placement remains outside AxonFleet.

`AxonPopulation`

The ordered cohort object. It stores `AxonInstance` rows, preserves input
order, and can contain one row when a workflow should still use population
execution.

`AxonSimulation`

The executable root object. It binds one `Axon`/`AxonInstance` or an
`AxonPopulation` to duration, time step, recording policy, and execution
options. Its current `.run()` implementation delegates to the public population
execution path, including one-row `B=1` batch runs.

`SolverAxon`

The NumPy-only runtime-neutral representation built from an `Axon` or
`AxonInstance` in `axonfleet.runtime.solver_axon`. It combines the flattened
layout with formulation and simulation-level periaxonal overrides.

## Module Responsibilities

`src/axonfleet/axons/section.py`

Defines local descriptive objects:

- `Section`
- `PeriaxonalLayer`

`src/axonfleet/axons/layout.py`

Defines the user-facing spatial layout:

- `LayoutElement`
- `Layout`

`src/axonfleet/axons/flattened.py`

Defines the private derived per-compartment geometry consumed by runtime
materialization.

`src/axonfleet/axons/axon.py`

Defines:

- `Axon`

`src/axonfleet/axon_instance.py`

Defines:

- `AxonInstance`

`src/axonfleet/population.py`

Defines:

- `AxonPopulation`

`src/axonfleet/simulation.py`

Defines:

- `AxonSimulation`

`src/axonfleet/runtime/solver_axon.py`

Defines:

- `SolverAxon`
- `build_solver_axon`

This is the runtime-neutral bridge from descriptive axons to numerical arrays.

`src/axonfleet/axons/unmyelinated.py`

Defines ready-made unmyelinated templates such as `HodgkinHuxley`,
`RattayAberham`, `Sundt`, `Tigerholm`, `Schild94`, and `Schild97`.

`src/axonfleet/axons/myelinated.py`

Defines myelinated templates such as `MRG`, `GainesMotor`, and
`GainesSensory`.

`src/axonfleet/axons/templates/mrg_like_double_cable.py`

Defines the reusable MRG-like double-cable layout template. It describes the
node/MYSA/FLUT/STIN geometry and periaxonal structure. Use
`Layout.sequence(..., phase_shift=...)` for generic repeated motifs. Use
`x_shift` on `MRG(...)` or `MRGLikeDoubleCableTemplate(...)` when a pool needs
intrinsic MRG node phase shifts, such as NRV fractional `node_shift`
conversion. Use
`Layout.with_x_shift(...)` only for simple local translation of an already
defined layout. Do not store anatomical placement on axon instances.

## Examples

### Explicit Mono-Section Axon

```python
import axonfleet as axs

section = axs.axons.Section(
    "axon",
    membrane=axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
    diameter=0.5 * axs.um,
    Ra=100.0 * axs.ohm_cm,
    Cm=1.0 * axs.uF_per_cm2,
)

axon = axs.axons.Axon(
    layout=axs.axons.Layout.single_uniform(section, length=1.0 * axs.mm, compartments=101),
    formulation=axs.axons.CableFormulation.SINGLE_CABLE,
    Vinit=-67.5 * axs.mV,
    Temp=6.3 * axs.degC,
)
```

### Non-Uniform Mono-Section Axon

```python
layout = axs.axons.Layout.single_non_uniform(
    section,
    x=axs.units.Q_([0.0, 15.0, 40.0, 120.0, 300.0], "micrometer"),
)
```

This is the only public constructor for custom non-uniform compartment centers
in one section.

### Different Sections In A Motif

```python
layout = axs.axons.Layout.sequence(
    [node, mysa, flut, stin],
    section_lengths=axs.units.Q_([1.0, 3.0, 20.0, 80.0], "micrometer"),
    compartments=[1, 1, 2, 4],
    lengths=312.0 * axs.um,
)
```

`section_lengths` describes one motif. `lengths` is the total requested layout
length; it must be an integer multiple of `sum(section_lengths)`.

## Boundary Rules

`axonfleet.axons` and `axonfleet.membranes` should stay descriptive:

- no solver kernels;
- no backend construction;
- no rate-table compilation;
- no time-stepping policy;
- no stimulation protocol state.

Membrane/cable runtime compilation belongs in
`axonfleet.runtime.jax.preparation.base`.
Stimulation runtime compilation belongs in
`axonfleet.runtime.jax.inputs`.
Pool and batch row preparation belongs in `axonfleet.preparation`.
