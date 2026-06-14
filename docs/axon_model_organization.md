# Axon Model Organization

AxonScope models an axon as a descriptive object first. Solver arrays are
derived only when a solver asks for them.

```text
MembraneModel -> Section -> Layout -> Axon -> AxonInstance/AxonPopulation -> AxonSimulation -> solver runtime
```

The important split is:

- `Section` describes a local membrane/material prototype.
- `Layout` places sections in one-dimensional space and assigns compartment
  counts.
- `FlattenedLayout` is the derived per-compartment representation used by the
  solver bridge. It is not the modeling API.

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
- `Layout.sequence(..., section_lengths=..., lengths=...)`
- `PeriaxonalLayer(radial_conductance=..., radial_capacitance=..., axial_resistance=...)`

Internally, AxonScope stores explicit canonical floats such as `diameter_um`,
`length_um`, `Ra_ohm_cm`, and `Cm_uF_cm2`. The unit-bearing public boundary is
there to prevent ambiguous calls before publication.

```python
import axonscope as axs

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

`MembraneModel`

A runtime-independent membrane description from `axonscope.membranes`. It can
be a built-in membrane, a composite membrane, or later a DSL-generated membrane.
It does not know about solvers, batches, JAX, or time stepping.

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

layout.plot(position_unit=axs.um, compartment_labels="auto")
```

`FlattenedLayout`

The canonical per-compartment arrays derived by
`axs.axons.flatten_layout(layout)`. It contains arrays such as `x_um`,
`lengths_um`, `diam_um`, membrane models, section names, and section indices.
This object lives in `axonscope.axons.flattened` so the descriptive layout stays
separate from solver-facing arrays.

`Axon`

The descriptive model object. It owns the layout, cable formulation, initial
state, and descriptive aggregate properties such as `length` and
`diameter`. Per-compartment diameter values are available through
`axon.diameter_values(unit=...)` when a custom layout is heterogeneous. Solver
arrays are still derived at the solver boundary. `Axon` does not own
stimulation protocols or solver runtime arrays.

`AxonInstance`

The simulation protocol wrapped around one descriptive `Axon`. It owns global
position, intracellular clamps, and extracellular stimulation contexts.

`AxonPopulation`

The ordered cohort object. It stores `AxonInstance` rows, preserves input
order, and can contain one row when a workflow should still use population
execution.

`AxonSimulation`

The executable root object. It binds one `Axon`/`AxonInstance` or an
`AxonPopulation` to duration, time step, recording policy, and execution
options. Its current `.run()` implementation delegates to the public scalar and
pool execution paths.

`SolverAxon`

The NumPy-only solver-side representation built from an `Axon` or
`AxonInstance` in `axonscope.solvers.axon_runtime`. It combines the flattened
layout with formulation and simulation-level periaxonal overrides.

## Module Responsibilities

`src/axonscope/axons/section.py`

Defines local descriptive objects:

- `Section`
- `PeriaxonalLayer`

`src/axonscope/axons/layout.py`

Defines the user-facing spatial layout:

- `LayoutElement`
- `Layout`

`src/axonscope/axons/flattened.py`

Defines derived per-compartment geometry:

- `FlattenedLayout`
- `flatten_layout`

`src/axonscope/axons/axon.py`

Defines:

- `Axon`

`src/axonscope/axon_instance.py`

Defines:

- `AxonInstance`

`src/axonscope/population.py`

Defines:

- `AxonPopulation`

`src/axonscope/simulation.py`

Defines:

- `AxonSimulation`
- `simulate`
- `simulate_pool`

`src/axonscope/solvers/axon_runtime.py`

Defines:

- `SolverAxon`
- `build_solver_axon`

This is the solver-side bridge from descriptive axons to numerical arrays.

`src/axonscope/axons/unmyelinated.py`

Defines ready-made unmyelinated templates such as `HodgkinHuxley`,
`RattayAberham`, `Sundt`, `Tigerholm`, `Schild94`, and `Schild97`.

`src/axonscope/axons/myelinated.py`

Defines myelinated templates such as `MRG`.

`src/axonscope/axons/templates/mrg_like_double_cable.py`

Defines the reusable MRG-like double-cable layout template. It describes the
node/MYSA/FLUT/STIN geometry and periaxonal structure.

## Examples

### Explicit Mono-Section Axon

```python
import axonscope as axs

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

`axonscope.axons` and `axonscope.membranes` should stay descriptive:

- no solver kernels;
- no backend construction;
- no rate-table compilation;
- no time-stepping policy;
- no stimulation protocol state.

Membrane/cable runtime compilation belongs in `axonscope.solvers.runtime`.
Stimulation runtime compilation belongs in `axonscope.stimulation.runtime`.
Pool and batch array assembly belongs in `axonscope.dispatcher`.
