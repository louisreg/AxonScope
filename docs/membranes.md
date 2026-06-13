# Membranes

`axonscope.membranes` is the public namespace for runtime-independent membrane
descriptions.

Membrane objects are descriptive. They do not own JAX functions, solver
backends, time stepping, or compiled state. Solvers translate them to the
current hand-written channel implementations only when a simulation runtime is
prepared.

```text
axs.membranes.MembraneModel -> axon Section/Layout -> solver runtime -> channel_models
```

## Public Surface

```python
import axonscope as axs

membrane = axs.membranes.HodgkinHuxley(celsius=6.3)
```

Built-in templates:

- `Passive`
- `HodgkinHuxley`
- `RattayAberham`
- `Sundt`
- `Tigerholm`
- `Schild94`
- `Schild97`
- `AxNode`
- `Composite`
- `SectionLayout`

The package layout mirrors these responsibilities:

```text
src/axonscope/membranes/
  model.py           MembraneModel, Composite, ensure_membrane_model
  builtins.py        built-in descriptive membrane templates
  section_layout.py  named section-to-membrane assignment
  __init__.py        public facade
```

## Units

Plain numbers are interpreted in AxonScope's canonical membrane units. Pint
quantities are accepted and converted at construction time. After construction,
`MembraneModel.params` stores plain floats.

Canonical units:

| Parameter family | Canonical unit |
| --- | --- |
| membrane voltage, reversal potential | `millivolt` |
| temperature | `degree_Celsius` |
| diameter | `micrometer` |
| conductance density | `siemens / centimeter ** 2` |
| passive membrane resistance `Rm` | `ohm * centimeter ** 2` |
| Tigerholm concentrations | `millimolar` |
| Tigerholm pump current density | `milliampere / centimeter ** 2` |

The explicit suffix parameters keep their suffix unit, for example
`ena_mV`, `diameter_um`, and `gnabar_S_cm2`.

```python
import axonscope as axs

hh = axs.membranes.HodgkinHuxley(
    gnabar=120.0 * axs.mS / axs.cm**2,
    ena=50.0 * axs.mV,
    celsius=6.3 * axs.degC,
)

passive = axs.membranes.Passive(
    Rm=10_000.0 * axs.ohm * axs.cm**2,
    EL=-70.0 * axs.mV,
)
```

## Composite Membranes

`Composite` combines several membrane mechanisms into one membrane assigned to
one section.

```python
membrane = axs.membranes.Composite(
    [
        axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
        axs.membranes.Passive(Rm=10_000.0 * axs.ohm_cm2, EL=-70.0 * axs.mV),
    ]
)

section = axs.axons.Section(
    "axon",
    membrane=membrane,
    diameter=0.5 * axs.um,
    Ra=100.0 * axs.ohm_cm,
    Cm=1.0 * axs.uF_per_cm2,
)

layout = axs.axons.Layout.single_uniform(section, length=1000.0 * axs.um, compartments=101)
```

Conceptually, this is still one membrane description for one section. Spatial
length and compartment count live in `Layout`, and the solver may compile the
membrane to a composite channel model internally.

## Section Layouts

`SectionLayout` assigns membrane models to named anatomical section kinds. It
is useful for templates such as MRG-like myelinated fibers, where a cable
contains node, MYSA, FLUT, and STIN sections.

```python
section_membranes = axs.membranes.SectionLayout(
    node=axs.membranes.AxNode(),
    mysa=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
    flut=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
    stin=axs.membranes.Passive(Rm=1e6 * axs.ohm_cm2, EL=-80.0 * axs.mV),
)

template = axs.axons.MRGLikeDoubleCableTemplate(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
axon = axs.axons.Myelinated(layout=template.layout(membranes=section_membranes))
```

`SectionLayout` is descriptive. It does not create solver-side heterogeneous
compute layouts. That responsibility stays in `axonscope.icm`.

## Boundary With `channel_models`

`axonscope.channel_models` contains solver-side implementations such as
`HodgkinHuxleyICM`, `PassiveICM`, and composite channel classes. Public axon
construction should prefer `axonscope.membranes`.

The narrow bridge is `compile_membrane_model` in `axonscope.solvers.runtime`.
This keeps the public model layer ready for a future DSL without coupling it to
today's compute classes.
