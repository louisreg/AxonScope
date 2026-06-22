# Stimulation

`axonscope.stimulation` contains backend-independent stimulation descriptions:
temporal waveforms, electrodes, and physical contexts.

The main idea is to keep each object responsible for one physical layer:

```text
Stimulus -> Electrode/Clamp -> Context -> AxonInstance/AxonPopulation -> AxonSimulation/run -> solver runtime
```

## Public Surface

```python
import axonscope as axs
from axonscope.stimulation import (
    Stimulus,
    IntracellularContext,
    IntracellularCurrentClamp,
    ExtracellularContext,
    ExtracellularFootprint,
    ExtracellularDrive,
    ExtracellularStimulation,
    ExtracellularPotential,
    AnalyticalExtracellularContext,
    NRVExtracellularContext,
    Electrode,
    PointSourceElectrode,
)

drive_id = axs.DriveId("center contact")
```

Package layout:

```text
src/axonscope/stimulation/
  stimuli.py      temporal waveforms
  electrodes.py   spatial extracellular electrode descriptions
  extracellular.py static footprints, drives, and dense inspection objects
  contexts.py     intracellular and extracellular physical contexts
  __init__.py     public facade
```

## Stimulus

`Stimulus` is a temporal waveform. It stores time in milliseconds and supports
sample-and-hold or linear interpolation.

```python
stimulus = Stimulus.pulse(
    start=0.2 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=0.8 * axs.nA,
)
```

Public constructor time arguments must carry units and are stored internally in
milliseconds. The only implicit time default is an omitted `start`, which means
0 ms:

```python
import axonscope as axs

stimulus = Stimulus.biphasic(
    start=200.0 * axs.us,
    cathodic_duration=100.0 * axs.us,
    cathodic_amplitude=-500.0 * axs.uA,
)
```

Amplitude units are preserved by the waveform and interpreted by the physical
object that uses it:

- intracellular contexts normalize waveform amplitudes to `nanoampere`;
- extracellular electrodes normalize waveform amplitudes to `ampere`.

Plain amplitude numbers are interpreted in the consuming object's canonical
unit.

## Simulation Protocol

`Axon` objects are descriptive: geometry, layout, formulation, and membrane
model. Positions and stimulation live on `AxonInstance`.

```python
axon = axs.axons.HodgkinHuxley(
    length=1.0 * axs.mm,
    diameter=0.5 * axs.um,
    compartments=101,
    celsius=6.3 * axs.degC,
)
sim = axs.AxonInstance(axon)
```

Solvers and `axs.simulate(...)` accept `sim` directly. For workflows that
should carry duration, time step, recording, and run options together, wrap one
or more instances in the executable root object:

```python
simulation = axs.AxonSimulation(
    sim,
    duration=1.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.voltage(),
)
run = simulation.run()
result = run.single
```

Passing a pure `Axon` is still allowed for a no-stimulation simulation.

## Intracellular Stimulation

The explicit public path is to build an intracellular context, then attach it
to an `AxonInstance`:

```python
current = axs.stimulation.Stimulus.pulse(
    start=0.2 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=0.8 * axs.nA,
)

clamp = axs.stimulation.IntracellularCurrentClamp(
    position=500.0 * axs.um,
    current=current,
)
sim.add_intracellular_context(context=clamp)
```

If a waveform amplitude is passed as a plain number, it is interpreted by the
consuming object: `0.8` means `0.8 nA` in an intracellular current clamp.

For compact scripts, `add_current_clamp(...)` is a convenience wrapper around
the same context object:

```python
sim.add_current_clamp(position=500.0 * axs.um, current=current)
```

`IntracellularContext` is the base contract. The current concrete subclass is
`IntracellularCurrentClamp`, and the runtime compiler currently supports this
point-injection context. The stored `clamp.current` is a `Stimulus` normalized
to nanoamperes.

## Extracellular Stimulation

Extracellular stimulation separates the electrode geometry from the temporal
stimulus.

```python
stimulus = axs.stimulation.Stimulus.pulse(
    start=0.2 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=-50 * axs.uA,
)

electrode = axs.stimulation.PointSourceElectrode(
    x=500.0 * axs.um,
    y=0.0 * axs.um,
    z=100.0 * axs.um,
)

context = axs.stimulation.AnalyticalExtracellularContext(
    electrodes=[electrode.with_stimulus(stimulus)],
    sigma=0.3 * axs.S_per_m,
)
sim.add_extracellular_context(context=context)
```

Here, `amplitude=-50 * axs.uA` is normalized to amperes internally because the
waveform is used as an electrode current.

For explicit construction:

```python
context = axs.stimulation.AnalyticalExtracellularContext(
    electrodes=[electrode.with_stimulus(stimulus)],
    sigma=0.3 * axs.S_per_m,
)
assert context.electrodes[0].stimulus.y_unit == "ampere"
```

`with_stimulus` returns a copy with the same geometry and a new stimulus, which
makes it safe to reuse an electrode geometry across several simulations.

For multi-electrode stimulation, use one context containing all electrodes:

```python
context = axs.stimulation.AnalyticalExtracellularContext(
    electrodes=[
        electrode_a.with_stimulus(stimulus_a),
        electrode_b.with_stimulus(stimulus_b),
    ],
    sigma=0.3 * axs.S_per_m,
)
sim.add_extracellular_context(context=context)
```

For exploratory plots, analytical contexts evaluate and plot the summed field:

```python
values = context.evaluate(x_positions, t, voltage_unit=axs.mV)
context.plot_footprint(x_positions, voltage_unit=axs.mV, current_unit=axs.uA)
context.plot_evaluation(x_positions, t, voltage_unit=axs.mV)
context.plot_activation_function(x_positions, voltage_unit=axs.mV, current_unit=axs.uA)
```

For reusable extracellular stimulation, build static footprints first, then
pair each footprint with one temporal stimulus:

```python
footprint = context.build_footprint(
    electrode,
    x_positions,
    source_id="center contact",
)

drive = axs.ExtracellularDrive(
    id=axs.DriveId("center contact"),
    footprint=footprint,
    stimulus=stimulus,
)

extracellular = axs.ExtracellularStimulation([drive])
vext_mV = extracellular.evaluate(t, voltage_unit=axs.mV)
```

`ExtracellularFootprint` contains only spatial transfer samples in V/A.
`ExtracellularDrive` contains exactly one footprint and one stimulus.
`ExtracellularStimulation` sums drives for inspection without eagerly
duplicating electrode geometry. Use `extracellular.potential(...)` only when a
dense `ExtracellularPotential` object is useful for plotting or diagnostics.

The solver-facing extracellular contract is `footprint_for_electrode(...)`.
Runtime code can compile any `ExtracellularContext` subclass that implements
that method, including future FEM-backed contexts.

`NRVExtracellularContext` reserves that future FEM path:

```python
nrv_context = axs.stimulation.NRVExtracellularContext(
    electrodes=[electrode.with_stimulus(stimulus)],
    medium="endoneurium_bhadra",
    fem_model=None,
    metadata={"source": "future NRV FEM"},
)
```

It validates and carries the configuration now, but raises
`NotImplementedError` if a solver asks it for a footprint before NRV/FEM
evaluation is implemented.

## Electrode Footprints

`PointSourceElectrode` uses quantity-oriented public coordinates. Pass lengths
with units; they are normalized to internal micrometers during construction.

```python
electrode = axs.stimulation.PointSourceElectrode(
    x=0.5 * axs.mm,
    z=100.0 * axs.um,
)
```

Use `AnalyticalExtracellularContext.footprint_per_current(...)` for user-facing
footprints with explicit voltage/current/position units. The lower-level
`footprint_for_electrode` method remains solver-facing and expects positions in
meters.

Use `AnalyticalExtracellularContext.build_footprint(...)` when the next step is
an `ExtracellularDrive` or `ExtracellularStimulation`.
`PointSourceElectrode.build_footprint(...)` is also available for simple
one-electrode analytical scripts when passing `sigma` directly is clearer.

In pool simulations, point-source electrode coordinates are global coordinates.
`sim.set_position(...)` places each axon simulation in that same global frame;
AxonScope converts the point-source transverse offsets internally before
evaluating the one-dimensional footprint.

## Solver Boundary

Stimulation objects stay descriptive. Runtime compilation lives in
`axonscope.stimulation.runtime`:

- `Stimulus` becomes a JAX-ready temporal callable;
- `IntracellularContext` objects become a current-density injection callable
  through `compile_intracellular_contexts(...)`;
- `ExtracellularContext` objects become a summed imposed-potential callable
  through `compile_extracellular_contexts(...)`.

This keeps stimulation construction independent from runtime choices such as
single axon solving, pool batches, JAX compilation, or precomputed
footprint tensors.
