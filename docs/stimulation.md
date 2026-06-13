# Stimulation

`axonscope.stimulation` contains backend-independent stimulation descriptions:
temporal waveforms, electrodes, and physical contexts.

The main idea is to keep each object responsible for one physical layer:

```text
Stimulus -> Electrode/Clamp -> Context -> AxonSimulation protocol -> solver runtime
```

## Public Surface

```python
import axonscope as axs
from axonscope.stimulation import (
    Stimulus,
    IntracellularContext,
    IntracellularCurrentClamp,
    ExtracellularContext,
    AnalyticalExtracellularContext,
    NRVExtracellularContext,
    Electrode,
    PointSourceElectrode,
)
```

Package layout:

```text
src/axonscope/stimulation/
  stimuli.py      temporal waveforms
  electrodes.py   spatial extracellular electrode descriptions
  contexts.py     intracellular and extracellular physical contexts
  __init__.py     public facade
```

## Stimulus

`Stimulus` is a temporal waveform. It stores time in milliseconds and supports
sample-and-hold or linear interpolation.

```python
stimulus = Stimulus.pulse(
    start=0.2,
    duration=0.1,
    amplitude=0.8,
)
```

Time arguments accept canonical numbers in milliseconds or Pint quantities:

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
model. Positions and stimulation live on `AxonSimulation`.

```python
axon = axs.axons.HodgkinHuxley(
    length_um=1.0 * axs.mm,
    diameter_um=0.5 * axs.um,
    compartments=101,
    celsius=6.3 * axs.degC,
)
sim = axs.AxonSimulation(axon)
```

Solvers accept `sim` directly. Passing a pure `Axon` is still allowed for a
no-stimulation simulation.

## Intracellular Stimulation

The explicit public path is to build an intracellular context, then attach it
to an `AxonSimulation`:

```python
current = axs.stimulation.Stimulus.pulse(
    start=0.2 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=0.8 * axs.nA,
)

clamp = axs.stimulation.IntracellularCurrentClamp(
    position_um=500.0 * axs.um,
    current=current,
)
sim.add_intracellular_context(context=clamp)
```

Here, `amplitude=0.8` means `0.8 nA` because the waveform is used in an
intracellular current clamp.

For compact scripts, `add_current_clamp(...)` is a convenience wrapper around
the same context object:

```python
sim.add_current_clamp(position_um=500.0 * axs.um, current=current)
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
    x_um=500.0 * axs.um,
    y_um=0.0 * axs.um,
    z_um=100.0 * axs.um,
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

`PointSourceElectrode` uses public coordinates in micrometers. Plain numbers
are interpreted as `um`; Pint quantities are converted.

```python
electrode = axs.stimulation.PointSourceElectrode(
    x_um=0.5 * axs.mm,
    z_um=100.0 * axs.um,
)
```

Use `AnalyticalExtracellularContext.footprint_per_current(...)` for user-facing
footprints with explicit voltage/current/position units. The lower-level
`footprint_for_electrode` method remains solver-facing and expects positions in
meters.

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
