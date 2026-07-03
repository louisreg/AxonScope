# Stimulation

`axonscope.stimulation` contains backend-independent stimulation descriptions:
temporal waveforms, sampled extracellular footprints/drives, and physical
intracellular contexts.

The main idea is to keep each object responsible for one physical layer:

```text
Stimulus -> Clamp or Footprint/Drive -> AxonInstance/AxonPopulation -> AxonSimulation/run -> solver runtime
```

## Public Surface

```python
import axonscope as axs
from axonscope.stimulation import (
    Stimulus,
    IntracellularContext,
    IntracellularCurrentClamp,
    ExtracellularFootprint,
    ExtracellularDrive,
    ExtracellularStimulation,
    ExtracellularPotential,
)

drive_id = axs.DriveId("center contact")
```

Package layout:

```text
src/axonscope/stimulation/
  stimuli.py       temporal waveforms
  extracellular.py static footprints, drives, and dense inspection objects
  contexts.py      intracellular contexts
  __init__.py      public facade
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
- extracellular drives normalize waveform amplitudes to `ampere`.

Plain amplitude numbers are interpreted in the consuming object's canonical
unit.

## Simulation Protocol

`Axon` objects are descriptive: intrinsic geometry, layout, formulation, and
membrane model. Simulation-local stimulation lives on `AxonInstance`; world
placement does not.

```python
axon = axs.axons.HodgkinHuxley(
    length=1.0 * axs.mm,
    diameter=0.5 * axs.um,
    compartments=101,
    celsius=6.3 * axs.degC,
)
sim = axs.AxonInstance(axon)
```

`AxonSimulation` accepts `sim` directly. It carries duration, time step,
recording, and run options together for one or more instances:

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

The high-level extracellular path is sampled and typed:

```text
ExtracellularFootprint -> ExtracellularDrive -> ExtracellularStimulation
```

Helpers that know about analytical geometry or external anatomical placement
must convert that information before attachment. `AxonInstance` receives only
the sampled stimulation.

```python
axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
positions = axon.layout.position_values(unit=axs.um) * axs.um

electrode = axs.analytical.PointSourceElectrode(
    x=axon.node_position("center", unit=axs.um),
    z=100.0 * axs.um,
)

stimulus = axs.Stimulus.pulse(
    start=0.2 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=-50.0 * axs.uA,
)

extracellular = axs.analytical.point_source_stimulation(
    electrode,
    positions,
    sigma=0.3 * axs.S_per_m,
    stimulus=stimulus,
)

sim = axs.AxonInstance(axon)
sim.add_extracellular_stimulation(stimulation=extracellular)
```

`PointSourceElectrode` lives in `axs.analytical` because it is a quick-start
helper, not a solver/runtime concept. It can also be used one layer lower:

```python
footprint = axs.analytical.point_source_footprint(
    electrode,
    positions,
    sigma=0.3 * axs.S_per_m,
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
`ExtracellularStimulation` sums drives for inspection and attachment without
duplicating geometry. Use `extracellular.potential(...)` only when a dense
`ExtracellularPotential` object is useful for plotting or diagnostics.
The same public objects own their generic inspection plots:

```python
footprint.plot(position_unit=axs.um, voltage_unit=axs.mV, current_unit=axs.uA)
extracellular.plot_footprints(
    position_unit=axs.um,
    voltage_unit=axs.mV,
    current_unit=axs.uA,
)
extracellular.plot_potential(t, time_unit=axs.ms, voltage_unit=axs.mV)
extracellular.potential(t, voltage_unit=axs.mV).plot(time_unit=axs.ms)
```

For multi-source stimulation, build several drives on the same intrinsic
position support:

```python
extracellular = axs.ExtracellularStimulation([cathode_drive, anode_drive])
sim.add_extracellular_stimulation(stimulation=extracellular)
```

During threshold or recruitment protocols, keep the footprint fixed and replace
only the drive stimulus:

```python
drive = sim.extracellular_stimulation.drives[0]
updated = sim.extracellular_stimulation.replace_drive(
    drive.id,
    stimulus=new_stimulus,
)
sim.add_extracellular_stimulation(stimulation=updated, replace=True)
```

NRV/FEM/LIFE workflows should use the adapter helpers in
`axonscope.integrations.nrv` through two bridges. NRV builds the external
geometry, fiber population, electrodes, and FEM field; AxonScope receives an
intrinsic axon population first, then sampled footprints for that population.

```python
from axonscope.integrations import nrv as axs_nrv

# `nerve` is an already-built NRV object with its extracellular stimulation.
axons = axs_nrv.population_from_nrv(
    nerve,
    nerve_length_um=10_000.0,
    include_unmyelinated=True,
)
footprints = axs_nrv.footprints_from_nrv(nerve, axons)
pool = footprints.stimulated_population(
    electrode_index=0,
    stimulus=axs.Stimulus.pulse(
        start=0.1 * axs.ms,
        duration=0.1 * axs.ms,
        amplitude=-60.0 * axs.uA,
    ),
    drive_id_prefix="nrv_life",
)
```

There is no public `NRVExtracellularContext` placeholder. Direct NumPy/SciPy/FEM
lowering is not a current public API; the supported path is adapter -> sampled
`ExtracellularFootprint` -> `ExtracellularDrive` ->
`ExtracellularStimulation`.

## Electrode Footprints

Analytical point-source helpers use quantity-oriented coordinates. Pass lengths
with units; they are normalized to internal micrometers during construction.

```python
electrode = axs.analytical.PointSourceElectrode(
    x=0.5 * axs.mm,
    z=100.0 * axs.um,
)
```

Point-source coordinates are inputs to analytical footprint construction, not
placement stored on an `AxonInstance`. When an external workflow owns axon
placement, pass offsets to `axs.analytical.point_source_footprint(...)`,
`point_source_drive(...)`, or `point_source_stimulation(...)`. Solver execution
then sees only intrinsic axon positions and sampled footprints.

## Runtime Boundary

Stimulation objects stay descriptive. During execution, the selected backend
lowers them into numerical callables or tensors:

- `Stimulus` becomes a temporal waveform representation;
- `IntracellularContext` objects become current-density injection inputs;
- `ExtracellularStimulation` objects become imposed-potential inputs, including
  dense or factorized footprint tensors when a batch route can use them.

Public examples and docs should treat that lowering as backend-owned. This keeps
the public stimulation package independent from runtime choices such as
single-axon solving, pool batches, JAX compilation, or precomputed footprint
tensors.
