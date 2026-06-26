# AxonScope

AxonScope is a pre-release Python framework for peripheral nerve axon
simulations. It focuses on explicit physical units, descriptive axon/stimulation
objects, validated solver behavior, and batch-ready execution paths for axon
pools.

The API is still being stabilized before publication. Prefer the current clean
public names shown here and in `examples/`; old temporary aliases should not be
treated as supported compatibility.

## Capabilities

- Unmyelinated and myelinated one-dimensional axon descriptions.
- Hodgkin-Huxley, Rattay-Aberham, Sundt, Tigerholm, Schild94/Schild97, and MRG
  style templates.
- Intracellular current clamps and sampled extracellular stimulation, including
  analytical point-source quick-start helpers.
- Executable `AxonSimulation` root object plus public `simulate(...)` and
  `simulate_pool(...)` wrappers.
- Structured post-hoc analysis definitions with per-axon statuses, population
  denominators, and small low-level spike/raster helpers.
- Automatic pool dispatch with scalar fallback, strict batches, parameter
  batches, and padded double-cable batches.
- Fast unit tests and optional NRV validation tests.

## Installation

This repository uses a `src/` layout and Python 3.11+.

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[examples]"
python -m pip install -e ".[dev]"
python -m pip install -e ".[benchmark]"
python -m pip install -e ".[dev,nrv]"
```

The `nrv` extra installs Python-side helper dependencies only. A working local
NRV/NEURON setup is still required for `tests/nrv`.

## Public API Shape

AxonScope separates descriptive models from simulation protocols and numerical
execution:

```text
membranes/axons/stimulation -> AxonInstance/AxonPopulation -> AxonSimulation/run -> results
```

Use short physical names with Pint quantities at public boundaries:

```python
import axonscope as axs

axon = axs.axons.HodgkinHuxley(
    length=500.0 * axs.um,
    diameter=0.5 * axs.um,
    compartments=41,
    celsius=6.3 * axs.degC,
)
```

Internal implementation fields may use canonical suffixes such as
`length_um`, `diameter_um`, or `dt_ms`; the preferred public style is
quantity-oriented: `length`/`diameter` for axons and `duration`/`dt` for
simulation wrappers.

Primary namespaces:

```text
axs.axons          descriptive axon geometry and templates
axs.membranes      runtime-independent membrane descriptions
axs.stimulation    stimuli, electrodes, clamps, and contexts
axs.results        AxonSimulationResult, AxonResultView, recording manifests, visualization
axs.analysis       structured analysis definitions, statuses, reports
axs.performance    simulation memory estimates and runtime/device policies
axs.inspection     printable planning, dispatch, lowering, kernel, and result reports
axs.protocols      threshold, sweep, and recruitment workflows
axs.dispatcher     pool dispatch inspection and advanced execution helpers
axs.solvers        stable solver facade, solver options, and batch execution knobs
axs.signals        typed, extensible recording signal descriptors
axs.positions      typed position selectors for analyses and criteria
axs.identifiers    opaque identifiers such as AxonId, DriveId, and SignalId
```

Optional external adapters live outside the root facade. For example,
`axonscope.integrations.nrv` converts NRV-owned fascicle geometry, LIFE/FEM
footprints, and NRV recruitment results into AxonScope fiber rows, sampled
footprints, stimulation objects, and comparison summaries.

## Quick Start

```python
import axonscope as axs

axon = axs.axons.HodgkinHuxley(
    length=500.0 * axs.um,
    diameter=0.5 * axs.um,
    compartments=41,
    celsius=6.3 * axs.degC,
)

sim = axs.AxonInstance(axon)
sim.add_current_clamp(
    position=250.0 * axs.um,
    current=axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.5 * axs.ms,
        amplitude=2.0 * axs.nA,
    ),
)

run = axs.simulate(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
)
result = run.single

center = result.nearest_position_index(250.0 * axs.um)
print(result.t.shape, result.Vm[:, center].shape)
```

For workflows that should carry their execution settings as one object, use
the root `AxonSimulation`:

```python
simulation = axs.AxonSimulation(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.full(),
)
run = simulation.run()
result = run.single
```

Runtime/device/precision requests use the typed public execution policy:

```python
policy = axs.ExecutionPolicy(
    runtime=axs.Runtime.JAX,
    device=axs.Device.cpu(),
    precision=axs.PrecisionPolicy.float32(),
)
run = axs.simulate(sim, duration=5.0 * axs.ms, dt=0.01 * axs.ms, execution_policy=policy)
result = run.single
```

Pipeline planning can be printed before launching solver kernels:

```python
simulation.inspect(print_summary=True)
```

## Extracellular Stimulation

```python
axon = axs.axons.MRG(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
center_x = axon.layout.position_values(unit=axs.um)[axon.n_compartments // 2] * axs.um

electrode = axs.analytical.PointSourceElectrode(
    x=center_x,
    z=500.0 * axs.um,
)
stimulus = axs.Stimulus.biphasic(
    start=0.5 * axs.ms,
    cathodic_amplitude=80.0 * axs.uA,
    cathodic_duration=0.05 * axs.ms,
    interphase=0.02 * axs.ms,
)
extracellular = axs.analytical.point_source_stimulation(
    electrode,
    axon.layout.position_values(unit=axs.um) * axs.um,
    sigma=0.3 * axs.S_per_m,
    stimulus=stimulus,
)

sim = axs.AxonInstance(axon)
sim.add_extracellular_stimulation(stimulation=extracellular)

run = axs.simulate(sim, duration=2.0 * axs.ms, dt=0.01 * axs.ms)
result = run.single
```

MRG layouts can phase the repeated node motif along their local
one-dimensional axis, for example when importing NRV node-shifted fiber tables:

```python
axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5, x_shift=80.0 * axs.um)
```

`x_shift` sets the intrinsic distance from the axon start to the first node
start. It changes local compartment/node positions only; it is not an
`AxonInstance` world coordinate.

Point-source geometry is a quick-start helper under `axs.analytical`. It is
sampled into an `ExtracellularFootprint`, paired with a stimulus as an
`ExtracellularDrive`, and attached as `ExtracellularStimulation`.

See `docs/stimulation.md` for the stimulation model and
`docs/axon_model_organization.md` for axon geometry/layout details.

## Pools And Recording

`AxonPopulation` is the explicit public container for cohorts. It stores
`AxonInstance` rows, preserves input order, and can contain one row when a
workflow should still use population execution. `simulate_pool(...)` and the
root `AxonSimulation` accept either an `AxonPopulation` or a sequence of `Axon`
or `AxonInstance` objects. Pool runs return `AxonSimulationResult`, whose
indexed rows are lightweight `AxonResultView` objects in population order.

```python
population = axs.AxonPopulation([sim_a, sim_b], name="demo pool")

results = axs.simulate_pool(
    population,
    duration=1.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
    progress=True,
)

for result in results:
    print(result.diagnostics["dispatch_method"], result.record_indices)

center_vm = results.signal(axs.signals.Vm)
first_row = results.axon(0)
vm_manifest = results.recording_manifest.signal(axs.signals.Vm)
```

Inspect grouping before a run with:

```python
plan = axs.dispatcher.build_dispatch_plan([sim_a, sim_b])
axs.dispatcher.print_dispatch_plan(plan)
axs.dispatcher.plot_dispatch_plan(plan)
```

`Recording` is the public storage policy. Current single-axon runs support Vm
and selected observable groups; pool runs currently support Vm retention modes
such as full, center, probes, and explicit indices.
Signals are descriptors rather than a closed enum, so custom signal descriptors
can be added later without changing the recording API.

```python
full = axs.Recording.full()
center = axs.Recording.center(axs.signals.Vm)
probes = axs.Recording.probes(axs.signals.Vm, count=8)
indices = axs.Recording.indices([0, 10, 20], axs.signals.Vm)
```

See `docs/pool_dispatch.md` and `docs/results_recording_analysis.md` for the
full contracts.

## Analysis And Protocols

Structured post-hoc analysis definitions live under `axs.analysis` and consume
one-axon result views or whole `AxonSimulationResult` objects:

```python
report = result.report(
    axs.analysis.Activation(threshold=-20.0 * axs.mV, target=axs.positions.DISTAL),
    axs.analysis.PeakVoltage(target=axs.positions.CENTER),
)

activation = report["activation"]
print(activation.values, activation.statuses, activation.population.n_valid)
```

Low-level helpers live in the same `axs.analysis` namespace:

```python
velocity_m_s = axs.analysis.conduction_velocity(result)
spike_t_ms, spike_x_um = axs.analysis.rasterize(result, threshold_mV=-10.0)
```

Activation and peak-voltage definitions can also create lightweight online Vm
observers for chunked traces:

```python
observer = axs.analysis.Activation(threshold=-20.0 * axs.mV).online_observer(
    positions=result.position_values(unit=axs.um) * axs.um,
)
observer.update(result.time_values(unit=axs.ms) * axs.ms, result.Vm * axs.mV)
activation = observer.finalize()
```

For compact solver-side threshold reductions, pass compatible analysis
definitions as simulation observers. `Recording.none()` keeps the result
trace-free and returns packed VmRaster observations:

```python
simulation = axs.AxonSimulation(
    population,
    duration=1.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[
        axs.analysis.Activation(
            threshold=-20.0 * axs.mV,
            target=axs.positions.CENTER,
        ),
    ],
)
results = simulation.run()
raster = results.observations[axs.VM_RASTER_OBSERVATION_KEY]
```

Repeated stimulation workflows live in `axs.protocols`:

```python
threshold = axs.protocols.find_activation_threshold(
    simulation_factory=lambda current: make_simulation(current),
    bounds=(1.0 * axs.uA, 500.0 * axs.uA),
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.analysis.ActivationCriterion(threshold=-20.0 * axs.mV),
)
```

The equivalent executable-root form is:

```python
simulation = axs.AxonSimulation(
    population,
    duration=1.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
)
results = simulation.run()
```

## Examples

Examples are the executable learning path. Keep them in sync with API changes.

```bash
python examples/basic/01_first_intracellular_simulation.py
python examples/basic/02_stimuli_and_units.py
python examples/basic/03_point_source_footprint.py
python examples/basic/04_extracellular_mrg_simulation.py
python examples/basic/05_population_pool_run.py
python examples/basic/06_activation_velocity.py
python examples/basic/07_threshold_vs_diameter.py
python examples/basic/08_recruitment_curve_population.py
```

Advanced workflow examples:

```bash
python examples/advanced/object_model/01_axon_simulation_root.py
python examples/advanced/recording_analysis/01_recording_options.py
python examples/advanced/recording_analysis/05_vmraster_observer_only.py
python examples/advanced/protocols/01_threshold_vs_parameters.py
python examples/advanced/protocols/02_recruitment_waveforms.py
python examples/advanced/runtime/01_runtime_policy.py
python examples/advanced/runtime/03_pipeline_inspection.py
python examples/with_nrv/01_realistic_fascicle_geometry_comparison.py
```

See `examples/README.md` for the full learning path.

## Tests And Validation

Fast unit suite:

```bash
MPLBACKEND=Agg python -m pytest -q tests/unit --tb=short
```

Optional NRV validation suite:

```bash
MPLBACKEND=Agg python -m pytest -q tests/nrv --tb=short
```

NRV validation requires a local NRV-ready environment. Record only fresh, dated
validation results. See `docs/validation.md`.

## Benchmarks

Benchmarks are for performance measurement, not correctness validation. Use
`tests/nrv` for scientific validation first.

```bash
python benchmark/runtime/run.py --list
python benchmark/runtime/run.py --suite smoke
python benchmark/runtime/environment_info_demo.py
python benchmark/hotpaths/run.py --list
python benchmark/hotpaths/run.py --workload all --preset smoke
python benchmark/nrv_performance/run.py --list
```

GPU validation on Kaggle uses the same population timing runner in AxonScope-only
mode, with a synthetic NRV-shaped population so the Kaggle worker does not need
NRV installed:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark population_tsim_gpu \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60
```

Use `simulation.estimate()` before large runs to inspect retained Vm, dense
`Vstim`, factorized footprint, and stimulus-sample memory.

Generated benchmark results live under ignored `benchmark/results/` and
`benchmark/reports/` paths. See `benchmark/runtime/`, `benchmark/hotpaths/`,
and `benchmark/nrv_performance/` for detailed runners.

For NRV/FEM/LIFE recruitment performance on Kaggle GPU, use
`benchmark/kaggle/run_kernel.py --benchmark realistic_fascicle_nrv_gpu_full`.
The reproducible full preset uses one synthetic NRV nerve with four circular
fascicles, 100 axons per fascicle, 21 sequential amplitudes, and one NRV
validation amplitude. The June 25, 2026 P100 run is summarized in
`docs/benchmarks/nrv_fascicle_full_kaggle_2026_06_25.md`.

## Documentation

- `docs/axon_model_organization.md`: descriptive axon layer.
- `docs/membranes.md`: membrane model descriptions.
- `docs/stimulation.md`: stimuli, electrodes, clamps, and contexts.
- `docs/pool_dispatch.md`: pool dispatch and batching behavior.
- `docs/results_recording_analysis.md`: `Recording`, canonical results, analysis, and
  visualization.
- `docs/solver_organization.md`: solver package boundaries and time grids.
- `docs/validation.md`: fast checks and local NRV validation policy.
- `docs/benchmarks/`: curated benchmark summaries with retained conclusions.
- `docs/api_public_draft.md`: proposal-only API notes, not a runnable reference.
- `docs/recorders_observers_activation_strategy.md`: observer/recruitment
  roadmap with current implementation status.

## Changelog

See `CHANGELOG.md`.
