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
- Intracellular current clamps and analytical point-source extracellular
  stimulation.
- Executable `AxonSimulation` root object plus public `simulate(...)` and
  `simulate_pool(...)` wrappers.
- Post-hoc analysis helpers for rasterization, conduction velocity, peak
  voltage, and activation criteria.
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
membranes/axons/stimulation -> AxonInstance/AxonPopulation -> AxonSimulation/run -> SimResult
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
axs.results        SimResult, post-hoc analysis, visualization, activation
axs.protocols      threshold, sweep, and recruitment workflows
axs.dispatcher     pool dispatch inspection and advanced execution helpers
axs.solvers        solver classes, options, kernels, and runtime builders
axs.signals        typed recording signal selectors
axs.positions      typed position selectors for analyses and criteria
axs.identifiers    opaque identifiers such as AxonId and DriveId
```

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

result = axs.simulate(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
)

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
result = simulation.run()
```

Direct solver calls remain available for advanced users and use solver-level
time names:

```python
result = axs.solvers.CrankNicholson().solve(
    sim,
    tsim=5.0 * axs.ms,
    dt=0.01 * axs.ms,
)
```

## Extracellular Stimulation

```python
axon = axs.axons.MRG(
    diameter=10.0 * axs.um,
    nodes=5,
    compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4},
)
center_x = axon.layout.position_values(unit=axs.um)[axon.n_compartments // 2] * axs.um

electrode = axs.PointSourceElectrode(
    x=center_x,
    z=500.0 * axs.um,
)
stimulus = axs.Stimulus.biphasic(
    start=0.5 * axs.ms,
    cathodic_amplitude=80.0 * axs.uA,
    cathodic_duration=0.05 * axs.ms,
    interphase=0.02 * axs.ms,
)

sim = axs.AxonInstance(axon)
sim.add_extracellular_context(
    context=axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )
)

result = axs.simulate(sim, duration=2.0 * axs.ms, dt=0.01 * axs.ms)
```

For workflows that need reusable spatial transfer functions, analytical
contexts can build static `ExtracellularFootprint` objects, which are paired
with stimuli as `ExtracellularDrive` objects and grouped in
`ExtracellularStimulation`.

See `docs/stimulation.md` for the stimulation model and
`docs/axon_model_organization.md` for axon geometry/layout details.

## Pools And Recording

`AxonPopulation` is the explicit public container for cohorts. It stores
`AxonInstance` rows, preserves input order, and can contain one row when a
workflow should still use population execution. `simulate_pool(...)` and the
root `AxonSimulation` accept either an `AxonPopulation` or a sequence of `Axon`
or `AxonInstance` objects and return one `SimResult` per population row.

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

```python
full = axs.Recording.full()
center = axs.Recording.center(axs.signals.Vm)
probes = axs.Recording.probes(axs.signals.Vm, count=8)
indices = axs.Recording.indices([0, 10, 20], axs.signals.Vm)
```

See `docs/pool_dispatch.md` and `docs/results_recording_analysis.md` for the
full contracts.

## Analysis And Protocols

Post-hoc analysis consumes `SimResult` objects:

```python
velocity_m_s = axs.results.conduction_velocity(result)
peaks_mV = axs.results.peak_voltage(result)
spike_t_ms, spike_x_um = axs.results.rasterize(result, threshold_mV=-10.0)

event = axs.results.ActivationCriterion(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
).evaluate(result)
```

Repeated stimulation workflows live in `axs.protocols`:

```python
threshold = axs.protocols.find_activation_threshold(
    simulation_factory=lambda current: make_simulation(current),
    bounds=(1.0 * axs.uA, 500.0 * axs.uA),
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.results.ActivationCriterion(threshold=-20.0 * axs.mV),
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

Solver-side observers are future work. Use recorded traces plus post-hoc
analysis for current runnable workflows.

## Examples

Examples are the executable learning path. Keep them in sync with API changes.

```bash
python examples/basic/example_01_stimulus_waveforms.py
python examples/basic/example_02_point_source_electrode.py
python examples/basic/example_03_intracellular_hh.py
python examples/basic/example_04_extracellular_mrg.py
python examples/basic/example_05_pool_dispatch_basic.py
python examples/basic/example_06_velocity_vs_diameter.py
python examples/basic/example_07_threshold_vs_diameter.py
python examples/basic/example_08_recruitment_curve_population.py
```

Advanced workflow examples:

```bash
python examples/advanced/example_01_pool_dispatch_nrv.py --fibers 8
python examples/advanced/example_05_recording_options.py
python examples/advanced/example_06_activation_criterion.py
python examples/advanced/example_07_recruitment_curve.py
python examples/advanced/example_08_root_axon_simulation.py
python examples/advanced/example_09_axon_population.py
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
python benchmark/nrv_performance/run.py --list
```

Generated benchmark results live under ignored `benchmark/results/` and
`benchmark/reports/` paths. See `benchmark/runtime/` and
`benchmark/nrv_performance/` for detailed runners.

## Documentation

- `docs/axon_model_organization.md`: descriptive axon layer.
- `docs/membranes.md`: membrane model descriptions.
- `docs/stimulation.md`: stimuli, electrodes, clamps, and contexts.
- `docs/pool_dispatch.md`: pool dispatch and batching behavior.
- `docs/results_recording_analysis.md`: `Recording`, `SimResult`, analysis, and
  visualization.
- `docs/solver_organization.md`: solver package boundaries and time grids.
- `docs/validation.md`: fast checks and local NRV validation policy.
- `docs/api_public_draft.md`: proposal-only API notes, not a runnable reference.
- `docs/recorders_observers_activation_strategy.md`: observer/recruitment
  roadmap with current implementation status.

## Changelog

See `CHANGELOG.md`.
