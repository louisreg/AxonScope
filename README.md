# AxonFleet

AxonFleet is a pre-release Python framework for one-dimensional peripheral
nerve axon simulations. It focuses on explicit physical units, descriptive
axon and stimulation objects, validated solver behavior, and batch-ready
execution for axon pools.

The public API is still evolving before publication. Use the current names in
this README and in `examples/` as the supported surface; old temporary aliases
are not kept for compatibility.

## What AxonFleet Owns

AxonFleet owns:

- one-dimensional axon descriptions and membrane dynamics;
- intracellular clamps and sampled extracellular footprints along axons;
- simulation execution through `AxonSimulation`;
- recording, results, analysis, thresholds, recruitment, sweeps, inspection,
  validation, and performance evidence.

AxonFleet does not own nerve/fascicle geometry, histology segmentation, 3D axon
trajectories, electrode CAD, surgical placement, or FEM field solving. External
tools should produce sampled extracellular footprints; AxonFleet combines those
footprints with temporal stimuli and runs the cable and membrane dynamics.

## Installation

AxonFleet uses Python `>=3.12,<3.13` and a `src/` layout.

```bash
python -m pip install -e .
```

Useful extras:

```bash
python -m pip install -e ".[examples]"
python -m pip install -e ".[dev]"
python -m pip install -e ".[benchmark]"
python -m pip install -e ".[benchmark,cuda12]"  # NVIDIA CUDA 12 + Triton
python -m pip install -e ".[dev,nrv]"
```

The base install remains CPU-only. The `cuda12` extra is the supported NVIDIA
stack and includes JAX CUDA, `jax-triton`, and Triton. The `nrv` extra installs
NRV and its declared Python dependencies; system FEM dependencies may still
require the upstream NRV environment on some platforms.

## First Simulation

```python
import axonfleet as axs

axon = axs.axons.HodgkinHuxley(
    length=500.0 * axs.um,
    diameter=0.5 * axs.um,
    compartments=41,
    celsius=6.3 * axs.degC,
)

instance = axs.AxonInstance(axon)
instance.add_current_clamp(
    position=250.0 * axs.um,
    current=axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.5 * axs.ms,
        amplitude=2.0 * axs.nA,
    ),
)

result = axs.AxonSimulation(
    instance,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
).run().single

center = result.nearest_position_index(250.0 * axs.um)
print(result.t.shape, result.Vm[:, center].shape)
```

`AxonSimulation.run()` is the convenience path. For composed work, create
immutable plans and execute them through one runner so planning state can be
reused deliberately. Population rows remain descriptive until that runner is
asked to execute, estimate, inspect, or explicitly materialize them:

```python
plan = axs.AxonSimulation(
    instance,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
).plan()

runner = axs.Runner()
estimate = runner.estimate(plan)
result = runner.run(plan)
```

Several independent or ordered plans can share the same runner state through a
backend-neutral study:

```python
study = axs.StudyPlan(
    name="comparison",
    tasks=(
        axs.StudyTask("baseline", baseline_plan),
        axs.StudyTask("follow_up", follow_up_plan, depends_on=("baseline",)),
    ),
)
results = runner.run(study)
follow_up = results["follow_up"]
```

Studies are fail-fast. `StudyExecutionError` retains completed task results, and
`CancellationToken` requests cooperative cancellation between tasks or
protocol iterations; an already dispatched kernel is allowed to finish.

Run the executable version:

```bash
python examples/basic/01_first_intracellular_simulation.py
```

## Current Public Shape

The main workflow is:

```text
membranes + axons + stimulation
        -> AxonInstance or AxonPopulation
        -> AxonSimulation(...).run()
        -> AxonSimulationResult / AxonResultView
```

Primary public namespaces:

```text
axs.axons          axon geometry, layouts, templates
axs.membranes      class-based membrane models and equation authoring
axs.stimulation    stimuli, clamps, footprints, drives, stimulations
axs.recording      recording policies through axs.Recording
axs.results        canonical simulation results and VmRaster containers
axs.analysis       post-hoc analyses and solver-side VmRaster definitions
axs.protocols      thresholds, sweeps, recruitment workflows
axs.performance    estimates, runtime/device/precision policy
axs.inspection     host-side planning and pipeline inspection reports
axs.integrations   optional bridges such as NRV handoff helpers
```

Public arguments use Pint quantities: `length`, `diameter`, `duration`, `dt`,
`current`, `threshold`, and similar physical names. Internal implementation
fields may use canonical suffixes such as `_um`, `_ms`, or `_mV`.

## Axons And Membrane Models

Built-in axon templates include Hodgkin-Huxley, Rattay-Aberham, Sundt,
Tigerholm, Schild94, Schild97, MRG, and the Gaines motor and sensory
myelinated families.

Built-in membrane model truth lives in `src/axonfleet/membranes/models/`.
User-authored membranes are plain Python classes. Put custom model classes in a
`.py` file so source inspection and generated-code caching can read their
definition:

```python
from axonfleet.membranes.types import CurrentDensity, ResistanceArea, Voltage


class DemoLeak(axs.membranes.Model):
    model_kind = "demo_leak"

    Rm: ResistanceArea = 10_000.0 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        I_l: CurrentDensity = (Vm - self.EL) / self.Rm
        return I_l
```

Start with:

```bash
python examples/advanced/axon_models/05_custom_membrane_authoring.py
```

See `docs/membranes.md` for the membrane-authoring contract.

## Stimulation

Intracellular current clamps attach to an `AxonInstance`. Extracellular
stimulation is represented as sampled one-dimensional footprints plus temporal
stimuli:

```python
stimulus = axs.Stimulus.biphasic(
    start=0.5 * axs.ms,
    cathodic_amplitude=80.0 * axs.uA,
    cathodic_duration=0.05 * axs.ms,
    interphase=0.02 * axs.ms,
)

electrode = axs.analytical.PointSourceElectrode(
    x=250.0 * axs.um,
    z=500.0 * axs.um,
)

stimulation = axs.analytical.point_source_stimulation(
    electrode,
    axon.layout.position_values(unit=axs.um) * axs.um,
    sigma=0.3 * axs.S_per_m,
    stimulus=stimulus,
)

instance.add_extracellular_stimulation(stimulation=stimulation)
```

The analytical point-source helper is a quick-start path. For realistic
workflows, external tools such as NRV/FEM should generate the field footprint,
then AxonFleet should receive sampled footprint values. See `docs/stimulation.md`
and `examples/with_nrv/01_synthetic_fascicle_geometry.py`.

## Populations, Recording, And Results

`AxonSimulation` accepts one `AxonInstance`, a sequence of axons/instances, or
an explicit `AxonPopulation`.

```python
population = axs.AxonPopulation([instance_a, instance_b], name="demo")

results = axs.AxonSimulation(
    population,
    duration=1.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
).run()

for row in results:
    print(row.axon_id, row.record_indices, row.Vm.shape)
```

Recording policies control retained output:

```python
full = axs.Recording.full()
center = axs.Recording.center(axs.signals.Vm)
probes = axs.Recording.probes(axs.signals.Vm, count=8)
indices = axs.Recording.indices([0, 10, 20], axs.signals.Vm)
```

For trace-free threshold-style observer runs, use compatible analysis
definitions with `Recording.none()`. Activation-only requests retain one
boolean per axon under `observations["activation"]`; latency-only requests
retain one first-crossing timestep and return physical time under
`observations["latency"]`. `SpikeCount` retains count/first/last in constant
memory and optionally `max_spikes=K` timestamps plus explicit overflow.
`VmRaster(every_n_steps=N, target=...)` retains packed spatial probes using an
event-preserving OR over each temporal window under
`observations["vm_raster"]`; exact event analyses keep their dedicated compact
state rather than using a downsampled raster implicitly.

See `docs/results_recording_analysis.md` and `docs/pool_dispatch.md`.

## Runtime, Estimates, And Inspection

Runtime selection uses typed public values:

```python
policy = axs.ExecutionPolicy(
    runtime=axs.runtime.jax,
    device=axs.Device.cpu(),
    precision=axs.PrecisionPolicy.float32(),
)
```

Estimate and inspect before large runs:

```python
simulation = axs.AxonSimulation(
    population,
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
    execution_policy=policy,
)

estimate = simulation.estimate()
inspection = simulation.inspect(print_summary=True)
```

These reports are host-side planning tools. They route backend-specific lowering
facts through the backend execution boundary and do not run solver kernels by
default.

## Examples

Examples are executable documentation:

```bash
python examples/basic/01_first_intracellular_simulation.py
python examples/basic/03_point_source_footprint.py
python examples/basic/05_population_pool_run.py
python examples/basic/08_recruitment_curve_population.py
python examples/advanced/runtime/02_pipeline_inspection.py
python examples/advanced/observers/01_vmraster_observer_only.py
python examples/advanced/protocols/02_recruitment_waveforms.py
```

See `examples/README.md` for the full learning path. Benchmark and profiling
material lives under `benchmark/`, not under public examples.

Persistent generated-code and JAX/Triton artifacts share one deterministic
tree. Inspect it with `axs.cache.inspect()`, clean known sections with
`axs.cache.clean()`, relocate it with `AXONFLEET_CACHE=/path`, or disable
persistence with `AXONFLEET_CACHE=off`. See `docs/cache.md`.

## Tests And Validation

Fast local checks:

```bash
git diff --check
MPLBACKEND=Agg python -m compileall -q src tests/unit
MPLBACKEND=Agg python -m pytest -q tests/unit --tb=short
```

Optional NRV validation:

```bash
MPLBACKEND=Agg python -m pytest -q tests/nrv --tb=short
```

Run NRV validation when numerical behavior, solver semantics, stimulation
semantics, membrane dynamics, or NRV integration changes. Do not record stale
NRV pass counts. See `docs/validation.md`.

## Benchmarks

Benchmarks provide reproducible performance evidence, not scientific
correctness validation.

```bash
python benchmark/run.py --list
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu
python benchmark/run.py --script basic_examples --preset quick --platform cpu
```

Use:

- the registered curve and example workloads for timing and memory claims;
- the focused runtime, membrane, solver, and campaign scripts listed in
  `benchmark/README.md` for deeper profiling;
- `tests/nrv` for scientific validation before AxonFleet-vs-NRV comparisons.

Generated outputs live under ignored `benchmark/results/` and
`benchmark/reports/`. See `benchmark/README.md`.

## Documentation Map

- `GUIDELINES.md`: project philosophy and target architecture.
- `todo.md`: living cleanup and roadmap checklist.
- `docs/axon_model_organization.md`: axon descriptions and layouts.
- `docs/membranes.md`: membrane model authoring.
- `docs/stimulation.md`: stimuli, footprints, drives, and stimulations.
- `docs/results_recording_analysis.md`: results, recording, analysis, plots.
- `docs/pool_dispatch.md`: populations, dispatch, batching.
- `docs/solver_organization.md`: solver and backend boundaries.
- `docs/validation.md`: fast checks and NRV validation policy.
- `benchmark/README.md`: supported benchmark surface.

## Changelog

See `CHANGELOG.md`.
