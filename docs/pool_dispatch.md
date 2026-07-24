# Pool Dispatch

Pool simulation runs through the executable `AxonSimulation` root object.
`AxonPopulation` is the explicit public container for cohorts:

```python
import axonfleet as axs

sim_a = axs.AxonInstance(axon_a)
sim_b = axs.AxonInstance(axon_b)
population = axs.AxonPopulation([sim_a, sim_b], name="demo pool")

simulation = axs.AxonSimulation(
    population,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
    progress=True,
)
results = simulation.run()
```

There is no public `Fiber` wrapper. `Axon` objects carry descriptive
geometry/membrane information; `AxonInstance` objects carry per-row stimulation
protocols in the axon's intrinsic coordinate system; `AxonPopulation` preserves
the ordered cohort that is sent to dispatch.

## Public Flow

```text
AxonSimulation or AxonPopulation
  -> immutable PopulationPlan / SimulationPlan / SweepPlan
  -> Runner
  -> internal dispatch items
  -> dispatch groups
  -> runtime-batch builders when a batch path is available
  -> scalar/batch solver path
  -> AxonSimulationResult with dense cohorts and per-axon views
```

`AxonSimulation.run()` builds a `SimulationPlan` and delegates it to a
`Runner`. Its ordered inputs remain a `PopulationPlan` until the runner
materializes and caches the canonical `AxonPopulation`; explicit `estimate()`,
`inspect()`, or access to `AxonSimulation.population` intentionally cross the
same boundary. Generic pool sweeps and recruitment sweeps follow the same route:
`pool_sweep_plan(...)` and `recruitment_sweep_plan(...)` return immutable
`SweepPlan` descriptions, while `find_threshold_plan(...)` returns a
`ThresholdPlan`. Their convenience functions execute those plans immediately.
Numeric-axis preparation, value chunk scheduling, callable-bound resolution,
and threshold bisection happen only inside the runner.

Build a plan explicitly when several operations should intentionally share one
runner's preparation cache or when work must be inspected before execution:

```python
plan = axs.protocols.recruitment_sweep_plan(
    population,
    update=update,
    values=amplitudes,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.analysis.Activation(),
)
curve_rows = axs.Runner().run(plan)
```

Each runner owns its materialized population, dispatch plans, and bounded
`PreparedCohort` cache. `Runner.clear()` drops all three. Concrete runtime
caches for immutable compiled membrane programs, JAX executables, and Triton
artifacts remain backend/process-owned so a new runner does not force needless
recompilation.

Membrane descriptions follow the same lazy boundary. `Model`, `Composite`,
`SectionLayout`, `Section`, and flattened geometry retain ordinary Python
descriptions; reading positions, diameters, or plotting a layout does not
compile membrane source. Runner preparation resolves each distinct description
once while building solver axons. Parameter and unit errors that require the
membrane compiler therefore surface from `run()`, `estimate()`, `inspect()`, or
explicit membrane introspection rather than from descriptive constructors.

For several plans, `StudyPlan` adds stable named dependencies without adding a
protocol-specific scheduler. `Runner` executes its topological order, reports
each task through `runner.study.task` benchmark spans, and fails before running
dependent or later work. Completed values remain available on
`StudyExecutionError.completed`. `CancellationToken` is checked at safe runner
boundaries; in-flight backend kernels complete normally.

`AxonSimulation.run()` returns an `AxonSimulationResult`. Indexing or iterating
over it returns one `AxonResultView` per input item, in the same order:

```python
for result in results:
    print(result.diagnostics["pool_index"], result.diagnostics["dispatch_method"])

center_vm = results.signal(axs.signals.Vm)
```

Dispatch metadata stays diagnostic:

```python
result.diagnostics["pool_index"]
result.diagnostics["dispatch_group_id"]
result.diagnostics["dispatch_method"]
result.diagnostics["dispatch_group_size"]
result.diagnostics["dispatch_geometry_shared"]
result.diagnostics["dispatch_has_padding"]
```

`AxonResultView.axon` is the pure descriptive axon. `AxonResultView.simulation`
is the protocol object that was simulated. Results remain one view per
population row. The view is the public one-axon result surface.

## Spatial Position

AxonFleet core uses the intrinsic one-dimensional coordinate:

```text
s = 0 ... axon length
```

World/anatomical coordinates belong to external geometry packages or to small
analytical helper code in examples. If a point-source setup starts in an
external frame, sample it into a local footprint before attaching it to the
simulation protocol:

```python
electrode = axs.analytical.PointSourceElectrode(
    x=500.0 * axs.um,
    y=0.0 * axs.um,
    z=100.0 * axs.um,
)
extracellular = axs.analytical.point_source_stimulation(
    electrode,
    axon.layout.position_values(unit=axs.um) * axs.um,
    sigma=0.3 * axs.S_per_m,
    stimulus=extra_current,
    axon_y=20.0 * axs.um,
    axon_z=30.0 * axs.um,
)

sim = axs.AxonInstance(axon)
sim.add_extracellular_stimulation(stimulation=extracellular)
```

The offsets above are inputs to the helper only. They are not stored on
`AxonInstance`, and they do not become solver state.

## Stimulation Contexts

For now, each simulation carries its own intracellular and extracellular
contexts:

```python
sim.add_intracellular_context(
    context=axs.IntracellularCurrentClamp(
        position=250.0 * axs.um,
        current=stimulus,
    )
)
sim.add_extracellular_stimulation(stimulation=extracellular)
```

That keeps the public model didactic: a pool is a collection of already
described axon simulations.

Point-source helper coordinates are interpreted only while building sampled
footprints. By the time dispatch/preparation starts, each row has local sampled
extracellular stimulation. The dispatcher/runtime never requires a world
position on the axon instance.

## Execution Controls

There is no second raw pool execution API. `Runner` owns group scheduling and
canonical result assembly; user workflows enter it through
`AxonSimulation.run()` or an explicit runnable plan. Solver-level knobs are
passed as solver options, while batch memory and recording knobs are passed as
`BatchOptions` to `AxonSimulation`.

Observer-only pool runs (`Recording.none()` plus compatible solver-side
threshold observers, or `BatchOptions.none()`) use a stable default chunk size of
`axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS` time steps, currently `512`. The default
is aligned with the packed VmRaster word layout while longer duration sweeps reuse a
stable JAX kernel shape. The backend keeps the solver loop chunked but writes
observer hits into one preallocated full-duration packed VmRaster state, so the
default path avoids post-chunk raster recombination. Pass
`BatchOptions.none(time_chunk_steps=None)` explicitly when a single unchunked
scan is desired.

Compatible groups use batch kernels automatically. Incompatible groups fall
back to scalar solves. Observer-only singleton groups also use the batch route,
so compact population observers do not fork through the scalar solver. The
dispatcher now distinguishes three practical cases:

- **strict batch**: all rows share one exact cable geometry/runtime shape;
- **parameter batch**: rows share one compiled model and `Nx`, but cable
  geometry differs by row;
- **padded parameter batch**: double-cable rows share the same set of membrane
  families, but `Nx` or local section phase differs, so the dispatcher pads
  shorter rows internally and uses a row-indexed membrane backend.

A compatible non-padded single-cable batch group currently means:

- same cable formulation (`single-cable` or `double-cable`);
- same membrane layout/signature;
- same `n_compartments`, initial membrane voltage, and temperature.

When local cable/periaxonal arrays are identical the method is reported as
`batch-single-cable` or `batch-double-cable`. When those arrays vary but the
compiled membrane/runtime shape remains compatible, the method is reported as
`parameter-batch-single-cable` or `parameter-batch-double-cable`.

For double-cable groups, the dispatcher may also pad shorter rows to the
largest `Nx` in the group. Padding is solver-internal: public view `Vm` arrays
are sliced back to the original axon width, and center/probe recordings are
resolved against each original axon.

The intended per-row differences in a batch are cable geometry, attached
stimulation contexts, intracellular contexts, local extracellular footprints,
and, for padded double-cable groups, the number of compartments and intrinsic
layout phase (`MRG(..., x_shift=...)`). Per-row membrane parameter values are
prepared through row runtimes when the backend supports it; if the membrane
structure set differs, the dispatcher keeps rows in separate groups.

Inspect plans before running a pool with:

```python
inspection = axs.AxonSimulation(
    simulations,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
).inspect()

inspection.print()
inspection.plot()
```

Progress reporting is optional. `progress=True` uses Rich, while
`progress="plain"` uses simple text output that is easy to capture in logs. The
report is event-based: it announces the selected route for each dispatch group,
batch/recording planning, runtime/cohort preparation, input lowering,
cold compilation points, kernel solving, and result assembly. The
`compiling JAX kernel if needed` line is the one that can take time on a
first call because JAX compilation happens inside the kernel dispatch. JAX
kernels cannot stream from inside one compiled scan; for finer solver progress,
run chunked batches:

```python
results = axs.AxonSimulation(
    simulations,
    duration=10.0 * axs.ms,
    dt=0.005 * axs.ms,
    batch_options=axs.BatchOptions.full(time_chunk_steps=200),
    recording=axs.Recording.voltage(),
    progress=True,
).run()
```

Plain output is intentionally compact, for example:

```text
plan    building dispatch plan (rows=4)
Dispatch progress: 4 rows, 2 groups
group   g0 1/2 batch-single-cable (rows=3, Nx=101)
  route   g0 1/2  compatible batch route (batch, strict, padding=no)
  prepare g0 1/2  runtime (mode=single)
  prepare g0 1/2  cohort rows
  batch   g0 1/2  recording plan (recording=VmRaster)
  lower   g0 1/2  inputs (sparse_current_clamp -> factorized_footprint)
  kernel  g0 1/2  compiling JAX kernel if needed (recording=VmRaster)
  kernel  g0 1/2  solving JAX kernel
  kernel  g0 1/2  completed JAX kernel
  result  g0 1/2  assemble batch output (output=observations)
Simulation run completed: total=..., cold_start=..., rss=...
```

The old precomputed global extracellular API, public context/electrode
adapters, and policy-specific public batch paths have been removed.
Runtime-batch construction now starts from attached
`ExtracellularStimulation` objects and sampled intrinsic footprints, then
passes numeric arrays to solver kernels.

Advanced users should inspect this lowering through dispatch plans, benchmark
spans, or `AxonSimulation.inspect()` rather than importing backend input
builders or solver kernels directly from public examples. Tensor builders live
behind the backend boundary and are not a stable user API.

This boundary is intentional: public simulation and dispatcher code know about
public axons, intracellular contexts, extracellular stimulations, sampled
footprints, and grouping policy; backend code knows about arrays, time
integration, and numerical state.

## Module Responsibilities

Current files:

- `simulation.py`: public `AxonSimulation` orchestration;
- `runner.py`: plan execution, dispatch-plan reuse, group scheduling, and
  canonical result assembly;
- `dispatcher/plan.py`: normalization, compatibility signatures, and groups;
- `inspection/`: public simulation and dispatch-plan inspection records and
  views;
- `preparation/`: host-side axon, membrane, cohort, and stimulation row
  materialization;
- `runtime/execution.py`: the sole public-orchestration boundary into the JAX
  backend;
- `solvers/options.py`: batch execution and Vm-retention contracts.

Current package layout:

```text
dispatcher/
  plan.py             normalization and grouping
  execution.py        batch scheduling and execution

preparation/
  axon_rows.py         host numerical axon rows
  membrane_rows.py     membrane row plans
  stimulation_rows.py enabled extracellular rows
  cohort.py            aligned prepared group contract

runtime/
  execution.py        backend-neutral orchestration facade

solvers/
  options.py          batch execution and Vm retention
```

Users should enter through `AxonSimulation.run()`, `.estimate()`, and
`.inspect()`. Dispatcher, preparation, and concrete runtime modules are
internal implementation boundaries, not parallel public execution surfaces.
