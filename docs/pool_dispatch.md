# Pool Dispatch

Pool simulation runs through the executable `AxonSimulation` root object.
`AxonPopulation` is the explicit public container for cohorts:

```python
import axonscope as axs

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
  -> internal dispatch items
  -> dispatch groups
  -> runtime-batch builders when a batch path is available
  -> scalar/batch solver path
  -> AxonSimulationResult with dense cohorts and per-axon views
```

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

AxonScope core uses the intrinsic one-dimensional coordinate:

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

## Advanced Dispatch

`run_pool` is the lower-level dispatch entry point. It also accepts an
`AxonPopulation` or sequence of `Axon`/`AxonInstance` objects, but returns
private dispatch results instead of public result containers:

```python
from axonscope.dispatcher import run_pool

dispatch_results = run_pool(
    population,
    tsim_ms=5.0,
    dt_ms=0.01,
)
```

Use this path for debugging dispatcher metadata. Tutorials should prefer
`AxonSimulation`.

Solver-level knobs are passed as solver options. Batch memory/recording knobs
are passed separately as batch options:

```python
import axonscope as axs

dispatch_results = run_pool(
    [sim_a, sim_b],
    tsim_ms=5.0,
    dt_ms=0.01,
    solver_options=axs.SolverOptions(),
    batch_options=axs.BatchOptions(time_chunk_steps=50),
)
```

Observer-only pool runs (`Recording.none()` plus compatible solver-side
threshold observers, or `BatchOptions.none()`) use a stable default chunk size of
`axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS` time steps. The default is aligned with
the packed VmRaster word layout while longer duration sweeps still reuse a
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
- same `n_compartments`, `Vinit`, `Veinit`, and temperature.

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
- `dispatcher/plan.py`: normalization, compatibility signatures, and groups;
- `dispatcher/execution.py`: scalar/batch execution and `run_pool`;
- `dispatcher/inspection.py`: text and Matplotlib inspection helpers for
  dispatch plans;
- `preparation/runtime_batches.py`: host-side builders for batched solver inputs
  from intracellular contexts, extracellular stimulations, sampled footprints,
  and intrinsic positions.
- `solvers/options.py`: solver-owned numerical options, batch execution
  options, and batch Vm retention policies.

Current package layout:

```text
dispatcher/
  __init__.py        dispatch surface: run_pool, build/print/plot dispatch plans
  plan.py           normalization and grouping
  execution.py      scalar/batch execution and public run_pool entry point
  inspection.py     print/plot helpers for dispatch plans

preparation/
  runtime_batches.py  host-side runtime-batch row assembly

solvers/
  options.py        solver and batch execution options
```

Use `axonscope.dispatcher` for dispatch. Keep direct imports from
`axonscope.preparation.runtime_batches` inside internal tests, benchmark tools,
or deliberate runtime-assembly debugging so the public dispatch surface stays
small.
