# Pool Dispatch

Pool simulation can be run directly with `simulate_pool(...)` or through the
executable `AxonSimulation` root object. `AxonPopulation` is the explicit
public container for cohorts:

```python
import axonscope as axs

sim_a = axs.AxonInstance(axon_a)
sim_b = axs.AxonInstance(axon_b)
population = axs.AxonPopulation([sim_a, sim_b], name="demo pool")

results = axs.simulate_pool(
    population,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
    progress=True,
)
```

The equivalent root-object form is:

```python
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

`simulate_pool` returns an `AxonSimulationResult`. Indexing or iterating over it
returns one `AxonResultView` per input item, in the same order:

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
`simulate_pool`.

Solver-level knobs are passed as solver options. Batch memory/recording knobs
are passed separately as batch options:

```python
import axonscope as axs
from axonscope.channel_models import RateTableConfig

dispatch_results = run_pool(
    [sim_a, sim_b],
    tsim_ms=5.0,
    dt_ms=0.01,
    solver_options=axs.SolverOptions(
        rate_table_config=RateTableConfig(step_mV=0.05),
    ),
    batch_options=axs.BatchOptions(time_chunk_steps=50),
)
```

Compatible groups use batch kernels automatically. Incompatible groups fall
back to scalar solves. The dispatcher now distinguishes three practical cases:

- **strict batch**: all rows share one exact cable geometry/runtime shape;
- **parameter batch**: rows share one compiled model and `Nx`, but cable
  geometry differs by row;
- **padded parameter batch**: double-cable rows share a compatible membrane
  prefix, but `Nx` differs, so the dispatcher pads shorter rows internally.

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
and, for padded double-cable groups, the number of compartments. Per-row membrane
parameters are not batched yet; if membrane signatures differ, the dispatcher
keeps rows in separate groups.

Inspect plans before running a pool with:

```python
plan = axs.dispatcher.build_dispatch_plan(simulations)
axs.dispatcher.print_dispatch_plan(plan)
axs.dispatcher.plot_dispatch_plan(plan)
```

Progress reporting is optional. `progress=True` uses Rich when installed and
falls back to plain text otherwise. `progress="rich"` requires Rich, while
`progress="plain"` always uses simple text output. JAX kernels cannot stream
from inside one compiled scan; for finer solver progress, run chunked batches:

```python
results = axs.simulate_pool(
    simulations,
    duration=10.0 * axs.ms,
    dt=0.005 * axs.ms,
    batch_options=axs.BatchOptions.full(time_chunk_steps=200),
    recording=axs.Recording.voltage(),
    progress=True,
)
```

The old precomputed global extracellular API and policy-specific public batch
paths have been removed. Runtime-batch construction now starts from
already-attached axon contexts or precomputed electrode footprints, then passes
numeric arrays to solver kernels.

Advanced users should inspect this lowering through dispatch plans, benchmark
spans, or the planned pipeline inspection surface rather than importing
backend input builders or solver kernels directly from public examples. The
current JAX tensor builders live behind the JAX backend boundary and are not a
stable user API.

This boundary is intentional: `dispatcher` knows about public axons, contexts,
local footprints, and grouping policy; `solvers` know about arrays, time
integration, and numerical state.

## Module Responsibilities

Current files:

- `simulation.py`: public `simulate` and `simulate_pool` wrappers;
- `dispatcher/plan.py`: normalization, compatibility signatures, and groups;
- `dispatcher/execution.py`: scalar/batch execution and `run_pool`;
- `dispatcher/inspection.py`: text and Matplotlib inspection helpers for
  dispatch plans;
- `dispatcher/runtime_batches.py`: builders for batched solver inputs from
  intracellular contexts, extracellular contexts, intrinsic positions, and
  electrode footprints.
- `solvers/options.py`: solver-owned numerical options, batch execution
  options, and batch Vm retention policies.

Current package layout:

```text
dispatcher/
  __init__.py        dispatch surface: run_pool, build/print/plot dispatch plans
  plan.py           normalization and grouping
  execution.py      scalar/batch execution and public run_pool entry point
  inspection.py     print/plot helpers for dispatch plans
  runtime_batches.py  advanced runtime-batch input assembly

solvers/
  options.py        solver and batch execution options
```

Use `axonscope.dispatcher` for dispatch. Import advanced batch-input builders
from `axonscope.dispatcher.runtime_batches` so the dispatch surface stays small.
