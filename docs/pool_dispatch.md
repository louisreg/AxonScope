# Pool Dispatch

Pool simulation is intentionally list-based for now:

```python
import axonscope as axs

sim_a = axs.AxonSimulation(axon_a, y_um=20.0 * axs.um, z_um=30.0 * axs.um)
sim_b = axs.AxonSimulation(axon_b, y_um=-40.0 * axs.um, z_um=10.0 * axs.um)

results = axs.simulate_pool(
    [sim_a, sim_b],
    duration_ms=5.0 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center("Vm"),
    progress=True,
)
```

There is no public pool container and no public `Fiber` wrapper. `Axon`
objects carry descriptive geometry/membrane information; `AxonSimulation`
objects carry per-fiber positions and stimulation protocols. The public pool
input is simply `Sequence[Axon | AxonSimulation]`.

## Public Flow

```text
list[Axon | AxonSimulation]
  -> internal dispatch items
  -> dispatch groups
  -> runtime-batch builders when a batch path is available
  -> scalar/batch solver path
  -> list[SimResult]
```

`simulate_pool` returns one `SimResult` per input item, in the same order:

```python
for result in results:
    print(result.diagnostics["pool_index"], result.simulation.y_um, result.simulation.z_um)
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

`SimResult.axon` is the pure descriptive axon. `SimResult.simulation` is the
protocol object that was simulated. There is no separate pool container in the
public result.

## Spatial Position

Set pool placement on each simulation protocol:

```python
sim = axs.AxonSimulation(axon)
sim.set_position(y_um=20.0 * axs.um, z_um=30.0 * axs.um, x_offset_um=0.0 * axs.um)
```

Plain numeric positions are interpreted as micrometers. Pint quantities are
converted at the simulation boundary.

## Stimulation Contexts

For now, each simulation carries its own intracellular and extracellular
contexts:

```python
sim.add_intracellular_context(
    context=axs.IntracellularCurrentClamp(
        position_um=250.0 * axs.um,
        current=stimulus,
    )
)
sim.add_extracellular_context(
    context=axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(extra_current)],
        sigma=0.3 * axs.S_per_m,
    )
)
```

That keeps the public model didactic: a pool is a collection of already
described axon simulations.

Point-source electrode coordinates are interpreted in the same global frame as
`sim.set_position(...)`. The dispatcher/runtime converts each point source to
the transverse offset seen by each axon before building batched `Vstim` arrays.

## Advanced Dispatch

`run_pool` is the lower-level dispatch entry point. It also accepts
`Sequence[Axon | AxonSimulation]`, but returns private dispatch results instead
of public `SimResult` objects:

```python
from axonscope.dispatcher import run_pool

dispatch_results = run_pool(
    [sim_a, sim_b],
    tsim_ms=5.0,
    dt_ms=0.01,
)
```

Use this path for debugging dispatcher metadata. Tutorials should prefer
`simulate_pool`.

Solver-level knobs are passed as solver options. Batch memory/recording knobs
are passed separately as batch options:

```python
from axonscope.channel_models import RateTableConfig
from axonscope.solvers import BatchOptions, SolverOptions

dispatch_results = run_pool(
    [sim_a, sim_b],
    tsim_ms=5.0,
    dt_ms=0.01,
    solver_options=SolverOptions(
        rate_table_config=RateTableConfig(step_mV=0.05),
    ),
    batch_options=BatchOptions(time_chunk_steps=50),
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
largest `Nx` in the group. Padding is solver-internal: public `SimResult.Vm`
is sliced back to the original axon width, and center/probe recordings are
resolved against each original axon.

The intended per-row differences in a batch are cable geometry, attached
stimulation contexts, intracellular contexts, spatial offsets, and, for
padded double-cable groups, the number of compartments. Per-row membrane
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
    duration_ms=10.0 * axs.ms,
    dt_ms=0.005 * axs.ms,
    batch_options=axs.solvers.BatchOptions.full(time_chunk_steps=200),
    recording=axs.Recording.voltage(),
    progress=True,
)
```

The old precomputed global extracellular API and policy-specific public batch
paths have been removed. Runtime-batch construction now starts from
already-attached axon contexts or precomputed electrode footprints, then passes
numeric arrays to solver kernels.

```python
from axonscope.dispatcher.runtime_batches import build_footprint_vstim_midpoint_batch
from axonscope.solvers import SingleCableVStimBatchKernel

vstim_mid = build_footprint_vstim_midpoint_batch(
    stimulus=stimulus,
    footprint_V_per_A=footprint_V_per_A,
    tsim_ms=5.0,
    dt_ms=0.01,
)

result = SingleCableVStimBatchKernel(
    runtime,
    Cm_uF_cm2=runtime.axon.Cm_uF_cm2,
).run(
    extracellular_potential_mid_mV=vstim_mid,
)
```

This boundary is intentional: `dispatcher` knows about public axons, contexts,
spatial shifts, footprints, and grouping policy; `solvers` know about arrays,
time integration, and numerical state.

## Module Responsibilities

Current files:

- `simulation.py`: public `simulate` and `simulate_pool` wrappers;
- `dispatcher/plan.py`: normalization, compatibility signatures, and groups;
- `dispatcher/execution.py`: scalar/batch execution and `run_pool`;
- `dispatcher/inspection.py`: text and Matplotlib inspection helpers for
  dispatch plans;
- `dispatcher/runtime_batches.py`: builders for batched solver inputs from
  intracellular contexts, extracellular contexts, per-row positions, and
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
