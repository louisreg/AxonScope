# Solver And Backend Organization

The solver package owns stable solver-facing contracts: solver classes, solver
options, and the descriptive axon adapter used by execution backends.

Concrete JAX numerical execution lives under `axonscope.backends.jax`. That
backend receives descriptive axons, compiles them into runtime arrays, lowers
inputs and observers, and advances state in time. It should not own public model
construction, pool grouping policy, electrode placement policy, or result
analysis.

## Files

Solver-facing contracts:

- `__init__.py`: stable solver facade only. It exports `Solver`,
  `CrankNicholson`, `SolverOptions`, `BatchOptions`, `BatchRecording`, and
  `resolve_double_cable_block_solver`; kernels and runtimes are not facade
  exports.
- `base.py`: abstract solver class.
- `axon_runtime.py`: backend-facing descriptive axon adapter.
- `crank_nicholson.py`: public optimized solver class; delegates concrete
  execution to `backends/jax/scalar_runner.py`.
- `options.py`: solver-owned execution knobs. `SolverOptions` controls runtime
  preparation, currently rate tables. `BatchOptions` and `BatchRecording`
  control batch-kernel memory, retained Vm columns, optional time chunking, and
  the exact double-cable block linear-solver choice.

JAX backend implementation:

- `backends/jax/runtime.py`: bridge from descriptive axons/membranes to
  backend arrays:
  membrane backend, cable coefficients, stimulation callables or precomputed
  samples, extracellular absolute arrays, and time grid.
- `backends/jax/common.py`: numerical helpers shared by kernels, such as
  tridiagonal coefficients, diffusion operators, and small reference linear
  solvers.
- `backends/jax/kernels.py`: scalar single-axon kernels. These consume
  `SolverRuntime` and return raw `KernelResult` values.
- `backends/jax/batch_kernels.py`: batch kernels for homogeneous groups. These
  consume already assembled batched arrays and never decide which axons belong
  together.
- `backends/jax/batch_inputs.py`: JAX-side sparse/factorized input containers
  and materializers used by batch kernels.
- `backends/jax/observer_runtime.py`: JAX-side VmRaster plan/state update.
- `backends/jax/observables.py`: packaging helpers for membrane observables
  produced inside solver scans.
- `backends/jax/experimental.py`: prototype/reference solver variants used by
  tests and benchmarks.

## Boundaries

Dispatch decides *which axons run together*. Solver code decides *how numerical
arrays are integrated*. In particular:

- dispatch may pass `SolverOptions` through, but it should not inspect rate
  table settings;
- batch kernels accept arrays such as `Iinj[B, Nt, Nx]` and
  `Vstim[B, Nt, Nx]`;
- public `Recording` objects are translated to `BatchRecording` before batch
  execution;
- solver runtime can compile public membrane descriptions, but membrane
  descriptions themselves remain computation-independent.
- pseudo-double/pseudo-MRG validation modes are not solver options; they live
  under `benchmark/pseudo_double/` and must not be selected by `auto`.

## Active Solver Route Map

The retained execution surface is deliberately small. Every public simulation
path should pass through one of the routes below.

### Scalar Route

One-axon execution starts from `axs.simulate(...)` or `AxonSimulation.run(...)`
in `simulation.py`. Unless a caller supplies another solver, the public facade
creates `CrankNicholson`, then `CrankNicholson.solve(...)` delegates across the
backend boundary to `backends/jax/scalar_runner.py`.

The scalar JAX route is:

```text
AxonSimulation/run helper
  -> CrankNicholson.solve(...)
  -> run_jax_crank_nicholson(...)
  -> build_solver_axon(...)
  -> backends/jax/runtime.prepare_solver_runtime(...)
  -> backends/jax/kernels.SingleCableKernel or DoubleCableKernel
  -> internal scalar result
  -> AxonSimulationResult at the public boundary
```

`SingleCableKernel` covers normal single-cable solves and single-cable imposed
extracellular forcing. `DoubleCableKernel` is selected only for double-cable
axons with extracellular context. Scalar observer requests are lowered through
`build_vm_raster_plan(...)` before the kernel scan.

### Pool, Planning, And Fallback Route

Pool execution starts from `axs.simulate_pool(...)` or population
`AxonSimulation.run(...)`. Public orchestration stays in `simulation.py`, while
dispatch grouping stays in `dispatcher/execution.py`:

```text
simulate_pool(...)
  -> run_pool(...)
  -> build_dispatch_plan(...)
  -> _run_batch_group(...) when group size >= 2 and mode is single/double
  -> _run_scalar_group(...) otherwise
```

The scalar fallback route is intentional. Singletons, unsupported group modes,
and any future group kind that has not been promoted to batch execution must go
through `_run_scalar_group(...)`, which calls the same `CrankNicholson` scalar
route as one-axon public simulation.

### Single-Cable Batch Route

Compatible single-cable groups enter
`backends/jax/group_runner._run_single_cable_batch_group(...)`.

Preparation and lowering happen in this order:

```text
_prepare_batch_runtime(...)
  -> _prepared_cohort_for_group(...)
  -> _observer_plan_for_cohort(...)
  -> build_sparse_intracellular_current_density_batch(...)
     or build_intracellular_current_density_batch(...)
     or zero intracellular input
  -> build_factorized_vstim_midpoint_batch(...)
     or build_vstim_midpoint_batch(...)
  -> SingleCableVStimBatchKernel
  -> _dispatch_results_from_batch(...)
```

Sparse intracellular lowering is used for compatible current-clamp cohorts.
Dense intracellular lowering remains the general path. When an observer-only
single-cable batch has compatible point-source extracellular stimulation, the
factorized path keeps the field as current samples plus footprints and avoids a
dense `Vstim[B, Nt, Nx]` materialization. Otherwise the route uses dense
midpoint `Vstim`.

### Double-Cable Batch Route

Compatible double-cable groups enter
`backends/jax/group_runner._run_double_cable_batch_group(...)`.

Preparation and lowering happen in this order:

```text
_prepare_batch_runtime(...)
  -> _prepared_cohort_for_group(...)
  -> _observer_plan_for_cohort(...)
  -> build_intracellular_current_density_batch(...)
     or zero intracellular input
  -> build_factorized_vstim_midpoint_batch(..., include_initial_previous=True)
     or build_vstim_midpoint_and_initial_previous_batch(...)
  -> DoubleCableBatchKernel
  -> _dispatch_results_from_batch(...)
```

The retained exact double-cable block solvers are `thomas`, `pcr`, `pcr_soa`,
and `pcr_adaptive`, plus public `auto` resolution. `auto` is resolved before
kernel dispatch from the effective execution device. `pcr_adaptive` selects
`pcr_soa` for batches up to `B=4096`, then matrix-layout `pcr` above that.

Double-cable observer-only batches may use factorized point-source
extracellular lowering when the inputs are compatible. Dense midpoint and
initial-previous `Vstim` arrays remain the fallback for full recording or
unsupported factorized inputs.

### VmRaster, Dense/Factorized Vext, And Results

Public observer definitions are lowered to solver-side VmRaster plans by
`build_vm_raster_plan(...)`. Scalar kernels and batch kernels update packed
observer output during the scan. The public result key is strictly
`observations["vm_raster"]`; activation, latency, velocity, threshold, and
recruitment stay in post-processing. The result container and CPU unpacking live
under `axonscope.results`, not in solver runtime modules.

Dense extracellular input is produced by `build_vstim_midpoint_batch(...)` or
`build_vstim_midpoint_and_initial_previous_batch(...)`. Factorized
extracellular input is produced by `build_factorized_vstim_midpoint_batch(...)`
and should remain internal to backend lowering.

Scalar solver output is an internal payload converted to `AxonSimulationResult`
at the public `simulate(...)` boundary. Batch results become `DispatchResult`
rows or compact `DispatchCohortResult` records in
`_dispatch_results_from_batch(...)`, then `AxonSimulationResult` in the public
`simulate_pool(...)` boundary.

## Solver Options

There are two solver option containers:

- `SolverOptions`: numerical preparation options shared by scalar and batch
  execution. It currently carries `rate_table_config`.
- `BatchOptions`: batch-kernel execution options. It carries
  `BatchRecording`, optional `time_chunk_steps`, and
  `double_cable_block_solver`.

The current exact double-cable block-solver options are:

| Option | Resolution | Use |
| --- | --- | --- |
| `auto` | CPU/default backends resolve to `thomas`; GPU-like backends resolve to `pcr_adaptive`. | Normal default. |
| `thomas` | Uses the specialized exact block-Thomas scan. | CPU/default fallback and reference path. |
| `pcr` | Uses the exact matrix-layout parallel cyclic-reduction variant. | GPU diagnostic and larger-batch adaptive fallback. |
| `pcr_soa` | Uses the exact struct-of-arrays PCR variant. | GPU diagnostic for small/medium batches. |
| `pcr_adaptive` | Uses `pcr_soa` for batches up to `B=4096`, and `pcr` above that. | Explicit reproduction of the current GPU `auto` policy. |

Example:

```python
import axonscope as axs

batch_options = axs.BatchOptions.none(
    double_cable_block_solver="auto",
)
```

Forced choices are mainly diagnostic until benchmark evidence updates the
default policy. Split iterative, associative, Pallas, Triton, JAX-Triton,
CUDA FFI, and pseudo-double variants are archived or standby evidence. They
must not appear in user-facing docs or `BatchOptions` unless a later campaign
promotes one into the retained public solver surface.

## Time Grid

Current kernels use a fixed time step for every integration step. Therefore
the internal millisecond duration must be an integer multiple of the internal
millisecond step; otherwise the runtime raises `ValueError` instead of silently
rounding up and simulating past the requested final time. Public wrappers use
`duration` and `dt`, then convert them to internal `duration_ms`/`dt_ms` values
at the solver boundary.

The recorded time vector contains post-step samples:

```text
dt_ms, 2*dt_ms, ..., duration_ms
```

Midpoint stimulation samples are evaluated at:

```text
0.5*dt_ms, 1.5*dt_ms, ..., duration_ms - 0.5*dt_ms
```
