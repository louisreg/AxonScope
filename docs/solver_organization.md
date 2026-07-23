# Solver And Runtime Organization

The solver package owns only stable solver-facing option contracts. The
runtime-neutral descriptive axon adapter lives under `axonfleet.runtime`.

Concrete JAX numerical execution lives under `axonfleet.runtime.jax`. That
runtime receives descriptive axons, compiles them into runtime arrays, lowers
inputs and observers, and advances state in time. It should not own public model
construction, pool grouping policy, electrode placement policy, or result
analysis.

## Files

Solver-facing contracts:

- `__init__.py`: stable solver-facing facade only. It exports `BatchOptions`
  and `BatchRecording`; kernels, runtimes, and backend solver
  resolvers are not facade exports.
- `options.py`: solver-owned execution knobs. `BatchOptions` and
  `BatchRecording` control
  batch-kernel memory, retained Vm columns, and optional time chunking.
  Per-cable solver choice is selected through typed `ExecutionPolicy.solvers`.

Runtime-neutral preparation contracts:

- `runtime/solver_axon.py`: descriptive axon to numerical array adapter shared
  by current JAX execution and future reference runtimes.

JAX runtime implementation:

- `runtime/jax/types.py`: prepared JAX runtime dataclasses consumed by kernels
  and batch preparation.
- `runtime/jax/preparation/base.py`: preparation bridge from descriptive axons to
  backend arrays: cable coefficients, stimulation callables or precomputed
  samples, extracellular absolute arrays, time grid, and membrane runtime
  assembly.
- `runtime/jax/membranes/`: JAX membrane compilation and execution helpers:
  Model IR lowering, generated-program facade, membrane backends, heterogeneous
  layouts, gated/leak row stacking, and the membrane-to-JAX compiler bridge.
- `runtime/jax/cable_geometry.py`: JAX cable-geometry helpers,
  diffusion coefficients/operators, compartment areas, and extracellular
  absolute arrays. This is runtime preparation support, not a kernel module.
- `runtime/jax/kernels/block_tridiagonal.py`: the active CPU Thomas solver for
  double-cable 2x2 block-tridiagonal systems.
- `runtime/jax/kernels/double_cable_linear.py`: double-cable linear-system
  layouts, static-term preparation, system assembly, and the Triton
  node-first solve bridge.
- `runtime/jax/kernels/single_cable.py` and
  `runtime/jax/kernels/double_cable.py`: batch kernels for homogeneous groups.
  These consume already assembled batched arrays and never decide which axons
  belong together.
- `runtime/jax/kernels/double_cable_cpu.py` and
  `runtime/jax/kernels/double_cable_gpu.py`: backend-specific double-cable
  scan bodies. CPU owns the Thomas route; GPU owns the tiled-Thomas/Triton
  route. The shared `double_cable.py` file remains the route/chunk wrapper.
- `runtime/jax/kernels/chunking.py`, `runtime/jax/kernels/factorized.py`,
  `runtime/jax/kernels/inputs.py`, and `runtime/jax/recording/results.py`:
  shared batch-kernel support for chunking, factorized inputs, array coercion,
  result waits, VmRaster finalization, and padded-output trim.
- `runtime/jax/kernels/double_cable_step.py`: shared batched membrane-step
  helpers used by batch-native double-cable paths.
- `runtime/jax/kernels/triton_double_cable.py`: optional Triton double-cable
  linear-system kernels.
- `runtime/jax/inputs/payloads.py`: JAX-side sparse/factorized input
  containers used by batch kernels.
- `runtime/jax/inputs/extracellular.py`,
  `runtime/jax/inputs/intracellular.py`, and `runtime/jax/inputs/lowering.py`:
  JAX input materialization and semantic-to-kernel input lowering.
- `runtime/jax/recording/observer.py`: JAX-side VmRaster plan/state update.
- test-only dense/reference solver variants live under `tests/unit/solvers/`,
  not in the production JAX runtime package.

## Boundaries

Dispatch decides *which axons run together*. Solver code decides *how numerical
arrays are integrated*. In particular:

- dispatch passes typed execution and batch policies without owning
  runtime-specific numerical details;
- batch kernels accept arrays such as `Iinj[B, Nt, Nx]` and
  `Vstim[B, Nt, Nx]`;
- public `Recording` objects are translated to `BatchRecording` before batch
  execution;
- solver runtime can compile public membrane descriptions, but membrane
  descriptions themselves remain computation-independent;
- public `ExecutionPolicy.solvers` carries typed per-cable solver policy, while
  backend-private string solver labels are implementation details.

## Active Solver Route Map

The retained execution surface is deliberately small. Every public simulation
path should pass through one of the routes below.

### Single-Row Batch Route

A one-axon `AxonSimulation.run(...)` uses the same population lifecycle as
larger pools. Vm/VmRaster-compatible one-row groups are normalized to a batch
route with `B=1`; there is no separate scalar execution route. Dense
solver-side observable recordings such as gates, currents, conductances, and
state variables are not currently exposed through public execution until they
are implemented on the batch route.

The one-row route is:

```text
AxonSimulation.run()
  -> run_pool(...)
  -> build_dispatch_plan(...)
  -> _run_batch_group(...)
  -> runtime.execution.enqueue_batch_group(...)
  -> runtime/jax/group_runner.enqueue_jax_batch_group(...)
  -> runtime.execution.finalize_batch_group(...)
  -> runtime/jax/group_runner.finalize_jax_batch_group(...)
  -> SingleCableVStimBatchKernel or DoubleCableBatchKernel with B=1
  -> AxonSimulationResult at the public boundary
```

Observer requests are lowered through `build_threshold_observer_plan(...)` or
evaluated post-hoc from retained Vm using the same batch result assembly path as
larger groups.

### Pool And Planning Route

Pool execution starts from `AxonSimulation.run(...)` for both one-row and
many-row populations. Public orchestration stays in `simulation.py`, while
dispatch grouping stays in `dispatcher/execution.py`:

```text
AxonSimulation.run()
  -> run_pool(...)
  -> build_dispatch_plan(...)
  -> _run_batch_group(...) for supported single/double-cable groups, including B=1
```

Unsupported group modes or unsupported recording requests fail explicitly
instead of falling back to a second execution route.

### Single-Cable Batch Route

Compatible single-cable groups enter the single-cable branch of
`runtime/jax/group_runner.enqueue_jax_batch_group(...)`.

Preparation and lowering happen in this order:

```text
prepare_batch_runtime(...)
  -> runtime.group_preparation.prepared_cohort_for_group(...)
  -> lower_observers_for_cohort(...)
  -> lower_single_cable_intracellular_input(...)
  -> lower_single_cable_extracellular_input(...)
  -> SingleCableVStimBatchKernel
  -> runtime.result_assembly.dispatch_results_from_batch(...)
```

`runtime/jax/inputs/lowering.py` owns the representation decision. It wraps
`build_sparse_intracellular_current_density_batch(...)`,
`build_intracellular_current_density_batch(...)`,
`build_factorized_vstim_midpoint_batch(...)`, and
`build_vstim_midpoint_batch(...)`, then returns a `Lowered*Input` object for the
kernel. Sparse intracellular lowering is used for compatible current-clamp
cohorts. Dense intracellular lowering remains the general path. When a
single-cable batch has compatible static-footprint extracellular stimulation,
the factorized footprint path keeps the field as current samples plus
footprints and avoids a dense `Vstim[B, Nt, Nx]` materialization. Otherwise the
route uses dense midpoint `Vstim`.

### Double-Cable Batch Route

Compatible double-cable groups enter the double-cable branch of
`runtime/jax/group_runner.enqueue_jax_batch_group(...)`.

Preparation and lowering happen in this order:

```text
prepare_batch_runtime(...)
  -> runtime.group_preparation.prepared_cohort_for_group(...)
  -> lower_observers_for_cohort(...)
  -> lower_double_cable_intracellular_input(...)
  -> lower_double_cable_extracellular_input(...)
  -> DoubleCableBatchKernel
  -> runtime.result_assembly.dispatch_results_from_batch(...)
```

The retained exact double-cable block solvers are intentionally narrow. CPU
uses the backend-private `thomas` label. GPU uses the backend-private
`jax_triton_loop_xb` tiled-Thomas label. Public users select routes through
typed per-cable solver policies on `ExecutionPolicy.solvers`; `auto` is
resolved at that policy boundary before kernel dispatch.
JAX orchestration carries the selected route as one internal `JaxSolverEngine`
value into `DoubleCableBatchKernel.run(...)`; raw solver labels and internal
flags are not parallel public or kernel-call arguments.
Inspection and reporting use one runtime-owned solver-route summary derived
from that same policy resolution.
`KernelInspection.solver` exposes the same structured contract for single- and
double-cable groups: cable family, requested typed policy label, resolved
backend route, internal/artifact flag, and route-specific options. Backend
labels such as `jax_triton_loop_xb` may appear there as resolved artifacts, not
as public policy names.

Double-cable lowering uses the same `Lowered*Input` contract as single-cable,
but a double-cable-specific strategy because the kernel needs an
initial-previous extracellular value. It may use
`build_factorized_vstim_midpoint_batch(..., include_initial_previous=True)` for
observer-only shared-current rank-1 inputs. Dense midpoint and initial-previous
`Vstim` arrays from `build_vstim_midpoint_and_initial_previous_batch(...)`
remain the explicit fallback for full recording, rank-K double-cable inputs, or
unsupported factorized inputs.

Parameter-batched double-cable groups are allowed to contain rows with different
local MRG phase shifts (`x_shift`) and different padded widths as long as they
share the same membrane-structure set. The dispatcher owns that grouping policy;
the JAX backend receives a row-indexed membrane backend plus already padded
cable/extracellular arrays.

For parameter-batched groups,
`runtime/jax/preparation/runtime.py::prepare_batch_runtime(...)` prepares only
the representative fields that survive batching. It must not build a full
representative cable/extracellular runtime just to replace it immediately with
stacked row arrays.
`runtime/jax/preparation/stacking.py` owns the JAX array stacking for cable,
membrane, extracellular, and group `Cm` rows.
`runtime/jax/membranes/stacking.py` owns the JAX-specific gated/leak membrane
row encoding used while stacking heterogeneous membrane layouts. This is a
backend preparation optimization and must remain independent of a particular
membrane-model family.
`runtime/group_preparation.py` owns dispatch-group signatures and
prepared-cohort caches. `runtime/jax/preparation/caches.py` owns only bounded JAX
runtime and input-array cache storage.

### Threshold Observers, Dense/Factorized Vext, And Results

`runtime/jax/recording/lowering.py` owns batch recording/observer lowering:
it keeps padded `center`/`probes` Vm recording row-aware when all rows retain a
common output width, expands to full Vm only when row-aware recording is not
available, and lowers compatible public observer definitions to one solver-side
`ThresholdObserverPlan` through `build_threshold_observer_plan(...)`. Batch
kernels update either bounded activation flags or packed VmRaster output during
the scan, including one-row `B=1` runs. Activation-only output uses
`observations["activation"]`; definitions requiring temporal history retain
`observations["vm_raster"]`. Result containers and CPU conversion live under
`axonfleet.results`, not in solver runtime modules.

Chunked observer-only batch kernels use local threshold states per chunk. They
combine activation with boolean OR or assemble VmRaster into one full-duration
packed raster before result assembly. The observer-only result path can keep
compact dispatch cohort records instead of
materializing one Vm trace per axon. The
host-side assembly repacks whole `uint32` word slices per chunk, including
unaligned chunk starts, so this cost stays outside solver-specific code while
avoiding per-step or per-word Python loops. This keeps the public result
identical while stabilizing JAX kernel signatures across duration sweeps.

Dense extracellular input is produced by `build_vstim_midpoint_batch(...)` or
`build_vstim_midpoint_and_initial_previous_batch(...)`. Factorized
extracellular input is produced by `build_factorized_vstim_midpoint_batch(...)`
and should remain internal to backend lowering. Factorized builders cache
temporal stimulus evaluations within a batch, so cohorts sharing
temporal-equivalent stimuli evaluate the current waveform once per time grid
while keeping row-specific spatial footprints.

Batch outputs become private dispatch row records or compact dispatch cohort
records in `runtime/outputs/assembly.py`, then `AxonSimulationResult` at the
public `AxonSimulation.run()` boundary. JAX-specific batch result helpers stay
in `runtime/jax/recording/results.py` for device wait, pending VmRaster
finalization, and padded kernel-output trim. Post-hoc observer evaluation uses
the lightweight `runtime/outputs/rows.py` adapter only as a result view, not as an
execution route.

## Solver Options

There is one solver option container and one runtime policy surface:

- `BatchOptions`: batch-kernel execution options. It carries
  `BatchRecording` and optional `time_chunk_steps`.
- `ExecutionPolicy.solvers`: typed per-cable solver policy. Use
  `SolverPolicy` plus runtime-specific constructors under `axs.runtime.jax`
  instead of raw string public solver names.

`BatchOptions.none()` is the compact observer-only batch policy. It defaults to
`DEFAULT_OBSERVER_TIME_CHUNK_STEPS`, currently `128`, so duration sweeps reuse a
stable JAX kernel chunk shape while writing observer hits into one preallocated
full-duration packed VmRaster state. The default is chosen to align with packed
VmRaster words and avoid post-chunk raster recombination on short runs. Passing
`time_chunk_steps=None` explicitly keeps the unchunked single-scan behavior.

The current typed public choices are:

| Public policy | Backend route | Use |
| --- | --- | --- |
| `axs.runtime.jax.SingleCableSolver.auto()` | current JAX tridiagonal route | Single-cable default. |
| `axs.runtime.jax.SingleCableSolver.jax_tridiagonal()` | JAX tridiagonal route | Explicit single-cable route. |
| `axs.runtime.jax.DoubleCableSolver.auto()` | CPU Thomas or GPU tiled Thomas | Double-cable default by active device. |
| `axs.runtime.jax.cpu.DoubleCableSolver.thomas()` | `thomas` | Only supported explicit CPU double-cable route. |
| `axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(...)` | tiled Thomas GPU route | Supported explicit GPU double-cable route. Runtime artifacts may still record the internal kernel label. |

CPU double-cable policy is intentionally narrow: `auto` resolves to `thomas`,
and the only explicit CPU route is `axs.runtime.jax.cpu.DoubleCableSolver.thomas()`.
Non-Thomas CPU double-cable routes are unsupported and should not be kept as
active runtime choices.

Current solver-policy constraints are owned by `GUIDELINES.md` and enforced by
the runtime policy tests.

Example:

```python
import axonfleet as axs

policy = axs.ExecutionPolicy(
    runtime=axs.runtime.jax,
    device=axs.Device.gpu(0),
    precision=axs.PrecisionPolicy.float32(),
    solvers=axs.SolverPolicy(
        single_cable=axs.runtime.jax.SingleCableSolver.jax_tridiagonal(),
        double_cable=axs.runtime.jax.DoubleCableSolver.auto(),
    ),
)
```

Forced choices are mainly diagnostic until benchmark evidence updates the
default policy. Split iterative, associative, Pallas, static Triton, CUDA FFI,
and approximate double-cable surrogate variants are removed or archived outside
active surfaces. They must not appear in user-facing docs or `BatchOptions`.
Benchmark CLIs may keep string flags, but active workload code translates them
to typed policy objects at the benchmark boundary.

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
