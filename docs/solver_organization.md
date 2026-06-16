# Solver Organization

The solver package owns numerical execution. It receives descriptive axons,
compiles them into runtime arrays, and advances state in time. It should not
own public model construction, pool grouping, electrode placement policy, or
result analysis.

## Files

- `options.py`: solver-owned execution knobs. `SolverOptions` controls runtime
  preparation, currently rate tables. `BatchOptions` and `BatchRecording`
  control batch-kernel memory, retained Vm columns, optional time chunking, and
  the exact double-cable block linear-solver choice.
- `runtime.py`: bridge from descriptive axons/membranes to solver-side arrays:
  membrane backend, cable coefficients, stimulation callables or precomputed
  samples, extracellular absolute arrays, and time grid.
- `common.py`: numerical helpers shared by kernels, such as tridiagonal
  coefficients, diffusion operators, and small reference linear solvers.
- `kernels.py`: scalar single-axon kernels. These consume `SolverRuntime` and
  return raw `KernelResult` values.
- `batch_kernels.py`: batch kernels for homogeneous groups. These consume already
  assembled batched arrays and never decide which axons belong together.
- `crank_nicholson.py`: public optimized solver class.
- `experimental.py`: prototype/reference solver variants used by tests and
  benchmarks.
- `observables.py`: packaging helpers for membrane observables produced inside
  solver scans.

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
| `pcr_adaptive` | Uses `pcr_soa` for batches up to `B=1024`, and `pcr` above that. | Explicit reproduction of the current GPU `auto` policy. |

Example:

```python
from axonscope.solvers import BatchOptions

batch_options = BatchOptions.none(
    double_cable_block_solver="auto",
)
```

Forced choices are mainly diagnostic until benchmark evidence updates the
default policy. Planned names from roadmaps, such as split iterative,
associative, hybrid, or Pallas variants, should not appear in user-facing docs
until they are implemented, tested against Thomas, and wired into
`BatchOptions`.

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
