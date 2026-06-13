# Solver Organization

The solver package owns numerical execution. It receives descriptive axons,
compiles them into runtime arrays, and advances state in time. It should not
own public model construction, pool grouping, electrode placement policy, or
result analysis.

## Files

- `options.py`: solver-owned execution knobs. `SolverOptions` controls runtime
  preparation, currently rate tables. `BatchOptions` and `BatchRecording`
  control batch-kernel memory and retained Vm columns.
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

## Time Grid

Current kernels use a fixed time step for every integration step. Therefore
`duration_ms` must be an integer multiple of `dt_ms`; otherwise the runtime
raises `ValueError` instead of silently rounding up and simulating past the
requested final time.

The recorded time vector contains post-step samples:

```text
dt_ms, 2*dt_ms, ..., duration_ms
```

Midpoint stimulation samples are evaluated at:

```text
0.5*dt_ms, 1.5*dt_ms, ..., duration_ms - 0.5*dt_ms
```
