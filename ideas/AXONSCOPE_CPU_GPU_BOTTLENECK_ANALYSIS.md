# AxonScope CPU vs GPU Benchmark: Why the Results Point Outside the Core Solver

## Executive summary

The measured CPU and GPU runtimes are almost identical for medium and large
batches:

| Fibers | CPU median | GPU median | CPU / GPU |
|---:|---:|---:|---:|
| 5 | 110.7 ms | 253.8 ms | 0.436× |
| 50 | 1,513.0 ms | 1,538.7 ms | 0.983× |
| 500 | 69,771.6 ms | 70,675.4 ms | 0.987× |

The GPU is slower for five fibers, which is expected when launch and setup
overheads dominate. The important observation is what happens afterward:

- at 50 fibers, CPU and GPU differ by only about **1.7%**;
- at 500 fibers, CPU and GPU differ by only about **1.3%**;
- from 50 to 500 fibers, runtime increases by approximately **46×** on both
  CPU and GPU, although the number of fibers increases by only 10×.

This does not look like a solver that is compute-bound on one backend. It looks
like a dominant cost shared by both execution paths. The most likely candidates
are preprocessing, large array construction, memory traffic, dispatch planning,
and result packaging.

A second possibility is that the intended GPU kernel is not actually receiving
most of the work. The device should still be verified explicitly, but the
repository contains several concrete preprocessing patterns that can plausibly
explain the measurements.

This document is a static code analysis, not a profiler trace. The hypotheses
below should be confirmed with stage-level timing and a JAX profile.

---

## 1. What the scaling tells us

### 1.1 CPU and GPU converge to the same runtime

For 50 fibers:

```text
CPU = 1512.98 ms
GPU = 1538.67 ms
```

For 500 fibers:

```text
CPU = 69771.61 ms
GPU = 70675.44 ms
```

At these sizes, the two curves are effectively the same.

If the dominant cost were a large, well-vectorized numerical kernel running on
the selected backend, one would normally expect the CPU and GPU curves to
separate as the batch gets larger. Here they do not.

That strongly suggests one of the following:

1. most of the measured time is spent before or after the accelerated kernel;
2. the computation is dominated by memory construction and movement;
3. the solver structure exposes little useful GPU parallelism;
4. the GPU path falls back to a generic or inefficient kernel;
5. repeated compilation or synchronization is hiding the expected speed-up.

### 1.2 The scaling is strongly superlinear

From 50 to 500 fibers:

```text
CPU growth = 69771.61 / 1512.98 ≈ 46.1×
GPU growth = 70675.44 / 1538.67 ≈ 45.9×
```

The workload size increases by only 10×.

The approximate empirical scaling exponent over that interval is:

```text
runtime ∝ N^1.66
```

for both CPU and GPU.

The per-fiber runtime also increases sharply:

| Fibers | CPU per fiber | GPU per fiber |
|---:|---:|---:|
| 5 | 22.1 ms | 50.8 ms |
| 50 | 30.3 ms | 30.8 ms |
| 500 | 139.5 ms | 141.4 ms |

A pure linear batch computation should not become more than four times more
expensive per fiber when going from 50 to 500 fibers. This pattern is consistent
with allocation pressure, memory bandwidth limits, Python-side work that scales
poorly, or a planning/grouping step with worse-than-linear behavior.

---

## 2. The strongest visible suspect: extracellular-field construction

The single-cable batch dispatch prepares the runtime, builds intracellular
current arrays, builds extracellular potential arrays, and only then invokes
the solver kernel.

Relevant call sequence:

```python
runtime = prepare_solver_runtime(...)

iinj_mid = build_intracellular_current_density_batch(...)

vstim_mid = build_vstim_midpoint_batch(...)

out = SingleCableVStimBatchKernel(...).run(
    intracellular_current_density_mid=iinj_mid,
    extracellular_potential_mid_mV=vstim_mid,
    ...
)
```

Source:

- [`dispatcher/execution.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/execution.py#L2017-L2095)

The extracellular builder is implemented as a Python list comprehension over
all fibers:

```python
vstim_rows = [
    _build_vstim_row(
        row,
        t,
        x_positions_row_m=x_rows[i],
        axon_y_um=float(y_rows[i]),
        axon_z_um=float(z_rows[i]),
        dtype_local=dtype,
    )
    for i, row in enumerate(rows)
]

return jnp.stack(vstim_rows, axis=0)
```

Source:

- [`dispatcher/runtime_batches.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/runtime_batches.py#L2039-L2127)

Each row then performs another Python loop over extracellular contexts,
compiles the context for that fiber position, and applies a temporal `vmap`:

```python
vstim = jnp.zeros((nt, nx), dtype=dtype_local)

for ctx in contexts:
    compiled = compile_extracellular_context(
        ctx,
        x_positions_row_m,
        dtype_local=dtype_local,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
    )
    vstim = vstim + jax.vmap(compiled)(t_ms)
```

Source:

- [`dispatcher/runtime_batches.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/runtime_batches.py#L2150-L2191)

### Why this is suspicious

For every fiber, the current implementation may perform:

- Python iteration;
- JAX array indexing;
- conversion of JAX scalar positions through `float(...)`;
- extracellular-context compilation/preparation;
- a separate temporal `vmap`;
- creation of a separate `(Nt, Nx)` array;
- stacking all rows into `(B, Nt, Nx)`.

The explicit conversions are especially concerning:

```python
float(y_rows[i])
float(z_rows[i])
```

If these values reside on a device, converting each scalar to a Python `float`
can force device-to-host synchronization. Even on CPU, it serializes a
vectorizable operation into Python scalar work.

This builder can therefore dominate both CPU and GPU runs before the solver
gets a chance to benefit from the GPU.

---

## 3. Very large time-space arrays are materialized

For the benchmark parameters:

```text
Nt = duration / dt
   = 20 ms / 0.01 ms
   = 2000 time steps

Nx = 51 compartments
```

The extracellular potential has shape:

```text
(B, Nt, Nx)
```

At 500 fibers and float32 precision:

```text
500 × 2000 × 51 × 4 bytes ≈ 204 MB
```

At 1000 fibers:

```text
1000 × 2000 × 51 × 4 bytes ≈ 408 MB
```

The intracellular current builder also explicitly constructs an array with
shape `Iinj[B, Nt, Nx]` using a Python loop followed by `jnp.stack`:

```python
return jnp.stack(
    [
        _pad_time_space_array(
            _build_intracellular_current_density_row(...),
            ...
        )
        for index, axon in enumerate(axons)
    ],
    axis=0,
)
```

Source:

- [`dispatcher/runtime_batches.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/runtime_batches.py#L1697-L1757)

The kernel then receives both full arrays:

```text
Vstim: (B, Nt, Nx)
Iinj:  (B, Nt, Nx)
```

At 500 fibers, these two arrays alone represent roughly **408 MB** at float32,
before counting outputs, membrane state, gates, solver temporaries, or the
diffusion forcing.

At 1000 fibers, they represent roughly **816 MB**.

This is a likely explanation for the sudden increase in per-fiber cost at
larger batches. The run may become dominated by:

- allocation;
- materialization;
- memory bandwidth;
- accelerator transfer;
- allocator fragmentation;
- garbage collection and deferred deallocation.

A GPU cannot provide a large speed-up when the dominant operation is repeatedly
constructing and moving hundreds of megabytes of input arrays.

---

## 4. The imposed field has a factorized structure that is currently expanded

For a fixed electrode and a shared stimulus waveform, the extracellular field
usually has the form:

```text
Vext[b, t, x] = current[t] × footprint[b, x]
```

The repository already contains code that expresses this structure:

```python
current_A = jax.vmap(compile_stimulus(stimulus, dtype_local=dtype))(t)

return (
    scale[:, None, None]
    * current_A[None, :, None]
    * footprint[:, None, :]
    * 1e3
)
```

Source:

- [`dispatcher/runtime_batches.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/runtime_batches.py#L2000-L2035)

However, this still expands the result to the full `(B, Nt, Nx)` tensor.

Because the spatial diffusion operator is linear:

```text
L(current[t] × footprint[b, x])
=
current[t] × L(footprint[b, x])
```

the solver could instead receive:

```text
current:            (Nt,)
forcing_footprint:  (B, Nx)
```

At each time step, it would compute only:

```python
vstim_force_t = current_t * forcing_footprint
```

For `B=1000`, `Nt=2000`, and `Nx=51`, this changes the input representation
from hundreds of megabytes to well below one megabyte at float32, excluding the
solver state.

This is likely the highest-impact architectural optimization.

---

## 5. `Recording.center("Vm")` currently misses the shared-cable fast path

The benchmark correctly uses:

```python
axs.Recording.center("Vm")
```

as documented in the advanced recording example:

- [`example_05_recording_options.py`](https://github.com/louisreg/AxonScope/blob/main/examples/advanced/example_05_recording_options.py#L653-L677)

However, the single-cable batch kernel selects its specialized shared-cable
fast path only when full voltage is recorded:

```python
if record_full and chunk_steps is None and shared_cable:
    out = _run_single_cable_vstim_batch_vm_scan(...)
else:
    out = _run_single_cable_vstim_batch_array_chunks(...)
```

Source:

- [`solvers/batch_kernels.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/solvers/batch_kernels.py#L3370-L3452)

Since center-only recording sets `record_full=False`, it enters the generic
array/chunk path even when all fibers share the same cable geometry.

The generic path converts shared cable arrays into batched representations and
uses the more general execution machinery:

- [`solvers/batch_kernels.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/solvers/batch_kernels.py#L3752-L3806)

This does not prove that the generic path is the dominant bottleneck, but it is
a concrete performance regression specific to the benchmark configuration.

A dedicated fast path should preserve shared cable coefficients while emitting
only the requested center value at each time step.

Conceptually:

```python
if center_only and chunk_steps is None and shared_cable:
    out = _run_single_cable_vstim_batch_center_scan(...)
```

---

## 6. The solver itself may not be a good GPU workload

Inside the solver, each time step contains a tridiagonal solve:

```python
Vm_new = jax.lax.linalg.tridiagonal_solve(
    dl_row,
    d,
    du_row,
    rhs[:, None],
)[:, 0]
```

Source:

- [`solvers/batch_kernels.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/solvers/batch_kernels.py#L2653-L2679)

For this benchmark, each system is only 51 compartments wide.

The workload therefore combines:

- a sequential dependency over 2000 time steps;
- small tridiagonal systems of width 51;
- parallelism mainly across fibers.

Small tridiagonal solves are not automatically ideal for a GPU. The GPU needs a
large, explicit batch dimension and enough work per launch to amortize launch,
memory, and synchronization costs.

This can explain why the GPU is slower for five fibers. It does not fully
explain why CPU and GPU remain almost exactly equal at 500 fibers; the
preprocessing and memory behavior remain stronger suspects for that result.

---

## 7. The benchmark measures the complete dispatch path

`simulate_pool()` calls `run_pool()`, and `run_pool()` rebuilds the dispatch plan
for every invocation:

```python
plan = build_dispatch_plan(axons)
```

It then iterates over dispatch groups, prepares each group, executes it, and
reorders the results:

- [`dispatcher/execution.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/execution.py#L1768-L1861)

The batch group also calls `prepare_solver_runtime(...)` on each run:

- [`dispatcher/execution.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/execution.py#L2017-L2039)

Finally, the batched output is split into one Python `DispatchResult` object per
fiber:

```python
for row_index, item in enumerate(group.items):
    row_vm = Vm[row_index]
    ...
    results.append(DispatchResult(...))
```

Source:

- [`dispatcher/execution.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/execution.py#L2744-L2827)

Therefore, the measured time is approximately:

```text
dispatch planning
+ runtime preparation
+ intracellular input construction
+ extracellular field construction
+ device placement / transfer
+ solver kernel
+ synchronization
+ per-fiber result packaging
```

The benchmark is useful as an end-to-end API benchmark, but it is not a
solver-only benchmark.

The current results say that the **end-to-end pipeline** is backend-insensitive.
They do not yet show that the numerical solver kernel itself is equally fast on
CPU and GPU.

---

## 8. The numerical differences are not the performance explanation

The RMS difference stays around:

```text
0.0022 mV
```

for 50 and 500 fibers, while the maximum absolute difference increases to:

```text
0.1766 mV
```

at 500 fibers.

The nearly constant RMS error suggests that most samples remain very close.
The increasing maximum may come from a small number of samples near a sharp
action-potential transition, where small floating-point differences can move a
peak or threshold crossing slightly in time.

This deserves a separate correctness investigation, but it does not explain the
nearly identical runtimes.

Useful additional diagnostics would be:

```text
maximum difference by fiber
maximum difference by time
difference around spike threshold crossings
peak-time difference
CPU/GPU dtype and x64 configuration
```

---

## 9. Experiments that can localize the bottleneck quickly

### Experiment A — remove extracellular stimulation

Run the same pool with no extracellular context.

Interpretation:

- if the GPU becomes substantially faster, `Vstim` construction or transfer is
  the main bottleneck;
- if CPU and GPU remain equal, inspect dispatch, solver structure, and result
  packaging.

### Experiment B — time preprocessing and kernel separately

Add timers around:

```text
1. build_dispatch_plan
2. prepare_solver_runtime
3. build_intracellular_current_density_batch
4. build_vstim_midpoint_batch
5. kernel.run + block_until_ready
6. _dispatch_results_from_batch
```

The essential requirement is to synchronize only the stage being measured:

```python
start = perf_counter()
out = kernel.run(...)
out.Vm.block_until_ready()
kernel_seconds = perf_counter() - start
```

Without `block_until_ready()`, asynchronous GPU execution can make the kernel
appear artificially fast.

### Experiment C — benchmark full recording against center recording

Compare:

```python
axs.Recording.voltage()
```

and:

```python
axs.Recording.center("Vm")
```

on an otherwise identical shared-geometry pool.

If full recording is unexpectedly faster, it confirms that the
`record_full/shared_cable` branch selection is important.

### Experiment D — precompute `Vstim` and `Iinj`

Build `vstim_mid` and `iinj_mid` once, then repeatedly invoke only the batch
kernel.

This separates solver throughput from preprocessing.

### Experiment E — replace the analytical field with a precomputed footprint

Precompute one spatial footprint per fiber and reuse one temporal waveform.

If performance improves dramatically, the per-fiber analytical context
construction is the dominant cost.

### Experiment F — verify device placement

The GPU worker should print:

```python
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())
print("Selected device:", device)
print("Vstim device:", vstim_mid.device)
print("Iinj device:", iinj_mid.device)
print("Output device:", out.Vm.device)
```

This rules out an accidental CPU execution path or unexpected placement.

### Experiment G — detect recompilation

Run with:

```bash
JAX_LOG_COMPILES=1
```

or:

```python
jax.config.update("jax_log_compiles", True)
```

There should not be a new expensive compilation during every timed repetition
for an unchanged shape and static configuration.

---

## 10. Recommended optimization order

### Priority 1 — vectorize extracellular preprocessing

Replace the Python fiber loop and scalar `float(...)` conversions with one
batched JAX computation.

Target shape:

```text
positions:   (B, Nx, 3)
electrodes:  (E, 3)
waveform:    (Nt, E)
output:      (B, Nt, Nx)
```

Or, preferably, avoid creating that output and keep the field factorized.

### Priority 2 — keep the extracellular field factorized

Pass:

```text
waveform[Nt]
forcing_footprint[B, Nx]
```

to the solver instead of `Vstim[B, Nt, Nx]`.

### Priority 3 — special-case absent intracellular stimulation

When no intracellular contexts are attached, avoid constructing a full zero
array of shape `(B, Nt, Nx)`.

The solver can accept:

```text
Iinj = None
```

or a scalar zero and specialize that branch.

### Priority 4 — add a shared-cable center-recording fast path

Reuse shared cable coefficients and emit only the center trace.

### Priority 5 — cache prepared pools

Introduce a reusable object such as:

```python
prepared = axs.prepare_pool(
    simulations,
    duration_ms=duration,
    dt_ms=dt,
    recording=recording,
)

results = prepared.run()
```

It could cache:

- the dispatch plan;
- compiled axon representations;
- runtime structures;
- cable coefficients;
- recording indices;
- extracellular footprints;
- compiled kernels.

### Priority 6 — return a batched result object

Avoid creating hundreds or thousands of Python result objects unless requested.

For example:

```python
pool_result.Vm       # (B, Nt, Nrecorded)
pool_result.result(i)
```

The per-fiber objects could be materialized lazily.

---

## 11. Bottom line

The benchmark results are a strong warning that the current end-to-end path is
not exposing GPU throughput.

The key evidence is not merely that the GPU is slow. It is that CPU and GPU:

- become almost identical by 50 fibers;
- remain almost identical at 500 fibers;
- exhibit the same severe superlinear scaling.

That pattern is much more consistent with a shared preprocessing or
memory-management bottleneck than with the raw membrane solver alone.

The most suspicious code path is:

```text
per-fiber Python construction of Vstim
→ full B × Nt × Nx input materialization
→ generic center-recording kernel path
→ per-fiber result packaging
```

The next step should not be to increase `N` further. It should be to instrument
the pipeline and isolate:

```text
preprocessing time
kernel time
postprocessing time
```

My strongest prediction is that `build_vstim_midpoint_batch()` and the
materialization of the full extracellular tensor account for a large fraction
of the 50- and 500-fiber runtimes. The fastest proof is to rerun the benchmark
without extracellular stimulation, then with precomputed footprints.

---

## Source files reviewed

- [`src/axonscope/dispatcher/execution.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/execution.py)
- [`src/axonscope/dispatcher/runtime_batches.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/dispatcher/runtime_batches.py)
- [`src/axonscope/solvers/batch_kernels.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/solvers/batch_kernels.py)
- [`src/axonscope/simulation.py`](https://github.com/louisreg/AxonScope/blob/main/src/axonscope/simulation.py)
- [`examples/advanced/example_05_recording_options.py`](https://github.com/louisreg/AxonScope/blob/main/examples/advanced/example_05_recording_options.py)
- [`examples/benchmarks/benchmark_001_simple_batching.py`](https://github.com/louisreg/AxonScope/blob/main/examples/benchmarks/benchmark_001_simple_batching.py)
