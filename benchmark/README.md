# AxonScope Benchmarks

This directory is for reproducible performance evidence, validation-only solver
checks, and archived solver experiments. Public examples and tutorials must not
depend on benchmark internals.

## Surface Map

The lifecycle registry lives in `benchmark/registry.py`.

| Surface | Status | Use |
| --- | --- | --- |
| `benchmark/runtime/` | active | Named runtime suites for supported public execution paths. |
| `benchmark/hotpaths/` | active | Cold/warm, lowering, memory, and dispatch diagnostics. |
| `benchmark/nrv_performance/` | active | AxonScope-vs-NRV and realistic fascicle performance suites. |
| `benchmark/realistic_examples/` | active | Public workflow-level benchmarks. |
| `benchmark/kaggle/` | active | Remote GPU wrapper for active suites. |
| `benchmark/solvers/` | validation-only | Retained double-cable solver timing and agreement checks. |
| `benchmark/pseudo_double/` | experimental | Standby pseudo-double evidence, not a public route. |
| `benchmark/archived_solver_spikes/` | archive | Historical solver candidates. |
| `benchmark/triton_solver/` | archive | Historical Triton candidate. |
| `benchmark/jax_triton_solver/` | archive | Historical JAX-Triton candidate. |
| `benchmark/cuda_ffi_solver/` | archive | Historical CUDA FFI candidate. |
| `benchmark/cute_dsl/` | archive | Historical CuTe DSL candidate. |
| `benchmark/notebooks/` | archive | Notebook snapshots only. |
| `benchmark/reports/` | generated-output | Generated reports and figures. |
| `benchmark/results/` | generated-output | Raw benchmark outputs. |

`benchmark/results/` and `benchmark/reports/` are ignored by git. Do not make
architecture decisions by editing files there; summarize retained evidence in
tracked docs or TODO entries.

## Active Commands

List runtime suites:

```bash
python benchmark/runtime/run.py --list
```

Run the fast runtime smoke:

```bash
python benchmark/runtime/run.py --suite smoke
```

Run runtime memory scenarios:

```bash
python benchmark/runtime/run.py --suite pool_memory
```

Run the post-P7 membrane model source/codegen cache benchmark:

```bash
python benchmark/runtime/run.py --suite model_codegen
```

Run built-in plus custom membrane model codegen cases:

```bash
python benchmark/runtime/run.py --suite model_codegen_all
```

Use `model_codegen` when the question is source compilation, generated-code
cache behavior, source hash stability, or class-based membrane model overhead.
Use the other runtime/hotpath/NRV suites when the question includes cable
solving, dispatch, input lowering, result assembly, external comparison, or
GPU behavior.

List NRV/performance suites:

```bash
python benchmark/nrv_performance/run.py --list
```

Run a dry NRV smoke expansion:

```bash
python benchmark/nrv_performance/run.py --suite smoke --dry-run
```

Run the synthetic GPU population cold/warm suite locally or through Kaggle:

```bash
python benchmark/nrv_performance/run.py --suite population_tsim_gpu
```

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark population_tsim_gpu \
  --machine-shape NvidiaTeslaP100
```

List hotpath workloads:

```bash
python benchmark/hotpaths/run.py --list
```

Run a cold-path profile before making first-call performance claims:

```bash
python benchmark/hotpaths/run.py \
  --workload path_comparison_matrix \
  --sizes 1 \
  --jax-log-compiles \
  --prefix cold_path_probe
```

Run a small time+memory map before optimizing a stage:

```bash
python benchmark/hotpaths/run.py \
  --workload hotpath_matrix \
  --preset smoke \
  --memory-trace all \
  --memory-top-n 10 \
  --jax-device-memory-profile \
  --prefix memory_map_smoke
```

This writes per-stage timing to `events.jsonl`/`summary.csv`, measured memory
metadata into each event, aggregate memory rows to `memory_summary.csv`, and
JAX device-memory `.prof` artifacts for selected stages under
`device_memory_profiles/`. `memory_summary.csv` keeps estimated tensor bytes
next to measured RSS/device deltas and adds `memory_estimate_gap_note` when a
large measured delta exceeds the retained tensor estimate.

## Validation-Only Solver Commands

Use these for retained double-cable solver evidence, not as user-facing runtime
APIs:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run
python benchmark/solvers/bench_double_cable_end_to_end.py --dry-run
python benchmark/solvers/validate_double_cable_solver_agreement.py --dry-run
```

Only make solver-route claims after agreement tests and benchmark rows point to
the same conclusion.

## Metadata Policy

Benchmark JSON metadata should include:

- OS, Python, CPU core counts, host RAM, and git state;
- package versions for NumPy, SciPy, JAX/JAXlib, AxonScope, and related tools;
- JAX backend, devices, `jax_enable_x64`, process count, and device memory
  stats when exposed by the backend;
- GPU/VRAM information from `nvidia-smi` when available;
- key runtime environment variables, especially `XLA_FLAGS`,
  `JAX_PLATFORM_NAME`, `JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`,
  `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, and
  `XLA_PYTHON_CLIENT_ALLOCATOR`.

Hotpath trace `metadata.json` files include the effective compute backend
(`cpu`, `gpu`, `tpu`, or `unknown`), detected CPU/GPU model names when exposed
by the host, OS details, host RAM, package versions, JAX devices, and the
runtime environment variables listed above.

By default, JAX may preallocate a large fraction of GPU memory, commonly 75% on
the standard GPU allocator. A benchmark report must record any allocator or
preallocation environment override before comparing VRAM or memory pressure.

## Memory Policy

Current memory evidence layers:

- host RSS and process peak RSS for workflow-level benchmarks;
- opt-in per-span RSS, `tracemalloc`, JAX device `memory_stats()`, and
  `nvidia-smi` snapshots in hotpath traces through `--memory-trace`;
- optional JAX device memory `.prof` artifacts through
  `--jax-device-memory-profile` after synchronization-oriented stages such as
  `kernel.wait`;
- planned tensor estimates from `AxonSimulation.estimate()` and hotpath
  metadata;
- JAX array shape/dtype/device metadata for kernel inputs and outputs;
- best-effort JAX device `memory_stats()` and `nvidia-smi` VRAM snapshots when
  available.

Future memory work may add `memray`, a dedicated sampling profiler, or `jax-smi`
integration. Do not treat estimated tensor bytes as measured peak memory; use
`memory_summary.csv` to compare estimates against measured RSS/tracemalloc/device
signals.

## Claims Policy

Before making a performance claim:

- run the smallest smoke suite first;
- collect cold and warm timings separately;
- include `runtime.prepare`, `inputs.intracellular`, `inputs.extracellular`,
  `kernel.enqueue`, `kernel.wait`, cache hits/misses, and result assembly when
  discussing first-call performance;
- re-run hotpath, realistic, or Kaggle suites only when the claim depends on
  that evidence.
