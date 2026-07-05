# AxonScope Benchmarks

This directory is for reproducible performance evidence, validation-only solver
checks, and archived solver experiments. Public examples and tutorials must not
depend on benchmark internals.

## Surface Map

The lifecycle registry lives in `benchmark/registry.py`.

| Surface | Status | Command classes | Use |
| --- | --- | --- | --- |
| `benchmark/runtime/` | active | public-runtime, hotpath-diagnostic, model-codegen, validation-only | Named runtime suites for supported public execution paths. |
| `benchmark/hotpaths/` | active | hotpath-diagnostic | Cold/warm, lowering, memory, and dispatch diagnostics. |
| `benchmark/nrv_performance/` | active | external-comparison, hotpath-diagnostic, public-runtime | AxonScope-vs-NRV and realistic fascicle performance suites. |
| `benchmark/realistic_examples/` | active | public-runtime | Public workflow-level benchmarks. |
| `benchmark/kaggle/` | active | remote-GPU, generated-output | Remote GPU wrapper for active suites. |
| `benchmark/solvers/` | validation-only | validation-only | Retained double-cable solver timing and agreement checks. |
| `benchmark/pseudo_double/` | experimental | validation-only | Standby pseudo-double evidence, not a public route. |
| `benchmark/archived_solver_spikes/` | archive | archive | Historical solver candidates. |
| `benchmark/triton_solver/` | archive | archive | Historical Triton candidate. |
| `benchmark/jax_triton_solver/` | archive | archive | Historical JAX-Triton candidate. |
| `benchmark/cuda_ffi_solver/` | archive | archive | Historical CUDA FFI candidate. |
| `benchmark/cute_dsl/` | archive | archive | Historical CuTe DSL candidate. |
| `benchmark/notebooks/` | archive | archive | Notebook snapshots only. |
| `benchmark/reports/` | generated-output | generated-output | Generated reports and figures; some retained summaries are tracked. |
| `benchmark/results/` | generated-output | generated-output | Raw benchmark outputs. |

`benchmark/results/` is ignored by git and must never be used as architecture
source material. New generated `benchmark/reports/` files are also ignored, but
tracked report summaries may be cited only after a fresh review. For new claims,
summarize retained evidence in tracked docs, changelog notes, or TODO entries
instead of editing raw output directories.

## Command Classes

Use these labels consistently in docs and reports:

- `public-runtime`: supported AxonScope runtime or public workflow timing.
- `hotpath-diagnostic`: stage-level dispatch, lowering, memory, or kernel probe.
- `model-codegen`: class-based membrane source/codegen/cache timing.
- `validation-only`: solver/scientific agreement evidence, not a public runtime API.
- `external-comparison`: AxonScope-vs-NRV or NRV/FEM handoff timing.
- `remote-GPU`: Kaggle/Colab wrapper for GPU evidence outside the local machine.
- `archive`: retained historical code or notebooks, never fresh claim material.
- `generated-output`: generated metadata, reports, figures, or raw output files.

## Retained Commands

The retained command list is mirrored in `benchmark/registry.py`.

| Command | Class | Use |
| --- | --- | --- |
| `python benchmark/runtime/run.py --suite smoke` | public-runtime | Fast supported-runtime smoke before broader timing runs. |
| `python benchmark/runtime/run.py --suite full` | public-runtime | Default supported runtime matrix with warm repeats. |
| `python benchmark/runtime/run.py --suite profiled` | hotpath-diagnostic | Runtime matrix with a JAX profiler output directory. |
| `python benchmark/runtime/run.py --suite vstim_forcing` | public-runtime | Supported single-cable imposed-Vstim path comparison. |
| `python benchmark/runtime/run.py --suite vstim_batch` | hotpath-diagnostic | Batch-kernel diagnostic for imposed-Vstim input paths. |
| `python benchmark/runtime/run.py --suite double_cable_batch` | hotpath-diagnostic | Batch-kernel diagnostic for double-cable runtime paths. |
| `python benchmark/runtime/run.py --suite pool_memory` | hotpath-diagnostic | Pool memory/runtime probe for retained-output policies. |
| `python benchmark/runtime/run.py --suite model_codegen` | model-codegen | Built-in source/codegen cache and model-step smoke timing. |
| `python benchmark/runtime/run.py --suite model_codegen_simulations` | model-codegen | Tiny public AxonSimulation first/warm timings for class-based templates. |
| `python benchmark/runtime/run.py --suite model_codegen_all` | model-codegen | Built-in plus custom codegen, model-step, and template simulation timing. |
| `python benchmark/runtime/run.py --suite reference_solvers` | validation-only | Focused optimized-vs-dense HH solver comparison. |
| `python benchmark/hotpaths/run.py --workload hotpath_matrix --preset smoke` | hotpath-diagnostic | Compact stage coverage before deeper CPU/GPU profiling. |
| `python benchmark/hotpaths/run.py --workload cold_run_micro --sizes 1 --duration 1.0 --dt 0.02 --warmups 0 --memory-trace rss --prefix cold_run_micro` | hotpath-diagnostic | Short local P9 cold-run baseline covering retained Vm, VmRaster observer-only, and point-source extracellular paths. |
| `python benchmark/hotpaths/run.py --workload path_comparison_matrix --sizes 1 --jax-log-compiles --prefix cold_path_probe` | hotpath-diagnostic | Cold-start and compile-log evidence for first-call claims. |
| `python benchmark/hotpaths/run.py --workload hotpath_matrix --preset smoke --memory-trace all --memory-top-n 10 --jax-device-memory-profile --prefix memory_map_smoke` | hotpath-diagnostic | Per-stage time+memory map for optimization targeting. |
| `python benchmark/hotpaths/run.py --workload double_cable_observer --sizes 100 300 600 2000 --duration 10.0 --dt 0.01 --compartments 51 --warmups 1 --double-cable-block-solver auto` | hotpath-diagnostic | MRG double-cable VmRaster compact-output scaling probe. |
| `python benchmark/nrv_performance/run.py --suite smoke --dry-run` | external-comparison | Expand the smallest AxonScope-vs-NRV performance grid. |
| `python benchmark/nrv_performance/run.py --suite full` | external-comparison | Full HH/MRG AxonScope-vs-NRV performance grid. |
| `python benchmark/nrv_performance/run.py --suite mrg_extracellular_perf` | external-comparison | Focused MRG extracellular warm-runtime comparison. |
| `python benchmark/nrv_performance/run.py --suite population_cold_path_smoke` | hotpath-diagnostic | AxonScope-only cold/warm point-source timing with hotpath reports. |
| `python benchmark/nrv_performance/run.py --suite population_tsim` | external-comparison | Point-source population AxonScope-vs-NRV timing. |
| `python benchmark/nrv_performance/run.py --suite population_tsim_gpu` | public-runtime | Synthetic AxonScope population timing with explicit GPU execution policy. |
| `python benchmark/nrv_performance/run.py --suite population_tsim_gpu_1000` | public-runtime | Large synthetic AxonScope GPU population timing. |
| `python benchmark/nrv_performance/run.py --suite realistic_fascicle_smoke` | external-comparison | Small NRV LIFE/FEM handoff and AxonScope recruitment profile. |
| `python benchmark/nrv_performance/run.py --suite realistic_fascicle_synthetic_full` | external-comparison | Full-size synthetic NRV LIFE/FEM handoff profile. |
| `python benchmark/realistic_examples/bench_basic_examples.py --preset smoke --repeats 1` | public-runtime | Workflow-level public-example smoke timing. |
| `python benchmark/realistic_examples/bench_basic_examples.py --preset stress --platforms cpu gpu --profile` | public-runtime | CPU/GPU public-workflow stress pass with hotpath profiles. |
| `python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run` | validation-only | Retained double-cable linear solver timing matrix preview. |
| `python benchmark/solvers/bench_double_cable_end_to_end.py --dry-run` | validation-only | Retained double-cable end-to-end timing matrix preview. |
| `python benchmark/solvers/validate_double_cable_solver_agreement.py --dry-run` | validation-only | Agreement harness for retained solver-route changes. |
| `python benchmark/solvers/profile_double_cable_linear_solvers.py --help` | validation-only | Focused trace helper for retained linear-solver diagnostics. |
| `python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --benchmark population_tsim_gpu --machine-shape NvidiaTeslaP100` | remote-GPU | Run the synthetic population GPU validation preset remotely. |
| `python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --benchmark realistic_fascicle_nrv_gpu --machine-shape NvidiaTeslaP100` | remote-GPU | Run the NRV LIFE/FEM handoff smoke on a remote GPU. |
| `python benchmark/kaggle/prepare_kernel_metadata.py --username YOUR_KAGGLE_USERNAME --benchmark smoke` | generated-output | Generate Kaggle metadata/config files before remote submission. |

Archive and generated-output surfaces are classed in the surface map rather than
promoted as retained runnable commands.

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

Run the post-P7 membrane model source/codegen and model-step smoke benchmark:

```bash
python benchmark/runtime/run.py --suite model_codegen
```

Run public template simulations for the new class-based membrane families:

```bash
python benchmark/runtime/run.py --suite model_codegen_simulations
```

Run built-in plus custom membrane model codegen, model-step, and simulation cases:

```bash
python benchmark/runtime/run.py --suite model_codegen_all
```

Use `model_codegen` when the question is source compilation, generated-code
cache behavior, source hash stability, generated NumPy/JAX model-step overhead,
or the built-in class-based membrane smoke path. Use
`model_codegen_simulations` when the question is first/warm public
`AxonSimulation` timing for HH, Rattay-Aberham, Sundt, Tigerholm,
Schild94/Schild97, or MRG/AxNode templates. Use the other runtime/hotpath/NRV
suites when the question includes broader cable solving, dispatch, input
lowering, result assembly, external comparison, or GPU behavior.

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
  --workload cold_run_micro \
  --sizes 1 \
  --duration 1.0 \
  --dt 0.02 \
  --warmups 0 \
  --memory-trace rss \
  --prefix cold_run_micro
```

Run the broader compile-log matrix only after the short baseline points to a
first-call issue worth expanding:

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

Model-codegen benchmark JSON files contain separate `codegen_rows`,
`model_step_rows`, `simulation_rows`, and `correctness_rows` arrays. The runner
also writes sibling CSV files with `_codegen`, `_model_steps`, `_simulations`,
and `_correctness` suffixes. Treat timings as usable only when the relevant
correctness rows are `ok`; generated JAX is checked against generated NumPy,
and mappable generated outputs are checked against the NumPy interpreter.

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
