# Hotpath Diagnostic Workloads

This folder catalogs the Phase 2.5/7 workloads used to investigate CPU/GPU
bottlenecks, memory pressure, and retained-output costs before larger runtime
refactors.

These scripts are not the final benchmark framework. They are focused probes
for the new `axs.enable_benchmark(...)` instrumentation.

`axs.enable_benchmark(...)` currently supports only `level="hotpaths"`. Keep
that level until a real consumer needs separate `minimal` or `detailed` modes;
adding more labels before then would only create reporting drift.

## Registry

The canonical workload registry lives in:

- `benchmark/hotpaths/catalog.py`

The runnable entry point is:

- `benchmark/hotpaths/run.py`

List registered workloads:

```bash
python benchmark/hotpaths/run.py --list
```

Run a quick smoke trace:

```bash
python benchmark/hotpaths/run.py --workload all --preset smoke
```

Run the scale diagnostic requested for CPU/GPU investigation:

```bash
python benchmark/hotpaths/run.py --workload all --preset scale
```

Run the Phase 7 footprint/stimulus-only reuse probe:

```bash
python benchmark/hotpaths/run.py --workload footprint_reuse_sweep --preset scale --sweep-repeats 3
```

Run the Phase 7.6.1 solver-only/precomputed-input probe:

```bash
python benchmark/hotpaths/run.py --workload solver_only_precomputed --sizes 5 --warmups 1
```

For the exact double-cable linear solve itself, use the solver-focused runner
under `benchmark/solvers/`; it bypasses axon construction, dispatch, input
materialization, and result packaging.

Run the Phase 7.6.1 typed footprint/drive lowering probe:

```bash
python benchmark/hotpaths/run.py --workload typed_footprint_drive_matrix --sizes 5
```

Run the Phase 7.5 observer-only memory probe:

```bash
python benchmark/hotpaths/run.py --workload observer_only --preset scale
```

Run the Phase 7.6 realistic mixed-population probe:

```bash
python benchmark/hotpaths/run.py --workload realistic_mixed_population --preset scale
```

Run the Phase 7.6 compact coverage matrix:

```bash
python benchmark/hotpaths/run.py --workload hotpath_matrix --preset smoke
```

Run a cold-start diagnostic with JAX compile logging enabled:

```bash
python benchmark/hotpaths/run.py \
  --workload path_comparison_matrix \
  --sizes 1 \
  --jax-log-compiles \
  --prefix compile_log_probe
```

Run a longer trace-free kernel-scaling probe:

```bash
python benchmark/hotpaths/run.py \
  --workload observer_only \
  --sizes 500 1000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1
```

Run the same probe through chunked kernel time windows:

```bash
python benchmark/hotpaths/run.py \
  --workload observer_only \
  --sizes 500 1000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --time-chunk-steps 100
```

Run the double-cable long probe with automatic block-solver selection:

```bash
python benchmark/hotpaths/run.py \
  --workload double_cable_extracellular \
  --sizes 100 300 600 2000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --double-cable-block-solver auto
```

The automatic policy resolves to adaptive PCR on GPU and Thomas elsewhere, and
the manifest records both the requested and resolved solver choices.

Run the same double-cable probe with solver-side VmRaster output:

```bash
python benchmark/hotpaths/run.py \
  --workload double_cable_observer \
  --sizes 100 300 600 2000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --double-cable-block-solver auto
```

This keeps the same MRG extracellular setup but uses `Recording.none()` with
threshold-style observers instead of retaining a center Vm trace. The solver
returns packed `vm_raster` words; activation/latency-style summaries are
post-processing.

Capture a JAX profiler timeline for the same GPU-oriented path:

```bash
python benchmark/hotpaths/run.py \
  --workload double_cable_observer \
  --sizes 600 2000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --double-cable-block-solver auto \
  --jax-trace \
  --jax-trace-create-perfetto
```

Profiler output is written under
`benchmark/results/hotpaths/<run>/jax_traces/<workload>_n<size>/` and linked
from each manifest run entry as `jax_trace.trace_dir`. By default the trace
captures only `kernel.enqueue`, which keeps large-batch GPU timelines from
being flooded by Python dispatch events. Use `--jax-trace-scope run` only when
the dispatch/preparation timeline itself is the target.

Timing notes:

- `kernel.enqueue` measures Python-side submission of a JAX computation. On GPU,
  this can return before device work has finished.
- `kernel.wait` measures the explicit device synchronization. Use enqueue and
  wait together when comparing GPU runs.
- Warm runs use `--warmups` to separate first-call compilation/preparation from
  repeated execution. Treat any run without warmups as cold-start evidence.
  Manifest parameters, per-workload run records, and benchmark metadata include
  `timing_signature.label`: `cold_first_call` for `--warmups 0` and
  `warm_post_warmup` otherwise.
- Correctness validation and scientific acceptance are separate from these
  hotpath timings; performance probes should not replace unit, NRV, or
  physiology validation.

Force a double-cable block-solver choice for diagnostics:

```bash
python benchmark/hotpaths/run.py \
  --workload double_cable_extracellular \
  --sizes 100 300 600 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --double-cable-block-solver pcr
```

Use forced PCR-family choices as GPU-oriented diagnostics only for now. Normal
runs should start with `auto`, which keeps Thomas on CPU/default backends and
uses the adaptive PCR policy on GPU-like backends. The forced `pcr` Colab run
improved double-cable GPU totals strongly, but regressed CPU totals compared
with the default Thomas scan. `pcr_soa` forces the struct-of-arrays variant;
`pcr_adaptive` selects SoA for small/medium batches and matrix-layout PCR for
larger batches.

Run a longer realistic heterogeneous probe:

```bash
python benchmark/hotpaths/run.py \
  --workload realistic_mixed_population \
  --sizes 500 \
  --duration 5.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1
```

Outputs are written under `benchmark/results/hotpaths/`, which is intentionally
ignored by git.

## Colab CPU/GPU Runs

Local GPU execution is not assumed. For now, run GPU traces manually in Google
Colab. Publish the committed local revision with:

```bash
make bench-colab-push
```

Then open the Colab notebook below. The generated results are written under one
parent folder in the Colab checkout:

```text
benchmark/results/hotpaths/colab_cpu_gpu_<case>_YYYYMMDD_HHMMSS/
    gpu/
    cpu/
    comparison_summary.csv
```

The notebook zips that parent folder and downloads it directly through the
browser.

Colab runner and protocol:

- `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`
- `benchmark/hotpaths/COLAB.md`

The notebook has selectable cases. Use `setup_scale` for short preparation and
packaging regressions, then `kernel_observer_long` and
`kernel_realistic_long` when the goal is to see real CPU/GPU kernel scaling.

## Current Workloads

| Name | Purpose |
| --- | --- |
| `intracellular_only` | HH population with intracellular clamps only. Separates dispatch, runtime preparation, input materialization, kernel enqueue/wait, and result packaging without extracellular-field construction. |
| `point_source_extracellular` | HH population driven by analytical point-source extracellular contexts. Stresses the current generic `Vstim` preprocessing path tracked in the benchmark/CPU-GPU section of `todo.md`. |
| `double_cable_extracellular` | MRG double-cable population driven by analytical point-source extracellular contexts. Stresses the priority myelinated extracellular path before Phase 8 study/reuse APIs. |
| `double_cable_observer` | Same MRG double-cable extracellular population with threshold-style observers, `Recording.none()`, and packed VmRaster output. Compares retained center traces against observer-only output. |
| `footprint_reuse_sweep` | Repeated point-source pool runs with fixed geometry and changing stimulus amplitude. Measures the current cost of missing footprint/stimulus-only reuse and gives Phase 7.5/8 a baseline. |
| `solver_only_precomputed` | Direct backend workload with runtime and inputs prepared before timing. Separates kernel throughput from dispatch planning and input materialization for single-cable intra/extra rows. |
| `typed_footprint_drive_matrix` | Direct backend workload comparing analytical-context lowering against typed `ExtracellularFootprint`/`ExtracellularDrive` lowering, then executing the typed-drive dense `Vstim` path. |
| `observer_only` | HH population with threshold-style observers, `Recording.none()`, empty `vm_shapes`, and `vm_raster` in the manifest. Verifies the no-retained-Vm path. |
| `realistic_mixed_population` | Mixed HH/Rattay-Aberham population with varied diameters, compartment counts, intracellular clamps, and some analytical extracellular rows. Stresses heterogeneous dispatch, preparation, fallback, and result packaging. |
| `hotpath_matrix` | Compact matrix for center/probes recording, observer-only retention, point-source extracellular input, and mixed-population execution. Useful as the Phase 7.6 coverage run before deeper CPU/GPU work. |
| `path_comparison_matrix` | Controlled matrix for Phase 7.6.1: single-cable intra center/probes/full/observer, single-cable point-source extra center/probes/full/observer, and MRG double-cable point-source extra center/full. Useful to compare path families before optimizing. |

Each run manifest records `simulation.estimate().to_dict()` so timing traces can
be interpreted alongside estimated retained Vm, dense `Vstim`, and factorized
footprint sizes. Multi-simulation workloads also record `memory_estimates`,
`simulation_labels`, and `workload_metadata` with model, formulation, diameter,
compartment, stimulation, recording, observer summaries, and explicit
comparison axes for controlled matrix rows. Runs launched with
`--jax-log-compiles` also record that compile logging was enabled, so cold-start
logs can be matched to the benchmark manifest.

## Adding A Workload

1. Add the workload name and description to `benchmark/hotpaths/catalog.py`.
2. Add the builder branch in `benchmark/hotpaths/run.py`.
3. Keep output under `benchmark/results/hotpaths/`.
4. Add or update a unit test that checks the workload is listed by `--list`.
