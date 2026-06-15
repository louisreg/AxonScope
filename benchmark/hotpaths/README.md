# Hotpath Diagnostic Workloads

This folder catalogs the Phase 2.5/7 workloads used to investigate CPU/GPU
bottlenecks, memory pressure, and retained-output costs before larger runtime
refactors.

These scripts are not the final benchmark framework. They are focused probes
for the new `axs.enable_benchmark(...)` instrumentation.

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
| `footprint_reuse_sweep` | Repeated point-source pool runs with fixed geometry and changing stimulus amplitude. Measures the current cost of missing footprint/stimulus-only reuse and gives Phase 7.5/8 a baseline. |
| `solver_only_precomputed` | Direct backend workload with runtime and inputs prepared before timing. Separates kernel throughput from dispatch planning and input materialization for single-cable intra/extra rows. |
| `typed_footprint_drive_matrix` | Direct backend workload comparing analytical-context lowering against typed `ExtracellularFootprint`/`ExtracellularDrive` lowering, then executing the typed-drive dense `Vstim` path. |
| `observer_only` | HH population with solver-side peak-voltage and activation observers, `Recording.none()`, empty `vm_shapes`, and compact observation names in the manifest. Verifies the Phase 7.5 no-retained-Vm path. |
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
