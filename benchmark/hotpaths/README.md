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

Outputs are written under `benchmark/results/hotpaths/`, which is intentionally
ignored by git.

## GPU Runs

Local GPU execution is not assumed. For now, run GPU traces manually in Google
Colab. Publish the committed local revision with:

```bash
make bench-colab-push
```

Then open the Colab notebook below. The generated results are written under
`benchmark/results/hotpaths/<prefix>/` in the Colab checkout, zipped, and
downloaded directly through the browser.

Colab runner and protocol:

- `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`
- `benchmark/hotpaths/COLAB.md`

## Current Workloads

| Name | Purpose |
| --- | --- |
| `intracellular_only` | HH population with intracellular clamps only. Separates dispatch, runtime preparation, input materialization, kernel enqueue/wait, and result packaging without extracellular-field construction. |
| `point_source_extracellular` | HH population driven by analytical point-source extracellular contexts. Stresses the current generic `Vstim` preprocessing path highlighted in `ideas/AXONSCOPE_CPU_GPU_BOTTLENECK_ANALYSIS.md`. |
| `footprint_reuse_sweep` | Repeated point-source pool runs with fixed geometry and changing stimulus amplitude. Measures the current cost of missing footprint/stimulus-only reuse and gives Phase 7.5/8 a baseline. |
| `observer_only` | HH population with solver-side peak-voltage and activation observers, `Recording.none()`, empty `vm_shapes`, and compact observation names in the manifest. Verifies the Phase 7.5 no-retained-Vm path. |
| `realistic_mixed_population` | Mixed HH/Rattay-Aberham population with varied diameters, compartment counts, intracellular clamps, and some analytical extracellular rows. Stresses heterogeneous dispatch, preparation, fallback, and result packaging. |
| `hotpath_matrix` | Compact matrix for center/probes recording, observer-only retention, point-source extracellular input, and mixed-population execution. Useful as the Phase 7.6 coverage run before deeper CPU/GPU work. |

Each run manifest records `simulation.estimate().to_dict()` so timing traces can
be interpreted alongside estimated retained Vm, dense `Vstim`, and factorized
footprint sizes. Multi-simulation workloads also record `memory_estimates`,
`simulation_labels`, and `workload_metadata` with model, formulation, diameter,
compartment, stimulation, recording, and observer summaries.

## Adding A Workload

1. Add the workload name and description to `benchmark/hotpaths/catalog.py`.
2. Add the builder branch in `benchmark/hotpaths/run.py`.
3. Keep output under `benchmark/results/hotpaths/`.
4. Add or update a unit test that checks the workload is listed by `--list`.
