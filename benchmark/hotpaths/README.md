# Hotpath Diagnostic Workloads

This folder catalogs the Phase 2.5 workloads used to investigate CPU/GPU
bottlenecks before the larger planning/preparation refactor.

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

## Adding A Workload

1. Add the workload name and description to `benchmark/hotpaths/catalog.py`.
2. Add the builder branch in `benchmark/hotpaths/run.py`.
3. Keep output under `benchmark/results/hotpaths/`.
4. Add or update a unit test that checks the workload is listed by `--list`.
