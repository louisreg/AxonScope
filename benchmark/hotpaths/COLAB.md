# Colab CPU/GPU Hotpath Protocol

Use this when local GPU execution is not available. The local machine publishes
one committed AxonScope revision to the moving `bench-colab` branch, and Colab
always clones that branch before running matching hotpath workloads on GPU and
on a forced CPU JAX backend.

The ready-to-run notebook lives at:

- `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`

## 1. Start A GPU Runtime

In Google Colab:

1. Open a new notebook.
2. Select `Runtime > Change runtime type`.
3. Select a GPU accelerator.
4. Restart the runtime after dependency installation if Colab asks for it.

## 2. Publish The Local Revision

From the local repository, commit the exact code to test and push it to the
dedicated Colab branch:

```bash
git add -A
git commit -m "Benchmark Colab run"
make bench-colab-push
```

`make bench-colab-push` refuses dirty working trees because Colab can only see
committed code. It updates the remote `bench-colab` branch without switching
your current local branch.

Override the remote or branch only if needed:

```bash
make bench-colab-push GIT_REMOTE=origin BENCH_COLAB_BRANCH=bench-colab
```

## 3. Run The Notebook

Open `benchmark/hotpaths/colab_gpu_hotpaths.ipynb` in Colab, replace
`REPO_URL` once, choose `CASE`, then run the notebook cell.

Available notebook cases:

- `cpu_double_cable_extracellular_heavy`: CPU-only stress trace for the
  priority MRG double-cable extracellular path while GPU access is unavailable.
  It runs `double_cable_extracellular` at sizes `100/300/600`, `duration=10 ms`,
  `dt=0.01 ms`, and a target of `51` compartments.
- `setup_scale`: short all-workload trace. This keeps `duration=0.30 ms`,
  `dt=0.05 ms`, and small `Nx`; it is best for catching preparation,
  dispatch, transfer, and result-packaging regressions.
- `kernel_observer_long`: longer `observer_only` trace with `Recording.none()`,
  `duration=10 ms`, `dt=0.01 ms`, and `51` compartments. This is the preferred
  first GPU-scaling probe because it minimizes retained output while the solver
  loop does real work.
- `kernel_realistic_long`: longer `realistic_mixed_population` trace with
  `duration=5 ms`, `dt=0.01 ms`, and `51` compartments. Use this after
  `kernel_observer_long` to test realistic heterogeneity.
- `kernel_single_cable_extracellular_long`: longer
  `point_source_extracellular` trace with the same CPU/GPU comparison settings
  as the double-cable run: sizes `100/300/600`, `duration=10 ms`,
  `dt=0.01 ms`, and a target of `51` compartments.
- `kernel_double_cable_extracellular_long`: longer
  `double_cable_extracellular` trace with MRG double-cable rows, analytical
  point-source extracellular stimulation, sizes `100/300/600`,
  `duration=10 ms`, `dt=0.01 ms`, and a target of `51` compartments.
- `kernel_double_cable_extracellular_auto_long`: same long double-cable trace,
  but passes `--double-cable-block-solver auto` so GPU runs resolve to PCR and
  CPU runs resolve to Thomas. It includes sizes `100/300/600/2000` for the
  longer CPU/GPU comparison after the PCR result.
- `kernel_double_cable_observer_auto_long`: same MRG double-cable extracellular
  trace, but uses `Recording.none()` plus solver-side `PeakVoltage` and
  `Activation` observers at sizes `100/300/600/2000`. Use this to compare
  retained center traces against compact observer-only output on GPU.
- `kernel_double_cable_extracellular_pcr_long`: same long double-cable trace,
  but passes `--double-cable-block-solver pcr` to test the experimental
  parallel cyclic-reduction block solver against the default Thomas scan. Use
  it to reproduce forced-PCR results; current CPU evidence makes PCR much
  slower than the default Thomas solver.

The notebook clones `bench-colab`, installs `.[examples,benchmark]`, verifies
only the enabled backend(s), and runs the selected case for the enabled labels.
For CPU-only runs, keep `RUN_GPU = False` and `RUN_CPU = True`; the notebook
writes a per-run `cpu_summary.csv` in addition to the normal manifest and
workload summaries. For CPU/GPU comparison runs, enable both labels to also
write `comparison_summary.csv`.
For the `setup_scale` case, the effective commands are:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --duration 0.30 \
  --dt 0.05 \
  --compartments 11 \
  --warmups 1 \
  --sweep-repeats 3 \
  --prefix gpu \
  --out-dir benchmark/results/hotpaths/colab_cpu_gpu_setup_scale_YYYYMMDD_HHMMSS \
  --no-print-summary
```

```bash
JAX_PLATFORMS=cpu python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --duration 0.30 \
  --dt 0.05 \
  --compartments 11 \
  --warmups 1 \
  --sweep-repeats 3 \
  --prefix cpu \
  --out-dir benchmark/results/hotpaths/colab_cpu_gpu_setup_scale_YYYYMMDD_HHMMSS \
  --no-print-summary
```

The output folder is created inside the Colab checkout with both traces:

```text
/content/AxonScope/benchmark/results/hotpaths/colab_cpu_gpu_<case>_YYYYMMDD_HHMMSS/
    gpu/
    cpu/
    comparison_summary.csv
```

`comparison_summary.csv` compares the main stages for matching workload/size
pairs: `simulation.pool.total`, `dispatch.build_plan`, `runtime.prepare`,
`inputs.intracellular`, `inputs.extracellular`, `kernel.enqueue`,
`kernel.wait`, `results.split_batch`, and `results.to_public`.

For kernel-scaling evidence, switch `CASE` in the notebook instead of editing
the command by hand. The result folder name includes the case, for example:

```text
benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_YYYYMMDD_HHMMSS/
```

Then the notebook zips the parent `colab_cpu_gpu_.../` folder and downloads it
directly through the browser with `google.colab.files.download(...)`. No Google
Drive mount is used.

Important: `google.colab.files.download(...)` works from a Colab notebook cell,
not from the Colab terminal or a plain `python` REPL. If you ran the script in
the terminal, keep the printed archive path and run this in a notebook cell:

```python
from google.colab import files

files.download("/content/AxonScope/benchmark/results/hotpaths/<run_id>.zip")
```

After download, unzip the archive into your local:

```text
benchmark/results/hotpaths/
```

The output folder will contain:

- `gpu/manifest.json`
- `cpu/manifest.json`
- `comparison_summary.csv`
- each workload's `events.jsonl`, `summary.csv`, and `metadata.json` under the
  matching `gpu/` or `cpu/` folder

Each manifest also includes each workload's
`simulation.estimate().to_dict()` output, including dense `Vstim`, retained Vm,
factorized footprint, sampled-stimulus estimates, and the run parameters such
as warmup count, duration, `dt`, compartments, and sweep repeats.

## 4. Optional Local CPU Reference

The Colab notebook already runs a forced CPU reference. A local CPU reference is
still useful when you want to compare Colab against the development machine:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --duration 0.30 \
  --dt 0.05 \
  --compartments 11 \
  --warmups 1 \
  --sweep-repeats 3 \
  --prefix cpu_YYYYMMDD_HHMMSS \
  --no-print-summary
```

Run a local CPU version of the longer observer case:

```bash
python benchmark/hotpaths/run.py \
  --workload observer_only \
  --sizes 500 1000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --prefix cpu_kernel_observer_long_YYYYMMDD_HHMMSS \
  --no-print-summary
```

## 5. Compare Before Refactoring

Compare these stages first:

- `runtime.prepare`
- `inputs.intracellular`
- `inputs.extracellular`
- `kernel.enqueue`
- `kernel.wait`
- `results.split_batch`
- `results.to_public`

If CPU and GPU stay close while `runtime.prepare`, `inputs.extracellular`, or
the manifest's dense `Vstim` estimate dominates, prioritize preparation and
drive/observer reductions before adding higher-level study APIs.

If `kernel.wait` dominates and separates clearly by device, prioritize backend
kernel/runtime isolation.

## Troubleshooting

### `make bench-colab-push` Refuses To Run

The local working tree is dirty. Commit or stash first:

```bash
git status --short
```

### Colab Reports `Colab runtime is not using a GPU backend`

Switch the Colab runtime to a GPU accelerator and restart the runtime. The
trace is not useful for CPU/GPU comparison unless `jax.default_backend()` is
`gpu`.

### Colab Reports `Forced CPU backend did not activate`

The notebook runs CPU traces in a separate process with `JAX_PLATFORMS=cpu`.
Restart the runtime and rerun the notebook cell. If it still fails, Colab's JAX
installation changed enough that the CPU/GPU comparison protocol needs updating.

### Colab Still Runs Old Code

The usual causes are:

- local changes were not committed before `make bench-colab-push`;
- the notebook cloned a branch other than `bench-colab`;
- the Colab runtime still has an old editable install.

The provided cell removes `/content/AxonScope` before cloning. If needed, also
restart the runtime.

### `files.download(...)` Fails With `NoneType` Or `kernel`

The benchmark was run from the Colab terminal instead of a notebook cell. The
run is still valid if the archive was created. Copy the printed archive path and
download it from a notebook cell:

```python
from google.colab import files

files.download("/content/AxonScope/benchmark/results/hotpaths/<run_id>.zip")
```
