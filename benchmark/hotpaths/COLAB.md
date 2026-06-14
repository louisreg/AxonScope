# Colab GPU Hotpath Protocol

Use this when local GPU execution is not available. The local machine publishes
one committed AxonScope revision to the moving `bench-colab` branch, and Colab
always clones that branch before running the hotpath workloads.

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
`REPO_URL` once, then run the notebook cell.

The notebook clones `bench-colab`, installs `.[examples,benchmark]`, verifies
that JAX uses a GPU backend, and runs:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --warmups 1 \
  --prefix colab_gpu_YYYYMMDD_HHMMSS \
  --out-dir benchmark/results/hotpaths \
  --no-print-summary
```

The output folder is created inside the Colab checkout:

```text
/content/AxonScope/benchmark/results/hotpaths/<run_id>/
```

Then the notebook zips `<run_id>/` and downloads it directly through the
browser with `google.colab.files.download(...)`. No Google Drive mount is used.

After download, unzip the archive into your local:

```text
benchmark/results/hotpaths/
```

The output folder will contain:

- `manifest.json`
- each workload's `events.jsonl`
- each workload's `summary.csv`
- each workload's `metadata.json`

## 4. Run A Matching Local CPU Reference

Run a local CPU reference with the same hotpath preset and warmup count:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --warmups 1 \
  --prefix cpu_YYYYMMDD_HHMMSS \
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

If CPU and GPU stay close while `inputs.extracellular` or `runtime.prepare`
dominates, prioritize Phase 3 preparation/cohort reuse before touching kernels.

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

### Colab Still Runs Old Code

The usual causes are:

- local changes were not committed before `make bench-colab-push`;
- the notebook cloned a branch other than `bench-colab`;
- the Colab runtime still has an old editable install.

The provided cell removes `/content/AxonScope` before cloning. If needed, also
restart the runtime.
