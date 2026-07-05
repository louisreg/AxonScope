# Kaggle Benchmark Runner

This runner submits the current P11A benchmark surface to Kaggle through the
same `benchmark/run.py` script used locally. It does not know about legacy
benchmark suites.

Prepare a package without touching the network:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script threshold_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --dry-run \
  --no-publish-branch
```

Submit and wait for a GPU run:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script threshold_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --case-filter observer_only \
  --memory-trace all \
  --profile
```

Run recruitment on a T4:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script recruitment_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaT4
```

The local runner:

- publishes `HEAD` to `kaggle-bench/<short-sha>` by default, so Kaggle clones a
  stable branch;
- writes the uploaded kernel package under
  `benchmark/results/kaggle/<timestamp>_<script>_<preset>_<machine>/kernel/`;
- streams available Kaggle logs while polling status;
- downloads the zipped benchmark result archive after success;
- records local submission metadata in `submission.json`.

The Kaggle script clones the configured branch, installs `.[benchmark]`, checks
GPU availability when requested, runs:

```bash
python benchmark/run.py --script ... --preset ... --platform ... --output ...
```

and writes `kaggle_hardware.json`, `kaggle_command.json`, benchmark outputs,
and `axonscope_benchmark_results_<run_id>.zip`.

Use `--no-publish-branch --branch <branch>` only when the target branch already
contains the exact code to benchmark. Use `--require-clean-git` when you want
the runner to fail instead of warning about uncommitted local changes.
