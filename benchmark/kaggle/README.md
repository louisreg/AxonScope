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
  --case-filter observer_only
```

Submit and wait for a CPU-only Kaggle run:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script threshold_curves \
  --cpu \
  --case-filter observer_only
```

To compare the AxonScope CPU path and GPU path on a closer Kaggle runtime,
keep the GPU machine shape but switch only the benchmark platform:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script threshold_curves \
  --preset quick \
  --platform cpu \
  --machine-shape NvidiaTeslaP100 \
  --case-filter observer_only
```

Capture a JAX trace on a deliberately tiny GPU case:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script threshold_curves \
  --preset gpu_trace_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --case-filter observer_only
```

Submit the double-cable solver-policy campaign on the CPU path of a Kaggle GPU
machine:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p11c-solver-policy-cpu \
  --campaign double_cable_solver_policy \
  --preset quick \
  --platform cpu \
  --machine-shape NvidiaTeslaP100 \
  --curve-script threshold_curves,recruitment_curves \
  --solver auto,thomas \
  --recording observer_only,probe_vm \
  --n-axons 64,1024 \
  --nx 89,129 \
  --precision fp32 \
  --repeats 3 \
  --warmups 1
```

Run the matching GPU policy campaign by switching `--platform`, the slug, and
the solver set:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p11c-solver-policy-gpu \
  --campaign double_cable_solver_policy \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --curve-script threshold_curves,recruitment_curves \
  --solver auto,tiled_thomas \
  --recording observer_only,probe_vm \
  --n-axons 64,1024,4096,8192 \
  --nx 89,129 \
  --precision fp32 \
  --tiled-thomas-block-b 32,64 \
  --repeats 3 \
  --warmups 1
```

Submit the P11B time-chunk sweep campaign on the CPU path of a Kaggle GPU
machine:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p11b-time-chunk-sweep-cpu \
  --campaign time_chunk_sweep \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --machine-shape NvidiaTeslaP100 \
  --policies default,unchunked,50,250,500,1000 \
  --recordings full_vm,probe_vm,observer_only \
  --n-axons 1000 \
  --nx 101 \
  --tsim 10 \
  --dt 0.01 \
  --amplitude-count 5 \
  --diameters different_diameters \
  --memory-trace rss \
  --memory-top-n 0
```

Switch only `--platform gpu`, the slug, and the memory trace to `device` for
the matching GPU sweep.

Older P11B/P11C solver-stage, lowering, PCR-state, and large-population
analysis campaigns are no longer part of the active Kaggle runner. Keep current
Kaggle validation on curve scripts and solver-policy campaigns.

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

Keep whole-session JAX tracing on small trace presets only. Trace outputs can
grow quickly; use `gpu_smoke` for functional GPU acceptance and
`gpu_trace_smoke` for Perfetto/XPlane/device-memory artifacts. Device-memory
pprof capture defaults to `kernel.wait`; add
`--jax-device-memory-profile-stage <stage>` only on deliberately tiny trace
cases.

Use `--no-publish-branch --branch <branch>` only when the target branch already
contains the exact code to benchmark. Use `--require-clean-git` when you want
the runner to fail instead of warning about uncommitted local changes.
