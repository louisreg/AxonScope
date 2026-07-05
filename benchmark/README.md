# AxonScope Benchmark Surface

P11A resets benchmarking around two canonical curve scripts:

- `benchmark/curves/threshold_curves.py`
- `benchmark/curves/recruitment_curves.py`

Historical scripts, notebooks, reports, and raw outputs live under
`benchmark/legacy/pre_p11/`. They are archive material, not current performance
evidence.

## Commands

Use the shared launcher for local, GPU, and future NRV runs:

```bash
python benchmark/run.py --list
python benchmark/run.py --script threshold_curves --preset quick --platform cpu --dry-run
python benchmark/run.py --script recruitment_curves --preset gpu_smoke --platform gpu --dry-run
python benchmark/run.py --script threshold_curves --preset quick --platform cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --cpu
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --preset gpu_smoke --platform gpu --machine-shape NvidiaTeslaP100
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --preset gpu_trace_smoke --platform gpu --machine-shape NvidiaTeslaP100
```

Both curve scripts use the same option vocabulary:

```bash
python benchmark/run.py \
  --script threshold_curves \
  --preset local_smoke \
  --platform cpu \
  --recording probe_vm \
  --cable single_cable \
  --population single_model \
  --diameters same_diameter \
  --memory-trace all \
  --profile \
  --dry-run
```

Current real runs support AxonScope point-source activation-threshold and
recruitment curves. `--dry-run` still only writes `cases.csv` for case review.
Real execution writes timing, memory, environment, raw activation rows, and
curve summaries. Block thresholds and NRV execution are intentionally left as
future benchmark/baseline work until their adapter contracts are defined.

Use `--time-chunk-steps default` or omit the option to keep AxonScope's
recording-specific default; for observer-only runs this currently means the
stable VmRaster default. Use `--time-chunk-steps unchunked` or `none` to force
one full scan, and use an integer such as `--time-chunk-steps 500` for an
explicit local chunk size. Benchmark artifacts record both `time_chunk_policy`
and `time_chunk_steps` so default, unchunked, and explicit one-chunk runs can be
compared without ambiguity.

Kaggle runs use `benchmark/kaggle/run_kernel.py`, which packages a script
kernel around the same `benchmark/run.py` command, forwards extra options, and
downloads a zipped result directory after success. Use `--cpu` or `--platform
cpu` without `--machine-shape` for a CPU-only Kaggle run. Use `--platform cpu
--machine-shape NvidiaTeslaP100` when you deliberately want the CPU benchmark
path on a Kaggle GPU machine for closer CPU/GPU environment comparisons.

## Presets

Presets live in `benchmark/workloads/curve_options.py`:

- `quick`
- `local_smoke`
- `local_realistic`
- `cpu_publication`
- `gpu_smoke`
- `gpu_trace_smoke`
- `gpu_realistic`
- `nrv_smoke`
- `nrv_full`

They define scale and defaults for repeats, warmups, duration, `dt`, `Nx`,
`Naxons`, precision, recording mode, platform, memory tracing, profiling,
threshold iterations, and recruitment amplitude count.

`gpu_smoke` is a short GPU functional and memory smoke. It should not enable
whole-session JAX tracing or device-memory pprof capture by default. Use
`gpu_trace_smoke` when you explicitly want tracing: it is intentionally limited
to one small pool and two or three amplitude evaluations so Perfetto/XPlane
artifacts stay inspectable.

Device-memory pprof capture is stage-filtered. Curve scripts default to
`kernel.wait`; pass `--jax-device-memory-profile-stage runtime.prepare` or
repeat the flag to capture more stages. Use
`--jax-device-memory-profile-stage all` only on tiny trace cases.

FP64 runs require a JAX process with x64 enabled before importing JAX. For a
fresh shell, use the project environment and set `JAX_ENABLE_X64=1` before
running an FP64 preset.

## Outputs

Every real run should write a self-contained result directory:

- `environment.json`: machine, OS, Python, package, git, backend, CPU/GPU/RAM,
  precision, execution, recording, observer, cache, and NRV metadata.
- `cases.csv`: the exact benchmark cases requested.
- `events.jsonl`: stage-level wall-clock events.
- `summary.csv`: aggregate stage timing.
- `memory_summary.csv`: RSS, `tracemalloc`, device-memory, and profile summary.
- `artifacts/`: raw traces, device-memory profiles, and debug outputs.
- `plots/`: generated figures for accepted campaign outputs.
- `results.csv`: row-level activation observations for each tested amplitude.
- `curve_summary.csv`: threshold or recruitment summaries.
- `manifest.json`: the selected script, case name, options, and output map.

Do not make speed or memory claims from console output alone. Use a fresh result
directory with git metadata and saved traces.

## Instrumentation

For scripts, prefer the context-manager style:

```python
import axonscope as axs

with axs.benchmark(
    "benchmark/results/example",
    print_summary=False,
    sync_device=True,
    record_shapes=True,
    memory_trace="all",
    memory_top_n=10,
    profile=True,
    profile_backend="jax",
    jax_device_memory_profile=True,
):
    result = axs.AxonSimulation(...).run()
```

For notebooks and debugging, use the explicit enable/disable style:

```python
import axonscope as axs

session = axs.enable_benchmark(
    "benchmark/results/notebook_debug",
    print_summary=False,
    memory_trace="rss",
    profile=True,
    profile_backend="auto",
)
try:
    result = axs.AxonSimulation(...).run()
finally:
    report = axs.disable_benchmark(print_summary=True)
```

Use explicit instrumentation imports around non-solver preparation,
post-processing, or external baseline work:

```python
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata

with benchmark_span("stage.name"):
    record_benchmark_metadata(case="example")
```

## Trace Analysis

Summarize saved events and trace/profile artifacts with:

```bash
python benchmark/analysis/trace_summary.py benchmark/results/example
```

JAX profiler traces are TensorBoard/Perfetto artifacts. JAX device-memory
profiles are pprof artifacts; open them with `pprof --web <profile.prof>`.

## P11B Cold-Path Audit

Before changing solver routes or scheduling, turn fresh curve outputs into a
stage-level timing and memory map:

```bash
python benchmark/analysis/cold_path_audit.py \
  benchmark/results/p11b_baseline/threshold_large_cpu_7ebe7c3 \
  benchmark/results/p11b_baseline/recruitment_large_cpu_7ebe7c3 \
  --output benchmark/results/p11b_baseline/cold_path_cpu_audit_7ebe7c3
```

The audit writes:

- `cold_path_stage_rows.csv`: one row per benchmark span with timing, RSS,
  `tracemalloc`, device-memory, environment, git, and case metadata.
- `cold_path_group_summary.csv`: grouped P11B view for pool build, dispatch,
  runtime preparation, input lowering, kernel, and result assembly.
- `plots/cold_path_group_time.png`, `plots/cold_path_top_stages.png`, and
  `plots/cold_path_memory.png`.

Use `memory_trace=rss` or `device` for large local/GPU sweeps. Keep
`memory_trace=all`, JAX profiling, and device-memory pprof capture for tiny
trace cases only.

For optimization triage, rank nested event spans by exclusive self time:

```bash
python benchmark/analysis/bottleneck_report.py \
  benchmark/results/p11b_baseline/threshold_n1000_cpu_scout_f895a03 \
  benchmark/results/p11b_baseline/recruitment_n1000_cpu_scout_f895a03 \
  --output benchmark/results/p11b_baseline/bottleneck_n1000_current
```

The bottleneck report writes event-level rows, stage/group rankings, cache
signals, memory context, and a Markdown summary. Use it before making solver
optimization claims; it is a triage artifact, not a benchmark claim by itself.

For time-chunk policy triage, use the campaign runner instead of hand-written
loops:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,50,250,500,1000 \
  --recording observer_only \
  --memory-trace rss
```

It writes separate raw result directories per policy and a merged summary of
observed chunk metadata plus kernel/observer timings.

## Publishability

A benchmark result is publishable only if the run directory contains the full
case list, fresh environment/git metadata, timing traces, memory traces, and
the exact script/preset/options used. GPU claims need either local GPU metadata
or Kaggle metadata from the future P11A Kaggle runner. NRV comparisons wait for
the baseline adapter contract in `benchmark/baselines/`.
