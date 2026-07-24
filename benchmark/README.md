# AxonFleet Benchmark Surface

Benchmarks are development and validation tools, not public AxonFleet APIs.
They exercise the same public simulation workflows as the examples and may use
private instrumentation only to explain performance.

## Main Launcher

List the registered scripts and presets:

```bash
python benchmark/run.py --list
```

Run a local smoke case:

```bash
python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu
```

The registered scripts are:

- `threshold_curves`: activation-threshold workloads;
- `recruitment_curves`: recruitment workloads;
- `basic_examples`: executable performance gates for basic examples 06-08;
- `with_nrv_examples`: executable gate for with-NRV example 01;
- `runner_group_scheduling`: runner grouping and scheduling costs;
- `runner_plan_validation`: P20 lazy-plan and local Runner CPU/GPU acceptance;
- `membrane_recording_validation`: canonical full-recording CPU/GPU numerical
  equivalence for every NRV-backed built-in family, Passive, and mixed
  MRG+Markov models;
- `recruitment_amplitude_batch`: realistic amplitude batching and reuse;
- `single_cable_triton_gate`: focused single-cable Triton acceptance gate;
- `membrane_temporal`: complete temporal membrane/cable workloads;
- `kinetic_transition_tables`: experimental Markov transition-table gate.

Use `--dry-run` before large or remote runs. It writes the resolved case table
without executing a simulation.

## Directory Map

- `analysis/`: generic trace, cache, HLO, bottleneck, chunk, and proof tools;
- `baselines/`: independent external scientific references;
- `campaigns/`: process-isolated benchmark matrices;
- `curves/`: canonical curve launchers and P18 Nav validations;
- `runner/`: Runner scheduling and runnable-plan validation;
- `examples/`: executable-documentation and startup workloads;
- `kaggle/`: remote CPU/GPU packaging and submission;
- `protocols/`: protocol-level amplitude batching workload;
- `solvers/`: focused candidate/acceptance gates;
- `workloads/`: shared curve options and execution;
- `results/`: ignored generated artifacts plus deliberately retained evidence.

## Curve Workloads

```bash
python benchmark/run.py --script threshold_curves --preset quick --platform cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu
```

Threshold benchmarks cover activation only. Conduction block requires a
separate validated protocol and is not an accepted option here.

The curve workloads share one pool/template/footprint construction path and
one amplitude evaluation path. Their principal matrix axes include cable
formulation, population shape, diameter cohort, recording, precision, chunk
size, amplitude batching, and repeat-pool reuse.

## Example Gates

```bash
python benchmark/run.py \
  --script basic_examples --preset quick --platform cpu --examples 06,07,08

python benchmark/run.py \
  --script with_nrv_examples --preset quick --platform cpu --examples 01
```

`benchmark/examples/basic_08_startup.py` separately instruments import,
population construction, footprint construction, simulation setup, and run
costs for a large version of basic example 08.

`benchmark/examples/runtime_benchmarking_options.py` demonstrates benchmark
instrumentation and output generation without becoming a public example.

## Runtime And Membrane Gates

Close the local Runner phase with the same acceptance workload on CPU and one
GPU:

```bash
python benchmark/run.py \
  --script runner_plan_validation \
  --preset cpu_publication \
  --platform cpu \
  --scales 1024,4096 \
  --output benchmark/results/p20_runner_validation_cpu
```

The campaign executes simple, mixed, numeric-axis, sweep, threshold, and study
plans; checks cold/warm reuse, `Runner.clear()`, structural invalidation, and
cooperative cancellation; and records compact 1024/4096 observer runs. Compare
the matching CPU/GPU `validation.json` files with
`benchmark/analysis/runner_plan_validation.py`.

Full membrane-recording equivalence is captured once per backend so a CUDA JAX
process never attempts to execute the CPU LAPACK route:

```bash
python benchmark/run.py --script membrane_recording_validation \
  --preset gpu_smoke --platform cpu \
  --output benchmark/results/membrane_recording_validation_cpu
python benchmark/run.py --script membrane_recording_validation \
  --preset gpu_smoke --platform gpu \
  --output benchmark/results/membrane_recording_validation_gpu
python -m benchmark.analysis.membrane_recording_validation \
  benchmark/results/membrane_recording_validation_cpu \
  benchmark/results/membrane_recording_validation_gpu \
  --output benchmark/results/membrane_recording_validation_cpu_gpu
```

The comparison retains a strict pointwise verdict and a bounded normalized
trajectory verdict. The latter handles floating-point drift between the CPU
Thomas and GPU tiled-Thomas solvers; it does not accept normalized RMSE above
0.1%, peak error above 0.5%, or Vm errors above their stricter limits.

Generated membrane cache loading:

```bash
python benchmark/membrane_runtime_cache.py \
  --output benchmark/results/membrane_runtime_cache/summary.json
```

Isolated membrane kinetics:

```bash
python benchmark/membrane_kinetics.py \
  --output benchmark/results/membrane_kinetics/summary.json
```

Full temporal workloads:

```bash
python benchmark/run.py \
  --script membrane_temporal --preset quick --platform cpu
```

`membrane_temporal.py` also supports persistent-compilation-cache replay and
JAX phase/HLO capture. `kinetic_transition_tables.py` remains an experimental
last-resort gate: transition tables are not a production runtime path unless
the measured end-to-end result beats exact generated kinetics.

The retained
`results/p17_generated_runtime_cache_local_20260718/summary.json` records the
small local generated-runtime cache comparison. Fresh timing claims require a
new run on the current commit.

## Scientific Validation

`curves/nav_isoform_voltage_clamp.py` reproduces the ModelDB 230137 Nav1.x
clamp surfaces with generated JAX membrane programs. The independent NEURON
reference is produced by `baselines/modeldb_230137_voltage_clamp.py`; MOD files
remain external.

```bash
MPLBACKEND=Agg python benchmark/curves/nav_isoform_voltage_clamp.py \
  --output benchmark/results/nav_isoform_voltage_clamp \
  --modeldb-reference /path/to/modeldb_230137_voltage_clamp.json
```

`curves/nav_cable_validation.py` validates a benchmark-only Nav1.6/KDR/leak
composition through both canonical cable formulations. It is not a new public
axon model.

`analysis/full_step_operator.py` is the executable algebraic proof that the
staged single- and double-cable updates match their assembled frozen operator.

## Campaigns

Time-chunk matrix:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,50,250,500 \
  --recordings full_recording,full_vm,probe_vm,observer_only \
  --output benchmark/results/time_chunk_sweep
```

`full_recording` uses the public `Recording.full()` contract, including all
membrane quantities exposed by each model. `full_vm` retains voltage only;
`observer_only` retains compact solver-side observations without dense Vm.

Typed double-cable policy matrix:

```bash
python benchmark/campaigns/double_cable_solver_policy.py \
  --platform cpu \
  --preset quick \
  --solver auto,thomas \
  --recording observer_only \
  --output benchmark/results/double_cable_solver_policy
```

See `campaigns/README.md` for the focused contract of each campaign.

## Kaggle CPU/GPU

The Kaggle runner packages the current checkout, submits one script or
campaign, streams logs, downloads outputs, and avoids archiving dependency and
cache directories.

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --script recruitment_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100
```

Use `--cpu` for the Kaggle CPU path. See `kaggle/README.md` for campaign and
profiling examples.

## Instrumentation

Benchmark sessions write nested spans through `axs.benchmarking`. Important
runtime stages include:

- `dispatch.build_plan`;
- `runtime.prepare`;
- `inputs.extracellular`;
- `kernel.enqueue`;
- `kernel.dispatch_jax`;
- `kernel.wait`;
- observer finalization and host transfer;
- public result assembly.

`kernel.dispatch_jax` includes asynchronous JAX submission and may contain
deferred device work. Interpret it together with `kernel.wait`, synchronized
wall time, and Perfetto traces.

Generic post-processing tools live under `analysis/`:

```bash
python benchmark/analysis/trace_summary.py benchmark/results/RUN
python benchmark/analysis/bottleneck_report.py benchmark/results/RUN \
  --output benchmark/results/RUN/bottleneck
python benchmark/analysis/cold_path_audit.py benchmark/results/RUN \
  --output benchmark/results/RUN/cold_path
```

## Outputs And Retention

A reproducible run should retain resolved options, environment and git
metadata, case/result tables, timing events, summaries, and any explicitly
requested profiles. GPU claims also require hardware metadata.

Generated outputs belong under `benchmark/results/` and are ignored by
default. Keep a small artifact in git only when a current document names it as
scientific or architectural evidence. Never retain package environments,
downloaded wheels, JAX caches, generated model caches, or profiler scratch
directories inside an archive.

Timing claims must come from fresh runs on suitable hardware. Laptop smoke
runs validate execution but are not stable performance evidence.
