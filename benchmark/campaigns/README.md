# Benchmark Campaigns

Campaigns group the two canonical curve scripts into reproducible benchmark
matrices. They should call `benchmark/run.py`; they should not implement solver
or workload logic directly.

P11A defines the publication campaign as a reproducible plan, not a one-shot
script yet. The campaign is built only from:

- `threshold_curves`;
- `recruitment_curves`.

## P11A Campaign Plan

| phase | purpose | preset | required axes |
| --- | --- | --- | --- |
| acceptance | prove the surface works locally and on Kaggle GPU | `quick`, `gpu_smoke` | threshold, recruitment, CPU, GPU, observer-only |
| recording modes | quantify output cost | `local_smoke`, future GPU override | observer-only, probe Vm, full Vm |
| scale curves | map asymptotic behavior | `local_realistic`, `gpu_realistic` | `dt`, `Nx`, `Naxons` |
| model/cable coverage | expose solver route differences | `local_smoke`, `gpu_smoke` | single-cable, double-cable, mixed populations |
| cohort coverage | expose batching and heterogeneity cost | `local_smoke`, `gpu_smoke` | same-diameter, different-diameter cohorts |
| precision | quantify dtype cost and numerical surface | `cpu_publication`, `gpu_realistic` | FP32, FP64 where supported |
| tracing | inspect optimization targets | `gpu_trace_smoke` | one small pool, two or three amplitudes, selected profile stages |
| baselines | compare external runtimes | future `nrv_smoke`, `nrv_full` | NRV after adapter contract |

Activation thresholds and recruitment curves are current real-execution
outputs. Block thresholds stay in the campaign matrix, but must not be executed
until their protocol semantics are defined. NRV comparison starts only after the
baseline adapter contract is implemented.

Every campaign must write a manifest with fixed presets, raw data paths, plot
paths, summary-table paths, git metadata, and hardware metadata.

## P11B Time-Chunk Sweep

Use the time-chunk campaign to compare AxonScope's observer/kernel chunking
policies without hand-written shell loops. It runs each policy in a separate
Python process and writes one result directory per policy plus a campaign
summary table.

Quick local smoke:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,100 \
  --recording observer_only \
  --amplitude-count 1 \
  --memory-trace rss \
  --memory-top-n 0 \
  --output benchmark/results/p11b_time_chunk_sweep_quick
```

Recording-mode smoke:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,100 \
  --recordings full_vm,probe_vm,observer_only \
  --amplitude-count 1 \
  --memory-trace rss \
  --memory-top-n 0 \
  --output benchmark/results/p11b_time_chunk_recording_smoke
```

Bounded P11B sweep shape:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,50,250,500,1000 \
  --recording observer_only \
  --n-axons 1000 \
  --nx 101 \
  --tsim 10 \
  --dt 0.01 \
  --amplitude-count 5 \
  --diameters different_diameters \
  --memory-trace rss \
  --memory-top-n 0 \
  --output benchmark/results/p11b_time_chunk_sweep_cpu
```

The campaign writes `time_chunk_sweep_manifest.json`,
`time_chunk_sweep_summary.csv`, and `time_chunk_sweep_report.md`. Summary rows
include the requested policy, the effective benchmark options, observed
`kernel.dispatch_jax` chunk metadata, `curve.simulate`, `kernel.enqueue`,
`kernel.dispatch_jax`, `kernel.wait`, `kernel.finalize_observer`, Vm
materialization, row/cohort assembly, and split-result timings. Use
`--recording` for one mode and `--recordings` for a matrix across full Vm,
probe Vm, and observer-only outputs.

## Publication Outputs

Publication runs should retain:

- raw result directories from every command;
- a campaign manifest with exact commands, presets, git SHA, and hardware;
- merged timing and memory tables derived from `summary.csv` and
  `memory_summary.csv`;
- curve tables derived from `curve_summary.csv`;
- plots for threshold curves, recruitment curves, and scale trends;
- trace summaries only from tiny trace cases, never from full campaign sweeps.
