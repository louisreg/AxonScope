# P11B Low-Level Bottleneck Notes

This note summarizes the CPU/GPU time-chunk matrix captured on 2026-07-06.
It is an optimization triage document, not a policy decision. Do not change the
default time-chunk policy from these results alone.

## Artifact Map

Inputs:

- Threshold CPU:
  `benchmark/results/kaggle/20260706_185153_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`
- Threshold GPU:
  `benchmark/results/kaggle/20260706_185207_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`
- Recruitment CPU:
  `benchmark/results/kaggle/20260706_182247_time_chunk_sweep_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`
- Recruitment GPU:
  `benchmark/results/kaggle/20260706_182247_time_chunk_sweep_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`

Derived report:

- `time_chunk_matrix_report.md`
- `time_chunk_matrix_rows.csv`
- `time_chunk_best_rows.csv`
- `plots/curve_time_heatmaps.png`
- `plots/best_policy_stage_group_breakdown.png`
- `plots/best_policy_kernel_result_breakdown.png`
- `plots/best_cpu_gpu_speedup.png`
- `plots/stage_breakdown_threshold_cpu.png`
- `plots/stage_breakdown_threshold_gpu.png`
- `plots/stage_breakdown_recruitment_cpu.png`
- `plots/stage_breakdown_recruitment_gpu.png`
- `plots/memory_cpu_rss_end.png`
- `plots/memory_gpu_jax_device_end.png`
- `plots/memory_gpu_nvidia_smi_end.png`

## Scope

Workload:

- `Naxons=1000`
- `Nx=101`
- `tsim=10 ms`
- `dt=0.01 ms`
- `fp32`
- different-diameter cohorts
- policies: `default`, `unchunked`, `50`, `250`, `500`, `1000`
- recordings: `full_vm`, `probe_vm`, `observer_only`
- scripts: `threshold_curves`, `recruitment_curves`
- platforms: CPU path and GPU path on Kaggle P100 image

Threshold runs used commit `6c92042`; recruitment runs used commit `091992b`.
Both commits are close enough for this cartography because the intermediate
change fixed benchmark orchestration for threshold options and did not change
solver behavior.

## Main Findings

CPU is kernel dominated.

Best CPU rows are dominated by the `kernel` group:

| workflow | recording | best policy | total | kernel group |
| --- | --- | --- | ---: | ---: |
| threshold | full Vm | `1000` | 16.34 s | 13.79 s |
| threshold | probe Vm | `1000` | 15.66 s | 13.32 s |
| threshold | observer only | `default` | 14.44 s | 12.35 s |
| recruitment | full Vm | `1000` | 35.41 s | 30.95 s |
| recruitment | probe Vm | `default` | 33.96 s | 30.10 s |
| recruitment | observer only | `default` | 33.16 s | 30.16 s |

For full/probe CPU, the detailed view shows `kernel.wait` as the main measured
sub-stage. For observer-only CPU, the detailed view shifts toward
`kernel.dispatch_jax`, with `kernel.combine_observer_chunks` visible when small
chunks are used.

GPU is not just a smaller CPU profile.

Best GPU rows:

| workflow | recording | best policy | total | top group |
| --- | --- | --- | ---: | ---: |
| threshold | full Vm | `unchunked` | 10.02 s | kernel, 4.57 s |
| threshold | probe Vm | `unchunked` | 8.98 s | kernel, 4.49 s |
| threshold | observer only | `unchunked` | 8.89 s | kernel, 4.78 s |
| recruitment | full Vm | `1000` | 14.25 s | curve/setup, 8.15 s |
| recruitment | probe Vm | `default` | 12.71 s | curve/setup, 5.32 s |
| recruitment | observer only | `unchunked` | 12.41 s | curve/setup, 7.07 s |

Small chunks are visibly bad on GPU, especially `50/default` observer-only in
the recruitment matrix. However, this note intentionally does not conclude that
the default policy should change. The immediate goal is to reduce low-level
costs so that future policy decisions are less forced by overhead.

## Memory Reading

CPU memory is RSS-based. Peak RSS for best CPU rows is roughly:

- full Vm: about 2.2 GiB
- probe Vm: about 1.9 GiB
- observer-only: about 1.1 GiB

GPU memory must be read in two layers:

- `peak_device_end_mib` is live JAX device allocation.
- `peak_nvidia_smi_end_mib` is process/context memory and includes allocator
  reservation.

Best GPU rows show live JAX memory around:

- full Vm: about 779 MiB
- probe Vm: about 424 MiB
- observer-only: about 9-34 MiB

`nvidia-smi` sits around 12.7 GiB for these runs because JAX reserves a large
device pool on the P100 image. Do not treat this as live arrays alone.

## Low-Level Optimization Targets

Priority 1: kernel execution and synchronization.

CPU full/probe and recruitment are dominated by the kernel group and especially
`kernel.wait`. The next useful question is whether this is true solve work,
host/device synchronization placement, repeated compilation/lowering, or
materialization hidden behind wait.

Concrete probes:

- Add narrower spans inside the JAX kernel call path around solve, scan/body,
  observer update, and host synchronization boundaries.
- Compare first/repeat event trees for the same row to separate compile,
  lowering, and actual execution.
- Keep CPU and GPU traces side by side for the same case to avoid optimizing a
  CPU-only artifact.

Priority 2: GPU curve/setup overhead in recruitment.

The best GPU recruitment rows are dominated by `curve/setup`, not by the
visible kernel sub-stage. This suggests overhead before or around each
amplitude evaluation.

Concrete probes:

- Split `curve/setup` further into amplitude update, pool/stimulus update,
  result preparation, and per-amplitude orchestration.
- Check whether recruitment repeats rebuild or re-lower structures that should
  be stable across amplitudes.
- Inspect whether same cohort signatures are reused across amplitudes after
  diameter quantization.

Priority 3: observer chunk combine/finalize costs.

Observer-only CPU and GPU disagree in shape. CPU can tolerate small chunks in
some runs; GPU strongly prefers large/unchunked chunks. The target is not to
pick a policy yet, but to make small-chunk overhead understandable.

Concrete probes:

- Keep `kernel.combine_observer_chunks` and `kernel.finalize_observer` visible
  in every observer run.
- Inspect array shapes and host/device transfers during combine/finalize.
- Verify whether chunk combination can stay device-side longer or avoid Python
  repacking.

Priority 4: result assembly and Vm materialization.

Full/probe GPU rows show visible live device memory and result materialization,
especially full Vm. This is not the first bottleneck everywhere, but it matters
for memory and publication-scale runs.

Concrete probes:

- Split full Vm materialization into device-to-host transfer, row assembly, and
  public record construction.
- Compare full Vm and probe Vm with the same solve path to isolate recording
  output cost.
- Keep observer-only as a contrast case, not as the only target.

## What Not To Do Yet

- Do not change default time-chunk policy yet.
- Do not add a backend-specific adaptive policy yet.
- Do not optimize only observer-only paths.
- Do not start high-level recruitment amplitude micro-batching yet.
- Do not make speed claims from these single-repeat matrices.

## Suggested Next Step

Add deeper low-level timing spans in the JAX execution path and curve runtime
setup path, then regenerate the same visual report on a smaller confirmation
matrix. The target is to turn broad groups like `kernel` and `curve/setup` into
actionable sub-stages before changing behavior.
