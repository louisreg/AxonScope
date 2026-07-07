# P11B Real Double-Cable Stage Profile

Date: 2026-07-07

Commit: `4053985fa5ce8f2eb26942dd7fbb905ea181acc9`

This note records the first Kaggle CPU/GPU run of
`benchmark/analysis/double_cable_real_stage_profile.py`. The profiler is a
benchmark-only introspection tool. It builds public AxonScope double-cable
workloads, reuses the current backend/runtime preparation, and measures hot
one-step JAX stages. It does not define runtime policy.

## Run Shape

- Platform image: Kaggle P100 image for both CPU-path and GPU-path runs.
- GPU hardware: Tesla P100-PCIE-16GB.
- OS: Linux 6.12.90 x86_64.
- Workload: double-cable MRG population, observer-only VmRaster.
- Requested shape: `Naxons=128`, `Nx=101`.
- Actual kernel shape: `kernel_group_size=128`, `actual_nx=89`.
- Diameters: `different_diameters`.
- Coefficients: `shared_coefficients=false`.
- Extracellular input: `factorized_footprint`.
- Precision: fp32.
- Repeats/warmups: 5 measured repeats, 1 warmup.

Artifact roots:

- CPU: `benchmark/results/kaggle/20260707_130525_double_cable_real_stage_profile_quick_cpu_NvidiaTeslaP100/outputs/extracted_cpu`
- GPU: `benchmark/results/kaggle/20260707_130525_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100/outputs/extracted_gpu`

## Hot Mean Timings

Measured mean times are in milliseconds. First-run times include compilation
and are preserved in the CSV/report artifacts, but the table below focuses on
hot measured repeats.

| stage | variant | CPU mean ms | GPU mean ms | CPU/GPU |
| --- | --- | ---: | ---: | ---: |
| block_solve | thomas_vmap | 0.682 | 2.077 | 0.33 |
| block_solve | pcr_matrix_vmap | 6.962 | 0.468 | 14.88 |
| block_solve | pcr_soa_vmap | 27.838 | 0.381 | 72.99 |
| block_solve | pcr_soa_batched | 28.087 | 0.372 | 75.40 |
| one_step_proxy | active solver | 1.542 | 0.465 | 3.32 |
| membrane_gate_update | GatedLeakStackMembraneBackend | 0.616 | 0.210 | 2.93 |
| membrane_conductance_terms | GatedLeakStackMembraneBackend | 0.353 | 0.223 | 1.58 |
| system_assembly | real_double_cable | 0.204 | 0.274 | 0.75 |
| extracellular_rhs_drive | factorized_footprint | 0.068 | 0.216 | 0.32 |
| observer_write | vm_raster_real | 0.090 | 0.238 | 0.38 |

Active solver selection:

- CPU resolved `auto` to `thomas`.
- GPU resolved `auto` to `pcr_soa`.

## Reading

CPU:

- Thomas remains the only sensible low-level block solver for this shape.
- PCR/SoA variants are very slow on CPU in this real prepared system.
- The active one-step proxy is 1.542 ms. Thomas solve alone is 0.682 ms, so
  the solve is a major cost but not the only CPU cost.
- Generated membrane work is visible: gate update plus conductance terms are
  about 0.969 ms when measured as separate jitted stages. These numbers are
  not additive with the fused one-step proxy, but they point at compiler/model
  work as a relevant CPU optimization axis.

GPU:

- PCR/SoA is clearly the relevant GPU solve family for this shape.
- Thomas is much slower on GPU than PCR/SoA.
- Best GPU solve is `pcr_soa_batched` at 0.372 ms. The active one-step proxy is
  0.465 ms, so hot one-step execution is now heavily solver-sensitive.
- Separate non-solver stages are still the same order of magnitude as the
  solver when measured as individual kernels: system assembly 0.274 ms,
  observer write 0.238 ms, factorized RHS drive 0.216 ms, gate update 0.210 ms.
  These separate measurements include launch boundaries and are not additive
  with the one-step proxy.

## Implications

- The latest evidence supports optimizing low-level double-cable solver code,
  especially GPU PCR/SoA layout and fusion choices.
- It is still too early to make a high-level policy decision. This profiler
  isolates one-step stages, while realistic curve benchmarks still measure the
  full workflow, including preparation, launch/finalize, recording, and result
  materialization.
- The next low-level pass should keep two tracks visible:
  1. Solver internals: PCR/SoA implementation, memory layout, batching, and
     generated code shape.
  2. Compiler/model internals: generated membrane gate/conductance work and
     whether it can be made cheaper without adding runtime-specific branches.

## Next Benchmark Step

Before implementing a new solver route, run one larger shape and one
same-diameter shape with the same profiler:

- `Naxons=512`, `Nx=101`, `different_diameters`, observer-only.
- `Naxons=512`, `Nx=101`, `same_diameter`, observer-only.

This should tell whether the current conclusion is stable as batch size grows
and whether coefficient sharing changes the bottleneck map enough to affect
implementation priority.
