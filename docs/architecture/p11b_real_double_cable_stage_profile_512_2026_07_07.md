# P11B Real Double-Cable Stage Profile at Naxons 512

Date: 2026-07-07

Commit: `6cdb966`

This note records the clean v2 Kaggle CPU/GPU follow-up for
`benchmark/analysis/double_cable_real_stage_profile.py` at `Naxons=512`. The
goal is to check whether the `Naxons=128` bottleneck reading survives a larger
batch and whether same-diameter coefficient sharing changes the optimization
priority.

## Run Hygiene

An earlier parallel submission used the same auto-generated local run
directory for multiple same-campaign jobs and mixed local Kaggle packages.
Those first attempts are ignored. The valid artifacts are the v2 runs below,
each launched with an explicit unique `--run-dir`.

The Kaggle launcher now includes the slug in auto-generated run directories to
prevent this class of collision in future parallel launches.

## Valid Artifacts

- Different diameters, CPU path:
  `benchmark/results/kaggle/20260707_133000_real_stage_diff_cpu_512_v2/outputs/extracted`
- Different diameters, GPU path:
  `benchmark/results/kaggle/20260707_133000_real_stage_diff_gpu_512_v2/outputs/extracted`
- Same diameter, CPU path:
  `benchmark/results/kaggle/20260707_134000_real_stage_same_cpu_512_v2/outputs/extracted`
- Same diameter, GPU path:
  `benchmark/results/kaggle/20260707_134000_real_stage_same_gpu_512_v2/outputs/extracted`

Shared run shape:

- Platform image: Kaggle P100 image.
- Workload: double-cable MRG population, observer-only VmRaster.
- Requested shape: `Naxons=512`, `Nx=101`.
- Actual kernel shape: `kernel_group_size=512`, `actual_nx=89`.
- Extracellular input: `factorized_footprint`.
- Precision: fp32.
- Repeats/warmups: 5 measured repeats, 1 warmup.

## Stage Means

Measured mean times are in milliseconds. First-run compile times remain in the
CSV artifacts; this table focuses on hot measured repeats.

| case | stage | variant | mean ms |
| --- | --- | --- | ---: |
| diff CPU | one_step_proxy | thomas_real | 4.318 |
| diff CPU | block_solve | thomas_vmap | 1.864 |
| diff CPU | membrane_gate_update | GatedLeakStackMembraneBackend | 1.394 |
| diff CPU | membrane_conductance_terms | GatedLeakStackMembraneBackend | 1.259 |
| diff CPU | system_assembly | real_double_cable | 0.638 |
| diff CPU | extracellular_rhs_drive | factorized_footprint | 0.145 |
| diff CPU | observer_write | vm_raster_real | 0.142 |
| diff GPU | one_step_proxy | pcr_soa_real | 0.487 |
| diff GPU | block_solve | pcr_soa_vmap | 0.438 |
| diff GPU | block_solve | pcr_soa_batched | 0.463 |
| diff GPU | block_solve | thomas_vmap | 2.121 |
| diff GPU | system_assembly | real_double_cable | 0.284 |
| diff GPU | membrane_gate_update | GatedLeakStackMembraneBackend | 0.233 |
| diff GPU | membrane_conductance_terms | GatedLeakStackMembraneBackend | 0.223 |
| diff GPU | observer_write | vm_raster_real | 0.223 |
| diff GPU | extracellular_rhs_drive | factorized_footprint | 0.204 |
| same CPU | one_step_proxy | thomas_real | 2.502 |
| same CPU | block_solve | thomas_vmap | 1.699 |
| same CPU | membrane_gate_update | HeterogeneousMembraneBackend | 0.701 |
| same CPU | membrane_conductance_terms | HeterogeneousMembraneBackend | 0.587 |
| same CPU | system_assembly | real_double_cable | 0.377 |
| same CPU | observer_write | vm_raster_real | 0.179 |
| same CPU | extracellular_rhs_drive | factorized_footprint | 0.088 |
| same GPU | one_step_proxy | pcr_soa_real | 0.556 |
| same GPU | block_solve | pcr_soa_vmap | 0.439 |
| same GPU | block_solve | pcr_soa_batched | 0.441 |
| same GPU | block_solve | thomas_vmap | 2.167 |
| same GPU | system_assembly | real_double_cable | 0.281 |
| same GPU | membrane_conductance_terms | HeterogeneousMembraneBackend | 0.260 |
| same GPU | membrane_gate_update | HeterogeneousMembraneBackend | 0.226 |
| same GPU | observer_write | vm_raster_real | 0.224 |
| same GPU | extracellular_rhs_drive | factorized_footprint | 0.205 |

Slow CPU-only solver variants remain diagnostic, not implementation targets:

- Different diameters CPU: `pcr_matrix_vmap` 24.319 ms, `pcr_soa_vmap`
  150.675 ms, `pcr_soa_batched` 155.670 ms.
- Same diameter CPU: `pcr_matrix_vmap` 25.841 ms, `pcr_soa_vmap` 158.464 ms,
  `pcr_soa_batched` 178.777 ms.

## Reading

CPU:

- Thomas remains the only sensible CPU block solver for this shape.
- Same-diameter coefficient sharing materially helps CPU hot one-step time:
  4.318 ms for different diameters versus 2.502 ms for same diameter.
- The CPU solve is important, but not the whole story. Different-diameter CPU
  one-step time is 4.318 ms while Thomas solve alone is 1.864 ms. Generated
  membrane work and assembly are large enough that compiler/model output
  should stay in the low-level optimization audit.
- Same diameter reduces measured gate, conductance, assembly, and RHS-drive
  stages, so diameter rounding/cohort grouping is a meaningful preparation
  lever. Keep it as a high-level grouping benefit, not as a runtime special
  path.

GPU:

- PCR/SoA remains the right GPU solver family. Thomas is about 2.1-2.2 ms,
  while PCR/SoA is about 0.44-0.46 ms.
- The fused active one-step proxy is close to the active PCR/SoA solve:
  0.487 ms versus 0.438-0.463 ms for different diameters, and 0.556 ms versus
  0.439-0.441 ms for same diameter. This is the strongest evidence so far that
  GPU low-level work should start with the solver.
- Separate non-solver stages are still around 0.20-0.28 ms when measured as
  individual JAX kernels. Treat those as launch-bound stage probes, not as
  additive costs inside the fused one-step proxy.
- Same-diameter sharing does not materially improve GPU one-step time at this
  shape. The GPU target should stay inside PCR/SoA layout/fusion and generated
  solver code rather than a new high-level runtime policy.

## Decision For The Next Pass

The evidence now supports starting a low-level solver/compiler inspection
pass, with this ordering:

1. Inspect current GPU PCR/SoA generated code and JAX lowering for the
   double-cable one-step proxy.
2. Compare `pcr_soa_vmap` versus `pcr_soa_batched` layout and memory behavior
   before adding any new solver route.
3. In parallel, inspect generated membrane gate/conductance code for the CPU
   different-diameter case, because compiler/model output is still a visible
   cost there.
4. Keep high-level cohort grouping and amplitude batching as later policy
   work. They are useful, but they are not the immediate low-level target.
