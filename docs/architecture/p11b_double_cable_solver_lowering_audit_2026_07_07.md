# P11B Double-Cable Solver Lowering Audit

Date: 2026-07-07

Commit: `e1101ab9d8b126457b393b7e9452fa34c7e6d24c`

This note records the first Kaggle GPU lowering/codegen audit for
`benchmark/analysis/double_cable_solver_lowering_audit.py`. The goal was to
inspect current GPU PCR/SoA lowering before adding any new double-cable solver
route.

## Run Shape

- Platform image: Kaggle P100 image.
- GPU hardware: Tesla P100-PCIE-16GB.
- OS: Linux 6.12.90 x86_64.
- Python: 3.12.13.
- JAX: 0.10.2.
- Workload: real prepared double-cable observer-only VmRaster inputs.
- Requested shape: `Naxons=512`, `Nx=101`.
- Actual kernel shape: `kernel_group_size=512`, `actual_nx=89`.
- Precision: fp32.
- Device: `cuda:0`.
- Compiled IR: optimized HLO from XLA GPU compilation.

Valid artifact roots:

- Different diameters:
  `benchmark/results/kaggle/20260707_141000_solver_lowering_diff_gpu_512/outputs/extracted`
- Same diameter:
  `benchmark/results/kaggle/20260707_141000_solver_lowering_same_gpu_512/outputs/extracted`

## Important Context

The two runs are not identical membrane/compiler cases:

- Different diameters used `GatedLeakStackMembraneBackend`,
  `JaxMembraneProgram`, `uses_generated_model_step=true`,
  `unique_row_diameters=5`, and `shared_coefficients=false`.
- Same diameter used `HeterogeneousMembraneBackend`,
  `HeterogeneousMembraneModel`, `uses_generated_model_step=false`,
  `unique_row_diameters=1`, and `shared_coefficients=true`.

That means the isolated `block_solve` lowering is the cleanest comparison for
PCR/SoA layout. The `one_step_proxy` lowering is still useful, but it mixes
solver, membrane backend, conductance terms, system assembly, and observer
boundary effects.

## Optimized HLO Counts

### Isolated Block Solve

| diameters | variant | lines | gather | broadcast | select | fusion | transpose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| different | `pcr_soa_batched` | 2267 | 184 | 80 | 117 | 7 | 0 |
| different | `pcr_soa_vmap` | 2271 | 184 | 86 | 117 | 7 | 0 |
| same | `pcr_soa_batched` | 2279 | 184 | 84 | 117 | 7 | 0 |
| same | `pcr_soa_vmap` | 2286 | 184 | 90 | 117 | 7 | 0 |

### One-Step Proxy

| diameters | variant | lines | gather | broadcast | select | fusion | transpose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| different | `pcr_soa_real` | 2862 | 182 | 130 | 125 | 10 | 0 |
| same | `pcr_soa_real` | 2999 | 186 | 147 | 125 | 19 | 3 |

## Reading

The isolated GPU PCR/SoA solver paths are almost structurally identical after
XLA optimization. `pcr_soa_batched` is slightly smaller than `pcr_soa_vmap`,
mostly through fewer broadcasts and a few fewer lines, but gather/select/fusion
counts are the same and no transpose remains in the optimized solver HLO.

Same-diameter coefficient sharing does not simplify the isolated optimized
PCR/SoA block solver at this shape. The optimized HLO counts remain within the
same small band as the different-diameter case.

The `one_step_proxy` differs more, especially for fusion count and a few
remaining transposes in the same-diameter run. Because that run also changes
the membrane backend and generated-model status, this should be treated as a
compiler/backend signal rather than a pure solver signal.

## Decision

Do not add a new `pcr_soa_vmap` versus `pcr_soa_batched` runtime route. The
current batch-native PCR/SoA path is a reasonable active GPU route and has no
obvious high-level lowering defect relative to the vmap shape.

The next low-level work should inspect the code generated inside the current
PCR/SoA implementation and the one-step composition around it, not add a new
public or policy route:

1. Drill into the current PCR/SoA HLO/fusion bodies and memory layout.
2. Add a CPU/generated-membrane lowering view for the different-diameter case,
   because CPU stage profiles still show membrane and assembly costs.
3. Prototype solver changes only as benchmark-only candidates with correctness
   gates, then validate in real double-cable curve benchmarks before any
   runtime policy change.

