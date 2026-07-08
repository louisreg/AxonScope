# P11C PTA Block-Thomas GPU Gate

Date: 2026-07-08

This note adds a narrow PTA/block-Thomas gate to P11C. It corrects the main
ambiguity in the current P11C evidence: the first P11C synthetic run tested a
tiled/padded JAX PCR-SoA candidate, not a true GPU-optimized Parallel Thomas
Algorithm candidate.

## Why This Gate Exists

The literature synthesis points to a different implementation class from the
current JAX candidates:

```text
PTA_BLOCK_THOMAS_2X2_TILED
```

This is not the same as:

- the current CPU/reference Thomas route;
- `solve_block_tridiagonal_2x2_scalar_batched`, which is a JAX/XLA scan over
  `Nx`;
- the current GPU `pcr_soa_batched` route;
- the first P11C `large_population_exact_double_cable_jax` candidate, which
  still delegates to PCR-SoA bodies after padding/tiling.

The P11C GPU run at commit `ed0ccff` showed a small synthetic win for short
systems (`Nx=47` and `Nx=89`) and a loss for `Nx=129`. That result is useful
layout evidence, but it does not reject a PfSolve-style PTA solver.

The follow-up JAX feasibility run at commit `b4f4618` added
`thomas_batched_scan` to the same synthetic solver matrix. It changed the
interpretation: the JAX scan is still poor at `Naxons=1024`, close but usually
slower at `Naxons=4096`, then becomes the fastest synthetic solver at
`Naxons=8192` and `16384` for all tested `Nx=47/89/129` and shared/batched
coefficient modes.

## Reference Reading

### PfSolve / Parallel Factorization Solver

Source:

- DOI: `10.1145/3716171`
- ETH page: https://www.research-collection.ethz.ch/items/fab84ea7-24e3-48ec-bd12-85477e1ef4e3

Useful facts for AxonScope:

- GPU tridiagonal and bidiagonal solver library.
- Based on Parallel Thomas Algorithm.
- Uses warp-level instructions and occupancy optimizations.
- Reports very low extra memory overhead.
- Targets many systems that fit in single-GPU memory.

AxonScope translation:

- Do not treat plain JAX `lax.scan` Thomas as the final PTA experiment.
- A serious PTA candidate needs explicit tile/lane ownership, scratch layout,
  and memory coalescing control.
- PfSolve is scalar tridiagonal, while AxonScope needs exact `2x2`
  block-tridiagonal solves, so it is design inspiration rather than a direct
  library drop-in.

### NVIDIA GTC 2009 Tridiagonal Solvers On GPU

Source:

- https://www.nvidia.com/content/gtc/documents/1058_gtc09.pdf

Useful facts for AxonScope:

- The presented pattern solves many independent tridiagonal systems.
- A simple mapping can put one thread on one system.
- Memory access dominates: uncoalesced sweep directions are much slower.
- Reordering data to use the coalesced direction produced large gains in the
  reported fluid-simulation example.

AxonScope translation:

- The key question is not only Thomas versus PCR.
- The key question is whether adjacent GPU lanes read adjacent axons at the
  same `Nx` index during forward/backward sweeps.
- The natural layout to test is `[tile, Nx_pad, BLOCK_B]`, with `BLOCK_B`
  mapped to lanes/threads.

### Pipelined TDMA

Source:

- https://arxiv.org/abs/2509.03933

Useful facts for AxonScope:

- Batch size is a performance parameter.
- Larger batches improve occupancy, but too-large batches can hurt pipeline
  efficiency.
- The paper is multi-GPU and flow-solver oriented, so it is not a direct
  AxonScope implementation plan.

AxonScope translation:

- Keep testing `Naxons=1024/4096/8192/16384+`.
- Treat `B_effective` as part of the solver gate.
- Do not decide from a single small-batch run.

### Hines GPU Solver

Source:

- https://arxiv.org/abs/1810.12742

Useful facts for AxonScope:

- The direct Hines tree algorithm is not the AxonScope target.
- The relevant lessons are fine-grained parallelism, work balancing, and
  setup-time layout decisions.

AxonScope translation:

- Bucket by `Nx`.
- Avoid mixing very different sizes in one tile.
- Keep layout decisions in backend-owned preparation objects.

### PaScaL_TDMA

Source:

- https://github.com/xccels/PaScaL_TDMA

Useful facts for AxonScope:

- CUDA implementation uses explicit kernels and shared-memory oriented
  movement.
- Multi-GPU/MPI scope is broader than current AxonScope needs.

AxonScope translation:

- Useful later for a CUDA/FFI or custom-kernel track.
- Not a Phase P11C-JAX dependency.

## Exact Candidate Shape

The candidate is:

```text
PTA_BLOCK_THOMAS_2X2_TILED
```

Internal layout:

```text
a00[tile, x, lane]
a01[tile, x, lane]
a10[tile, x, lane]
a11[tile, x, lane]
off0[tile, edge, lane]
off1[tile, edge, lane]
rhs0[tile, x, lane]
rhs1[tile, x, lane]
```

where:

```text
x    = 0..Nx_pad-1
lane = 0..BLOCK_B-1
tile = ceil(Naxons / BLOCK_B)
```

Padding rows remain electrically neutral:

```text
D   = I_2
L/U = 0
rhs = [0, 0]
```

Batch padding lanes also use neutral independent systems.

## Algorithm

Forward sweep:

```text
C_0 = inv(D_0) U_0
d_0 = inv(D_0) rhs_0

for x = 1..Nx_pad-1:
    M_x = D_x - L_x C_{x-1}
    C_x = inv(M_x) U_x
    d_x = inv(M_x) (rhs_x - L_x d_{x-1})
```

Backward sweep:

```text
x_{Nx-1} = d_{Nx-1}

for x = Nx_pad-2..0:
    sol_x = d_x - C_x sol_{x+1}
```

For AxonScope double-cable, `L` and `U` are diagonal across the two rails in
the current exact system structure, while `D` is a dense local `2x2` block.
Therefore all local algebra should stay scalarized:

```text
det = m00 * m11 - m01 * m10
inv00 =  m11 / det
inv01 = -m01 / det
inv10 = -m10 / det
inv11 =  m00 / det
```

No `[2, 2]` dense matrices should be materialized in the hot path.

## JAX Feasibility Boundary

The current `solve_block_tridiagonal_2x2_scalar_batched` already expresses the
same block-Thomas recurrence with `lax.scan`. It is useful as a JAX feasibility
baseline, but it is not a PfSolve-style PTA implementation:

- XLA owns the scan loop and scratch.
- There is no explicit warp/lane mapping.
- There is no shared-memory control.
- The previous P100 gate showed compact HLO but poor hot time versus PCR-SoA.

Therefore:

```text
bad JAX scan timing rejects only the JAX/XLA expression,
not the custom-kernel PTA idea.
```

P11C should use JAX for correctness and negative/feasibility evidence only.
If the JAX scan remains much slower at large `Naxons`, the next serious PTA
step is a future custom-kernel design, not more JAX scan variants.

## P11C-PTA Gates

### Gate 1 - JAX Large-Population Negative Baseline

Add `thomas_batched_scan` to the P11C synthetic profiler and compare:

```text
current_pcr_soa
thomas_batched_scan
large_population_exact_double_cable_jax
```

Matrix:

```text
Naxons = 1024, 4096, 8192, 16384
Nx     = 47, 89, 129
dtype  = fp32
coeffs = shared, batched
```

Expected interpretation:

- If `thomas_batched_scan` remains slower by multiple factors, mark JAX PTA as
  not promising.
- If large `Naxons` unexpectedly closes the gap, run a real-stage gate before
  making any runtime decision.

Result:

- Artifact root:
  `benchmark/results/kaggle/20260708_181129_large_population_double_cable_solver_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-pta-jax-gpu-b4f4618/extracted`.
- Hardware: Kaggle `Tesla P100-PCIE-16GB`.
- Commit: `b4f4618`.
- `thomas_batched_scan` is slower than current PCR-SoA at `Naxons=1024`
  (`2.5x-3.5x` depending on `Nx` and coefficient mode).
- At `Naxons=4096`, `thomas_batched_scan` is still slower, but the gap shrinks
  to roughly `1.1x-1.3x`.
- At `Naxons=8192`, `thomas_batched_scan` is the fastest solver in all tested
  shapes. It is about `0.59x-0.92x` of current PCR-SoA time.
- At `Naxons=16384`, `thomas_batched_scan` becomes clearly faster: about
  `0.39x-0.57x` of current PCR-SoA time, with throughput around
  `392M-470M node-solves/s`.
- The tiled/padded P11C PCR-SoA candidate remains mildly useful for short
  `Nx=47/89` at smaller `Naxons`, but it does not scale like the Thomas scan at
  `Naxons=8192/16384` and remains hurt by `Nx=129 -> Nx_pad=160`.

Interpretation:

```text
Large Naxons do close the JAX Thomas scan gap.
The previous P11B rejection at Naxons=512 does not apply to large-population
solver-only regimes.
This is not yet a runtime decision because the result is synthetic solver-only.
```

### Gate 2 - Real-Stage Large-Population Gate

After Gate 1:

- Keep current GPU runtime on PCR-SoA unless the real-stage evidence changes.
- Because JAX `thomas_batched_scan` wins synthetic large-population cases at
  `Naxons >= 8192`, build a real-stage large-population gate before accepting
  or rejecting the JAX expression.
- Keep `PTA_BLOCK_THOMAS_2X2_TILED` as a future custom-kernel candidate
  regardless of the JAX result, because PfSolve-style lane/scratch control is a
  different implementation class.
- Do not add a public solver option or `auto` policy branch.

Result at `Naxons=8192`:

- Artifact root:
  `benchmark/results/kaggle/20260708_182559_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-real-pta-gpu-n8192-b233fc7/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_182559`.
- Hardware: Kaggle `Tesla P100-PCIE-16GB`.
- Commit: `b233fc7`.
- Workload: real double-cable MRG-like prepared inputs, observer-only output,
  different diameters, fp32, target `Nx=101`, actual kernel `Nx=89`.
- `block_solve/thomas_batched_scan` is `2.669 ms` mean versus
  `3.796 ms` for `pcr_soa_batched`, or `0.703x` of PCR time
  (`1.42x` speedup).
- `one_step_proxy/thomas_batched_scan_real` is `3.196 ms` mean versus
  `4.350 ms` for `pcr_soa_batched_real`, or `0.735x` of PCR time
  (`1.36x` speedup).
- `one_step_proxy/thomas_batched_scan_real_precomputed_static` is
  `3.117 ms` mean versus `4.385 ms` for the matching PCR route, or
  `0.711x` of PCR time (`1.41x` speedup).
- The Thomas block solve is `85.7%` of the primary one-step proxy, so this
  large-population real-stage case is genuinely solver-bound.

Result at `Naxons=16384`:

- Artifact root:
  `benchmark/results/kaggle/20260708_183319_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-real-pta-gpu-16k-b233fc7-r2/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_183319`.
- Hardware: Kaggle `Tesla P100-PCIE-16GB`.
- Commit: `b233fc7`.
- Workload: same as the `Naxons=8192` gate, with target `Nx=101`,
  actual kernel `Nx=89`, observer-only output, different diameters, fp32.
- `block_solve/thomas_batched_scan` is `3.287 ms` mean versus
  `7.223 ms` for `pcr_soa_batched`, or `0.455x` of PCR time
  (`2.20x` speedup).
- `one_step_proxy/thomas_batched_scan_real` is `4.247 ms` mean versus
  `8.285 ms` for `pcr_soa_batched_real`, or `0.513x` of PCR time
  (`1.95x` speedup).
- `one_step_proxy/thomas_batched_scan_real_precomputed_static` is
  `4.104 ms` mean versus `8.241 ms` for the matching PCR route, or
  `0.498x` of PCR time (`2.01x` speedup).
- The Thomas block solve is `80.1%` of the primary one-step proxy, so the
  larger real-stage case remains solver-bound.

Interpretation:

```text
The synthetic large-Naxons Thomas signal survives realistic GPU hot-step gates
at Naxons=8192 and Naxons=16384.
The signal gets stronger at 16384 axons.
This is enough to justify a backend-private large-population implementation
experiment, but still not a public solver policy or automatic runtime branch.
```

### Gate 3 - Future Custom-Kernel Entry Criteria

Only open a custom-kernel PTA track if all are true:

- real large-population double-cable workloads are solver-bound on GPU;
- `Nx <= 96` or nearby buckets remain a dominant target;
- current PCR-SoA is still the bottleneck after preparation/result costs are
  controlled;
- the implementation can be generic to the exact double-cable system, not
  membrane-model-specific.

## jax-triton Tile/Lane Candidate

After the `Naxons=8192/16384` real-stage gates, P11C adds a benchmark-only
`jax_triton_tiled_thomas` candidate. It is a first implementation step toward
the GPU-tridiagonal papers, not a final PfSolve-equivalent solver:

- it calls Triton kernels through `jax_triton.triton_call`;
- it exposes an internal `[Nx, B]` layout so lanes in one program read adjacent
  axons at the same compartment index;
- one Triton program owns a tile of axons and runs the exact block-Thomas
  recurrence over `x`;
- it keeps global-memory forward scratch for now and does not yet implement the
  shared-memory/scratch scheduling, occupancy tuning, or full PTA implementation
  class described by PfSolve-style papers.

Gate interpretation:

```text
If jax_triton_tiled_thomas wins, implement the backend-private large-population
route around that layout.
If it loses to the JAX scan, keep the pure JAX large-Naxons route as the current
best low-level evidence and leave deeper Triton/CUDA work as a later track.
```

Kaggle P100 result at commit `f915367`, real-stage double-cable, fp32,
observer-only, target `Nx=101`, actual kernel `Nx=89`:

- Artifact root for `Naxons=8192`:
  `benchmark/results/kaggle/20260708_185401_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-thomas-gpu-8k-f915367/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_185403`.
- Artifact root for `Naxons=16384`:
  `benchmark/results/kaggle/20260708_185428_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11c-triton-thomas-gpu-16k-f915367/outputs/benchmark/results/double_cable_real_stage_profile_quick_gpu_20260708_185428`.

Warm hot-step result:

| Naxons | block solve PCR | block solve JAX scan | block solve Triton | Triton vs PCR | Triton vs JAX scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 3.766 ms | 2.748 ms | 0.678 ms | 5.55x faster | 4.05x faster |
| 16384 | 7.148 ms | 3.186 ms | 0.878 ms | 8.14x faster | 3.63x faster |

Warm one-step proxy result:

| Naxons | one-step PCR | one-step JAX scan | one-step Triton | Triton vs PCR | Triton vs JAX scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.397 ms | 3.141 ms | 1.519 ms | 2.89x faster | 2.07x faster |
| 16384 | 8.111 ms | 4.209 ms | 2.421 ms | 3.35x faster | 1.74x faster |

Constraint:

- The first standalone Triton block-solve call is extremely expensive on the
  Kaggle P100 runs: about `2.43e6-2.46e6 ms`, roughly `40-41 min`.
- Later one-step proxy first runs are much lower because the Triton kernels have
  already been compiled and cached inside the process.

Interpretation:

```text
jax_triton_tiled_thomas is the strongest warm large-population GPU solver
candidate so far.
It cannot be promoted until the cold compilation/cache behavior is understood
and controlled.
After Triton, non-solver work is again a major share of the one-step proxy, so
the next optimization target may move back to assembly/membrane/layout costs.
```

Numerical validation caveat:

- The first jax-triton Kaggle timing gate validated runtime execution and timing
  only. It did not yet compare Triton `Vi`/`Ve`/`Vm`/gates to a trusted
  non-Triton reference on the real assembled double-cable system.
- `benchmark/analysis/double_cable_real_stage_profile.py` now has a
  `--validate-solvers` mode for that check. It runs after timing measurements,
  writes `real_stage_validation.csv`, compares block-solve and one-step proxy
  outputs against a reference such as `thomas_batched_scan`, records
  `Vm = Vi - Ve` differences and block residual norms, and returns a non-zero
  status if validation fails.
- P11C should run this small numerical gate on Kaggle P100 before interpreting
  the warm Triton timing as physically coherent, then investigate the
  `40-41 min` first standalone Triton compile/run cost.

## CPU Synthetic Cross-Check

The long-running CPU synthetic P11C run on commit `ed0ccff` eventually finished:

- Artifact root:
  `benchmark/results/kaggle/20260708_174928_large_population_double_cable_solver_profile_quick_cpu_NvidiaTeslaP100_axonscope-p11c-large-pop-cpu-ed0ccff/outputs/benchmark/results/large_population_double_cable_solver_profile_quick_cpu_20260708_174928`.
- Hardware context: AxonScope CPU path on a Kaggle P100 machine.
- Matrix: `B=1024/4096/8192`, `Nx=47/89/129`, shared and batched coefficients,
  fp32.

Result:

- The tiled/padded large-population JAX candidate wins only for short
  `Nx=47`, with modest gains around `3.5%-9.8%` depending on `B` and coefficient
  mode.
- For `Nx=89`, current PCR-SoA is generally best; the only near tie is
  `B=4096`, shared coefficients, where the tiled route is effectively equal.
- For `Nx=129`, current PCR-SoA is clearly best; the padded `Nx=160` candidate
  loses by roughly `14%-27%`.

Interpretation:

```text
Do not move CPU execution toward the P11C tiled/padded route.
P11C remains a GPU large-population track.
```

CPU/GPU synthetic speedup, same commit `ed0ccff`, best available variant on
each platform and averaged over shared/batched coefficients:

| Naxons | Nx=47 | Nx=89 | Nx=129 |
| ---: | ---: | ---: | ---: |
| 1024 | 460x | 1072x | 1191x |
| 4096 | 967x | 1647x | 2185x |
| 8192 | 1364x | 2238x | 2719x |

Directional CPU-vs-GPU comparison using the later GPU `b4f4618` run with
`thomas_batched_scan` included shows why large-population GPU Thomas matters:

| Naxons | Nx=47 | Nx=89 | Nx=129 | GPU best variant |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 392x | 1047x | 1200x | tiled/PCR mix |
| 4096 | 937x | 1561x | 2144x | tiled/PCR mix |
| 8192 | 1357x | 2954x | 4357x | `thomas_batched_scan` |

These tables are solver-only, and the Thomas table is cross-run/cross-commit,
so use them as scaling signals rather than end-to-end workflow claims.

## Immediate Decision

P11C should continue from the benchmark-only gate to a backend-private
large-population implementation experiment. The `Naxons=8192` and `16384`
real-stage GPU runs are strong enough to justify integration work, but not
enough to change public runtime policy or add a public solver option.
