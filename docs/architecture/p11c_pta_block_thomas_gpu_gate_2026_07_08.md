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

### Gate 2 - PTA Design Decision

After Gate 1:

- Keep current GPU runtime on PCR-SoA unless the real-stage evidence changes.
- If JAX PTA is slow but short-`Nx` P11C results remain promising, keep
  `PTA_BLOCK_THOMAS_2X2_TILED` as a future custom-kernel candidate.
- Do not add a public solver option or `auto` policy branch.

### Gate 3 - Future Custom-Kernel Entry Criteria

Only open a custom-kernel PTA track if all are true:

- real large-population double-cable workloads are solver-bound on GPU;
- `Nx <= 96` or nearby buckets remain a dominant target;
- current PCR-SoA is still the bottleneck after preparation/result costs are
  controlled;
- the implementation can be generic to the exact double-cable system, not
  membrane-model-specific.

## Immediate Decision

P11C should not conclude from the current tiled PCR-SoA candidate alone. The
next cheap step is to run the large-population synthetic profiler with
`thomas_batched_scan` included. If that repeats the P11B result at larger
`Naxons`, the JAX PTA line can be closed while keeping PfSolve-style PTA as a
future custom-kernel candidate.
