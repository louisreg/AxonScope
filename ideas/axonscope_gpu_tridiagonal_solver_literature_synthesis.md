# AxonScope GPU Double-Cable Solver: Literature Synthesis and Implementation Plan

## Executive summary

The additional tridiagonal-solver resources strongly reinforce a practical conclusion for AxonScope:

> For `Nx = 30-100` and `B > 500`, the highest-return path is probably not a theoretically exotic solver first. It is a **many-small-systems GPU strategy**: batched/block Thomas with excellent memory layout, larger coalesced batches, and only then PCR/associative/Pallas if profiling proves they win.

AxonScope's double-cable linear system is not a scalar tridiagonal system; it is a specialized **2x2 block-tridiagonal** system. However, the GPU literature on many tridiagonal systems maps very well to AxonScope because the workload is also many independent short systems.

Recommended priority order:

```text
Phase 0   Baseline, profiling, correctness, memory layout experiments
Phase 1.1 PTA-style batched block-Thomas 2x2
Phase 1.2 Layout/coalescing benchmark: [B,Nx] vs [Nx,B] vs tiled [Nx_pad,BLOCK_B]
Phase 1.3 Dispatch scheduler: bucket/coalesce groups, increase B_effective
Phase 1.4 PCR_SOA batch-native exact solver
Phase 1.5 Split two-rail fixed-K iterative solver
Phase 2   Associative scan experiments
Phase 3   Pallas or FFI custom kernel only if JAX leaves performance on the table
```

The key update from the new references is this:

> For many short systems, a well-implemented Thomas-like method can beat more parallel algorithms because it does less arithmetic and can be made efficient with warp-level/tile-level batching and coalesced memory access.

This is especially relevant because AxonScope's target `Nx` is small.

---

## AxonScope target problem

### Workload

```text
Model: exact double-cable axon solver
System per time step: 2x2 block-tridiagonal
Cable length: Nx = 30-100
Batch size: B > 500 fibers
Typical use: threshold, activation, recruitment, conduction studies
Backend: JAX on GPU, possibly TPU, possibly future Pallas/FFI
```

### Linear system form

At each implicit time step, the double-cable system can be written as:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = r_i

x_i = [Vi_i, Ve_i]^T
D_i = 2x2 local block
L_i, U_i = off-diagonal 2x2 blocks, often diagonal in the two rails
```

Equivalently, if reordered by rail:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

where `Ti` and `Te` are scalar tridiagonal cable operators and `Cie`, `Cei` are local couplings.

### Current bottleneck

The current block Thomas solver has two spatial passes:

```text
forward elimination:     i = 0 -> Nx-1
backward substitution:   i = Nx-1 -> 0
```

This is stable and exact but exposes limited spatial parallelism. GPU utilization relies mostly on the batch axis `B`.

For `Nx = 30-100`, this may still be acceptable if `B_effective` is large and memory layout is optimal. That is exactly the insight reinforced by the new TDMA/PTA resources.

---

## Scientific motivation for preserving exact double-cable

The biological reason to keep an exact double-cable solver is not just numerical conservatism. Abdollahi & Prescott (2024) show that conduction velocity, reliability, and ephaptic interactions depend on how current partitions between axial, submyelin, transmyelin, and extramyelin pathways.

Important takeaways:

- The adapted MRG model includes nodes, paranodes, juxtaparanodes, and internodes, with active conductances at nodes.
- Condition 1 corresponds to a conventional double-cable MRG model whose extramyelin layer is connected to ground.
- Conditions 2 and 3 disconnect internodal extramyelin compartments from ground and vary longitudinal extramyelin resistance.
- Condition 2 conducts faster than Condition 1 and Condition 3 because less current is lost across the myelin/extracellular pathways and more axial current reaches the next node.
- The effect of intrinsic parameters, including myelin thickness, depends on extracellular boundary conditions.

This means pseudo-single-cable surrogates are useful for screening, but the exact double-cable path remains important for validation and for studying boundary-condition-dependent physiology.

---

## Literature synthesis

## 1. Parallel Thomas Algorithm / PTA for many independent systems

### Main idea

The Parallel Thomas Algorithm literature focuses on solving many independent tridiagonal systems efficiently on GPUs. The key is not necessarily to parallelize one single Thomas solve in `Nx`; it is to run many short Thomas solves in parallel and make memory access efficient.

This maps directly to AxonScope:

```text
PTA paper systems: many independent tridiagonal systems
AxonScope systems: many independent fibers, possibly fibers × amplitudes × configs
```

### Relevant lesson

For short systems, Thomas can be competitive or superior because:

```text
Thomas:
    low arithmetic work
    stable
    simple data dependencies
    good if many systems are available

PCR/CR:
    more parallel depth
    more arithmetic
    more temporary data movement
    may lose for small Nx
```

### AxonScope implication

Add a backend:

```python
DoubleCableLinearSolver.PTA_BLOCK_THOMAS
```

This should be an optimized batched block-Thomas solver, not a new mathematical solver.

Target implementation style:

```text
one tile/program/warp handles many fibers or one fiber depending backend
Nx is small and statically bucketed
2x2 operations are scalarized
coefficients are stored in SoA layout
memory access is coalesced
```

---

## 2. PfSolve / ACM 2025

### What it contributes

PfSolve is a GPU-oriented solver utility for bi-/tridiagonal systems. The ACM paper describes a high-performance GPU solution based on an optimized Parallel Thomas Algorithm with warp-level instructions and occupancy optimizations.

The public PfSolve repository also describes GPU bi-/tri-diagonal solvers based on Parallel Cyclic Reduction plus Thomas using single-warp GPU programming.

### Why it matters for AxonScope

PfSolve reinforces two ideas:

1. **Thomas-like algorithms are still highly relevant on GPU** when implemented for many systems with warp-level optimization.
2. **Hybrid PCR + Thomas** can be a strong practical strategy.

### AxonScope adaptation

PfSolve is scalar tridiagonal. AxonScope needs a specialized 2x2 block-tridiagonal version.

Direct integration may not be straightforward, but the design inspiration is strong:

```text
PfSolve scalar TDMA/PCR+Thomas
    -> AxonScope block-Thomas 2x2 PTA
    -> AxonScope block-PCR + block-Thomas hybrid
```

### Action item

Before investing heavily in associative scan or Pallas, benchmark:

```text
current block Thomas
PTA-style block Thomas [B,Nx]
PTA-style block Thomas [Nx,B]
PTA-style block Thomas tiled [Nx_pad,BLOCK_B]
PCR_SOA
hybrid PCR/Thomas
```

---

## 3. Pipelined-TDMA / arXiv 2025

### What it contributes

The Pipelined-TDMA paper targets scalable TDMA on multi-GPU systems. Its core message is relevant even for one GPU:

- Small TDMA systems can suffer from low GPU occupancy.
- Larger batches improve throughput.
- But excessively large batches can reduce pipeline efficiency or increase temporary-memory pressure.
- Kernel concurrency and overlapping non-scalable phases with scalable compute can improve utilization.

### AxonScope implication

This validates the dispatch/scheduler roadmap:

```text
increase B_effective
bucket Nx to 32/64/128
coalesce groups with compatible execution signatures
enqueue groups asynchronously only when useful
avoid full Vm outputs during throughput-critical runs
```

### Practical translation

Do not only improve the solver. Also improve the workload presented to the solver.

For AxonScope:

```text
B_effective = fibers × amplitudes × electrode configs × stimulation conditions
```

If each individual dispatch group is too small, the GPU may remain underoccupied even with a good solver.

---

## 4. NVIDIA GTC 2009 tridiagonal slides

### What it contributes

The NVIDIA tridiagonal-solver slides emphasize two practical GPU lessons:

1. Solving many independent tridiagonal systems on GPU is a natural pattern.
2. Memory coalescing can dominate performance.

The slides illustrate that different sweep directions can have very different performance depending on whether memory accesses are coalesced. Reordering data to make the sweep coalesced can yield large gains.

### AxonScope implication

The memory layout may matter as much as the algorithm.

Test these layouts:

```text
Layout A: [B, Nx]
    each fiber is contiguous

Layout B: [Nx, B]
    all fibers at the same compartment index are contiguous

Layout C: tiled [Nx_pad, BLOCK_B]
    good candidate for Pallas/FFI kernels
```

For Thomas-like batched updates, the loop is usually:

```python
for i in range(Nx):
    update all B fibers at spatial index i
```

That suggests `[Nx, B]` or tiled `[Nx_pad, BLOCK_B]` may give better coalescing than `[B, Nx]`.

This must be measured. Do not assume.

---

## 5. PaScaL_TDMA

### What it contributes

PaScaL_TDMA is a parallel and scalable TDMA library for many tridiagonal systems, with CPU/GPU and multi-GPU/MPI orientation. It uses modified Thomas algorithms, communication schemes, and CUDA-aware MPI for large PDE workloads.

### AxonScope implication

PaScaL_TDMA is not an obvious drop-in dependency for AxonScope because:

```text
it is PDE/HPC oriented
it is scalar TDMA oriented
it is multi-GPU/MPI oriented
AxonScope currently uses JAX and needs 2x2 block systems
```

But it is useful as a design reference for:

```text
many-system TDMA batching
multi-device sharding
communication-aware scheduling
future FFI/CUDA backend
```

### Action item

Keep PaScaL_TDMA as a reference for a future FFI backend, not as Phase 1.

---

## 6. CR, PCR, recursive doubling, hybrid solvers

The broader GPU tridiagonal literature compares:

```text
Thomas / TDMA
Cyclic Reduction (CR)
Parallel Cyclic Reduction (PCR)
Recursive Doubling (RD)
Hybrid CR/PCR/Thomas variants
```

For AxonScope:

```text
Thomas:
    best baseline, stable, low arithmetic, not Nx-parallel

PCR:
    exact, spatially parallel, more arithmetic and memory movement

Hybrid PCR/Thomas:
    likely attractive for Nx=30-100

Recursive doubling / associative scan:
    interesting but more complex/stability-sensitive
```

The key is that `Nx` is short. A more parallel algorithm is not automatically faster.

---

# Revised AxonScope solver roadmap

## Phase 0 — Baseline and profiling

### Goal

Understand whether the bottleneck is:

```text
solver arithmetic
gather/scatter temporaries
memory layout
kernel count
output materialization
Vext construction
dense zero Iinj
```

### Actions

Add solver-only benchmark:

```text
benchmark/solvers/bench_double_cable_linear_solvers.py
```

Matrix:

```text
B:      512, 1024, 2048, 4096, 8192
Nx:     32, 51, 64, 96, 100, 128
solver: current_thomas, pcr_soa, pta_block_thomas variants
dtype:  float32, float64
layout: BxNx, NxB, tiled
```

Add JAX trace script:

```text
benchmark/solvers/profile_double_cable_jax.py
```

Use:

```python
jax.profiler.trace(...)
jax.block_until_ready(out)
```

### Deliverables

```text
1. baseline solver-only table
2. end-to-end table
3. GPU trace for current Thomas
4. memory layout comparison
5. clear statement of whether solver or output dominates
```

---

## Phase 1.1 — PTA-style batched block-Thomas 2x2

### Goal

Implement the same exact block Thomas math, but optimized as many short independent systems.

### Backend name

```python
DoubleCableLinearSolver.PTA_BLOCK_THOMAS
```

### Implementation requirements

Use SoA representation:

```text
d00, d01, d10, d11       [B,Nx] or [Nx,B]
lower0, lower1           [B,Nx] or [Nx,B]
upper0, upper1           [B,Nx] or [Nx,B]
rhs0, rhs1               [B,Nx] or [Nx,B]
```

Avoid performance-critical arrays like:

```text
[B, Nx, 2, 2]
```

unless only used in a prototype.

### Variants to test

```text
PTA_BxNx
PTA_NxB
PTA_tiled_Nxpad_BLOCKB
```

### Why this comes before PCR

For `Nx=30-100`, the sequential Thomas depth is short. If `B_effective` is high and layout is good, Thomas may beat PCR because it does less work.

### Go/no-go

Keep as default GPU backend for small Nx if:

```text
PTA_BLOCK_THOMAS speedup >= 1.5x vs current Thomas
and correctness is identical within numerical tolerance
```

---

## Phase 1.2 — Layout and coalescing benchmark

### Goal

Determine the best internal memory layout.

### Layouts

```text
A. [B, Nx]
B. [Nx, B]
C. tiled [Nx_pad, BLOCK_B]
```

### Hypothesis

If the Thomas forward loop is:

```python
for i in range(Nx):
    update all B fibers at i
```

then `[Nx, B]` may produce more coalesced accesses than `[B, Nx]`.

### Test

For each solver backend:

```text
current Thomas
PTA block Thomas
PCR_SOA
split fixed-K
```

measure:

```text
solver-only time
memory bandwidth proxy
kernel count
HLO/trace behavior
```

### Decision

Pick one internal solver layout per backend. The public API can remain batch-first.

---

## Phase 1.3 — Dispatch scheduler and B_effective

### Goal

Increase the batch size seen by the GPU.

### Actions

Add execution buckets:

```text
mode
solver
Nx_pad bucket
dtype
recording mode
Iinj kind
Vext kind
membrane signature
```

Coalesce compatible groups:

```text
many small groups -> one larger JAX call
```

Use buckets:

```text
Nx <= 32 -> 32
Nx <= 64 -> 64
Nx <= 128 -> 128
```

Optionally enqueue multiple groups before waiting:

```text
async_groups=True
```

but only after coalescing is tested.

### Why this is part of solver optimization

The solver can only use the parallelism it is given. A short `Nx` system needs large `B_effective` to saturate the GPU.

### Go/no-go

Keep coalescing if:

```text
end-to-end speedup > 20%
JIT call count drops significantly
memory remains acceptable
```

Keep async scheduling if:

```text
end-to-end speedup > 10%
no memory pressure issues
```

---

## Phase 1.4 — PCR_SOA batch-native

### Goal

Add an exact spatially parallel solver backend.

### Backend

```python
DoubleCableLinearSolver.PCR_SOA
```

### Algorithm

PCR eliminates neighbors at strides:

```text
1, 2, 4, 8, ...
```

After `ceil(log2(Nx))` stages, rows are independent.

### Expected value

PCR may win if:

```text
Thomas remains limited by Nx dependency
B is not large enough to saturate GPU
XLA lowers PCR stages well
```

PCR may lose if:

```text
Nx is very small
extra arithmetic dominates
gather/scatter temporaries dominate
```

### Go/no-go

Keep as GPU backend if:

```text
speedup >= 1.5x vs best Thomas backend for relevant B,Nx
```

Otherwise keep as optional backend.

---

## Phase 1.5 — Split two-rail fixed-K solver

### Goal

Exploit the fact that double-cable can be written as two coupled scalar cable equations:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

Use scalar tridiagonal solves in fixed iterations:

```text
given Ve^k: solve Ti Vi^{k+1} = bi - Cie Ve^k
given Vi^k or Vi^{k+1}: solve Te Ve^{k+1} = be - Cei Vi
```

### Methods

```text
split_jacobi_fixed_K
split_gauss_seidel_fixed_K
preconditioned_richardson_fixed_K
```

### Why it is interesting

It reuses the single-cable GPU path and turns one block solve into:

```text
2 * K scalar tridiagonal solves
```

This can scale well if the scalar solver is highly optimized.

### Warning

Fixed `K` is approximate. Iterating to convergence is exact but less GPU-friendly.

### Validation

Compare against block Thomas:

```text
residual norm
max Vi/Ve/Vm error
activation agreement
threshold error
conduction velocity
first spike time
```

### Go/no-go

Keep if:

```text
K <= 4 gives physiologically negligible error
and speedup > best exact direct backend
```

---

## Phase 2 — Associative scan

### Goal

Explore exact parallel prefix formulations.

### Backward associative scan

After forward Thomas:

```text
x_i = d_i - C_i x_{i+1}
```

This is affine:

```text
f_i(x) = A_i x + q_i
```

Affine composition is associative, so the backward pass can use `jax.lax.associative_scan`.

Expected gain:

```text
modest, maybe 5-25%
```

### Transfer-matrix associative scan

Rewrite the whole system as:

```text
y_{i+1} = M_i y_i
```

where:

```text
y_i = [x_i, x_{i-1}, 1]
```

Then use matrix-product prefix scan.

Risks:

```text
conditioning
float32 stability
U_i invertibility
larger temporary matrices
```

### Go/no-go

Continue only if:

```text
float64 error < 1e-8 vs Thomas
float32 is stable on physical systems
performance is competitive with PCR/PTA
```

---

## Phase 3 — Pallas or FFI

### Goal

Only after JAX backends identify the winning algorithm, implement a custom kernel.

### Pallas candidates

```text
pallas_pta_block_thomas_small_nx
pallas_hybrid_pcr_thomas_small_nx
pallas_pcr_full_small_nx
```

### Why Pallas may help

```text
control tiling
control memory layout
reduce temporaries
scalarize 2x2 operations
avoid generic gather/scatter overhead
reduce kernel count
```

### FFI candidates

Only consider FFI if Pallas is insufficient and the best algorithm is clear.

Possible future FFI direction:

```text
CUDA block-Thomas 2x2 kernel inspired by PfSolve/PTA
CUDA hybrid PCR+Thomas 2x2 kernel
```

### Go/no-go

Use Pallas/FFI only if:

```text
end-to-end speedup >= 1.5x over best JAX backend
or solver-only speedup >= 2x over best JAX backend
```

---

# Backend selection policy

Initial policy:

```python
if device == "cpu":
    solver = THOMAS
elif Nx <= 100 and B_effective >= 512:
    solver = PTA_BLOCK_THOMAS
else:
    solver = THOMAS
```

After benchmarks:

```python
if PTA wins for Nx <= 64:
    use PTA_BLOCK_THOMAS
elif PCR wins for Nx >= 96:
    use PCR_SOA
elif split fixed-K is accurate and faster:
    use SPLIT_ITERATIVE for screening/threshold workflows
else:
    use THOMAS
```

Pallas/FFI should be opt-in until very stable.

---

# Benchmark matrix

## Solver-only benchmark

```text
B:       512, 1024, 2048, 4096, 8192
Nx:      32, 51, 64, 96, 100, 128
dtype:   float32, float64
layout:  BxNx, NxB, tiled
solver:  current_thomas, pta_block_thomas, pcr_soa, split_K2/K4, assoc_*
```

Metrics:

```text
steady-state time
compile time
node-solves/s
speedup vs current Thomas
max_abs_error vs Thomas float64
memory allocated
kernel count
trace interpretation
```

## End-to-end benchmark

```text
B:         512, 1024, 2048, 4096
Nx:        32, 51, 64, 96, 100
Nt:        500, 1000
output:    none, observer, center, full
Iinj:      none, dense_zero, nonzero
Vext:      dense, factorized
solver:    current_thomas, pta, pcr, split, assoc
```

Metrics:

```text
total wall time
solver-only portion if isolated
GPU memory
JIT call count
effective B per call
activation/threshold agreement
```

---

# Implementation details for PTA block-Thomas

## SoA 2x2 operations

Use scalarized 2x2 formulas.

For a block:

```text
D = [[d00, d01],
     [d10, d11]]
```

Inverse-vector solve:

```text
det = d00*d11 - d01*d10
x0 = ( d11*r0 - d01*r1) / det
x1 = (-d10*r0 + d00*r1) / det
```

Never call generic small-matrix inverse inside the hot loop.

## Layout variants

### BxNx

```text
rhs0[b, i]
rhs1[b, i]
```

Good for per-fiber contiguous local scans.

### NxB

```text
rhs0[i, b]
rhs1[i, b]
```

Likely better when each spatial step updates all fibers.

### Tiled

```text
rhs0[tile_i, tile_b]
```

Best candidate for Pallas/FFI.

## Padding

Use buckets:

```text
32, 64, 128
```

Padded rows should be identity/no-op:

```text
D = I
L = U = 0
rhs = 0
```

Slice outputs back to real `Nx`.

---

# Integration with AxonScope dispatcher

The scheduler should maximize:

```text
B_effective per compiled call
```

Use bucket keys:

```text
mode
solver
Nx_pad
dtype
recording mode
Iinj kind
Vext kind
membrane signature
```

Do not rely primarily on concurrent small GPU calls. Prefer:

```text
coalesce first
async enqueue second
```

This follows the same logic as Pipelined-TDMA: large enough batches improve GPU occupancy, but batch size should be managed to avoid excess temporary memory or pipeline inefficiency.

---

# Risks and mitigations

## Risk: PTA still underutilizes GPU

Mitigation:

```text
increase B_effective
use scheduler coalescing
try NxB/tiled layout
then test PCR
```

## Risk: PCR has too much overhead

Mitigation:

```text
use hybrid PCR+Thomas
use Pallas only if JAX overhead is visible
```

## Risk: split fixed-K is inaccurate

Mitigation:

```text
residual check
threshold/conduction validation
fallback to exact Thomas/PCR
```

## Risk: layout changes complicate code

Mitigation:

```text
keep public API batch-first
transpose internally inside solver backend
benchmark before committing
```

## Risk: full Vm output hides solver gains

Mitigation:

```text
benchmark observer-only and full-output separately
optimize compact outputs
factorize Vext
remove dense zero Iinj
```

---

# Final recommendation

The new literature shifts the recommended first solver engineering step.

Previously, the natural next step was:

```text
PCR_SOA first
```

After reviewing PTA/PfSolve/Pipelined-TDMA/NVIDIA/PaScaL_TDMA resources, the better order is:

```text
1. Optimize batched block-Thomas first.
2. Treat memory layout and coalescing as first-class solver work.
3. Increase B_effective through scheduler coalescing.
4. Then compare PCR_SOA and hybrid PCR/Thomas.
5. Then test split two-rail fixed-K.
6. Associative scan and Pallas/FFI remain later-stage options.
```

For AxonScope's target regime, `Nx=30-100` is short enough that Thomas may be algorithmically fine. The real issue may be that the current implementation does not present the GPU with enough coalesced, regular, many-system work.

The most actionable backend to implement next is:

```python
DoubleCableLinearSolver.PTA_BLOCK_THOMAS
```

with layout variants:

```text
PTA_BxNx
PTA_NxB
PTA_TILED_Nxpad_BLOCKB
```

This should be benchmarked against the current block Thomas and PCR_SOA before investing heavily in associative scan or Pallas.

---

# References

## Axon biology / double-cable motivation

- Abdollahi, N. and Prescott, S. A. (2024). *Impact of Extracellular Current Flow on Action Potential Propagation in Myelinated Axons*. Journal of Neuroscience. DOI: 10.1523/JNEUROSCI.0569-24.2024.
  - Uploaded file in this conversation: `e0569242024.full.pdf`

## GPU tridiagonal / TDMA resources

- Souri, M. et al. (2020). *Parallel Thomas approach development for solving tridiagonal matrix equations on GPUs*. Mechanics & Industry.
  - https://www.mechanics-industry.org/articles/meca/full_html/2020/03/mi170262/mi170262.html

- Tolmachev, D. et al. (2025). *High Performance Solution of Tridiagonal Systems on the GPU*. ACM.
  - https://dl.acm.org/doi/10.1145/3716171
  - https://www.research-collection.ethz.ch/items/fab84ea7-24e3-48ec-bd12-85477e1ef4e3

- PfSolve repository.
  - https://github.com/QuICC/PfSolve

- Kim, S. et al. (2025). *A Highly Scalable TDMA for GPUs and Its Application to Flow Solver Optimization*. arXiv:2509.03933.
  - https://arxiv.org/abs/2509.03933
  - https://arxiv.org/html/2509.03933v1

- NVIDIA GTC 2009. *Tridiagonal Solvers on the GPU and Applications to Fluid Simulation*.
  - https://www.nvidia.com/content/gtc/documents/1058_gtc09.pdf

- PaScaL_TDMA: Parallel and Scalable Library for TriDiagonal Matrix Algorithm.
  - https://github.com/xccels/PaScaL_TDMA

- Zhang, Y., Cohen, J., and Owens, J. D. *Fast Tridiagonal Solvers on the GPU*.
  - https://research.nvidia.com/sites/default/files/pubs/2010-01_Fast-Tridiagonal-Solvers/Zhang_Fast_2009.pdf

- Appleyard, J., et al. *Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers*.
  - https://people.maths.ox.ac.uk/gilesm/files/toms_16b.pdf

## JAX implementation references

- JAX asynchronous dispatch.
  - https://docs.jax.dev/en/latest/async_dispatch.html

- JAX benchmarking.
  - https://docs.jax.dev/en/latest/benchmarking.html

- JAX `associative_scan`.
  - https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html

- JAX Pallas.
  - https://docs.jax.dev/en/latest/pallas/index.html
