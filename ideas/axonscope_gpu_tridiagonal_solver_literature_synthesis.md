# AxonScope GPU Tridiagonal / Block-Tridiagonal Solver Literature Synthesis — Expanded v2

## Purpose

This document synthesizes the external GPU tridiagonal / TDMA / Hines-solver resources and translates them into a concrete AxonScope roadmap for improving the **exact double-cable solver**.

This version intentionally preserves the practical detail from the earlier notes and adds the Hines-matrix GPU paper as an additional source of design guidance, rather than replacing the previous material.

Target AxonScope regime:

```text
Nx = 30-100 compartments per fiber
B  > 500 fibers
Nt = many time steps
model = exact double-cable
linear system = many small 2x2 block-tridiagonal systems
main use cases = threshold, activation, recruitment, conduction validation
```

Main conclusion:

> For AxonScope's target regime, the first serious GPU optimization to test should be a **PTA-style batched block-Thomas 2x2 solver with SoA coefficients and interleaved/tiled memory layout**. PCR, associative scan, split iterative, and Pallas are still important, but they should be benchmarked against a strong many-small-systems Thomas baseline.

---

## Why exact double-cable remains worth optimizing

Simplified surrogate models can be useful for fast screening, but they cannot fully replace the exact double-cable solver.

Abdollahi & Prescott (2024) show that myelinated axon conduction depends strongly on how current distributes across:

```text
axial intracellular current
submyelin / periaxonal current
transmyelin current
extramyelin / extracellular current
```

Their adapted MRG double-cable simulations show that extracellular boundary conditions change conduction velocity, propagation reliability, energy efficiency, demyelination sensitivity, and ephaptic effects.

They compare:

```text
Condition 1:
    conventional double-cable / grounded extramyelin layer
    absorptive boundary
    more transmyelin leakage
    less axial current reaching the next node

Condition 2:
    internodal extramyelin compartments disconnected from ground
    node-associated extracellular space connected to ground
    dense-fascicle-like condition
    less transmyelin leakage
    more axial/submyelin current reaches the next node
    faster conduction

Condition 3:
    similar to Condition 2 but with lower longitudinal extramyelin resistance
    more current escapes into extramyelin longitudinal paths
    slower than Condition 2

Condition 2*:
    multifiber validation of the dense-fascicle interpretation
```

Implication for AxonScope:

```text
surrogate models:
    useful for large-scale screening

exact double-cable:
    still required as reference
    required for validation
    required when current pathways and boundary effects matter
```

Therefore the goal here is to improve the exact double-cable solver, not to replace it.

---

## AxonScope linear algebra target

At each implicit double-cable time step, the system is a 2x2 block-tridiagonal chain:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = r_i

x_i = [Vi_i, Ve_i]^T
D_i = 2x2 local block
L_i = lower 2x2 off-block
U_i = upper 2x2 off-block
```

Because the off-blocks are often diagonal in the two rails, a highly specialized SoA implementation should be possible.

The same system can also be written as two coupled scalar cables:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

where:

```text
Ti = scalar tridiagonal intracellular/axonal cable
Te = scalar tridiagonal periaxonal/extracellular cable
Cie, Cei = local diagonal couplings between rails
```

This gives two families of solver approaches:

```text
direct block solve:
    block Thomas
    PCR / cyclic reduction
    associative transfer scan
    Pallas/CUDA custom block solver

split two-rail solve:
    Jacobi / Gauss-Seidel / Richardson fixed-K
    two scalar tridiagonal solves per iteration
    exact only at convergence
```

---

# Resource 1 — PfSolve / Parallel Factorization Solver

## Reference

- PfSolve: Parallel Factorization Solver for tridiagonal and bidiagonal matrices.
- ACM Transactions on Mathematical Software, 2025.
- DOI: `10.1145/3716171`
- ACM URL: https://dl.acm.org/doi/10.1145/3716171
- ETH Research Collection URL: https://www.research-collection.ethz.ch/items/fab84ea7-24e3-48ec-bd12-85477e1ef4e3

## What it contributes

PfSolve is directly relevant because it focuses on GPU tridiagonal and bidiagonal solves. Its abstract and related descriptions emphasize:

```text
many tridiagonal systems
GPU-oriented Thomas-family factorization
warp-level instructions
occupancy optimizations
low extra memory overhead
support for arbitrary system sizes that fit on GPU
```

The main idea to transfer is not a literal library call, because AxonScope has a **2x2 block-tridiagonal** system rather than a scalar tridiagonal system.

The transferable lesson is:

> A highly optimized Thomas-family solver can be very strong on GPU when the workload consists of many short independent systems.

## AxonScope implication

Before spending too much effort on PCR or associative scan, implement and benchmark:

```text
PTA_BLOCK_THOMAS_2X2
```

This should be:

```text
exact
same numerical solution as current block Thomas
SoA-based
batch-native
layout-optimized
GPU-oriented
```

## Why this matters for Nx=30-100

For very long systems, Thomas' sequential depth is a major problem. But for AxonScope:

```text
Nx = 30-100
```

so the sequential depth is short. The main issue may be:

```text
not enough independent systems
bad coalescing
wrong layout
too many small kernels
too many array temporaries
```

rather than Thomas itself.

---

# Resource 2 — Mechanics & Industry 2020 Parallel Thomas Algorithm article

## Reference

- Mechanics & Industry 2020, Parallel Thomas Algorithm for GPUs.
- URL: https://www.mechanics-industry.org/articles/meca/full_html/2020/03/mi170262/mi170262.html

## What it contributes

This paper compares GPU variants for solving many tridiagonal systems and emphasizes a key point:

> Coalesced memory access can be as important as the algorithmic choice.

The paper discusses Parallel Thomas Algorithm (PTA), CR, and PCR-like approaches. Its key practical message for AxonScope is that a Thomas-style algorithm can perform very well when many systems are solved in parallel and memory access is organized correctly.

## AxonScope implication

Do not only benchmark:

```text
current Thomas vs PCR
```

Instead benchmark:

```text
current Thomas
PTA-style Thomas [B, Nx]
PTA-style Thomas [Nx, B]
PTA-style Thomas tiled [tile, Nx_pad, BLOCK_B]
PCR_SOA
```

If a layout-optimized Thomas wins, use it as the default GPU backend for small `Nx`.

---

# Resource 3 — NVIDIA GTC 2009 tridiagonal GPU slides

## Reference

- NVIDIA GTC 2009 presentation, tridiagonal solvers on GPU.
- URL: https://www.nvidia.com/content/gtc/documents/1058_gtc09.pdf

## What it contributes

The slides present a simple and important GPU pattern:

```text
one thread solves one tridiagonal system
many independent tridiagonal systems are solved in parallel
```

They also highlight that memory layout can dominate performance. In their stencil/ADI context, one sweep direction is slower because memory access is not coalesced; reordering / changing data access improves performance.

## AxonScope implication

The essential question for AxonScope is:

> During a Thomas forward/backward sweep, are neighboring GPU lanes reading contiguous memory?

If the solver stores data as:

```text
rhs[B, Nx]
```

and each lane/thread/program handles a fiber, then a given fiber is contiguous. But if the implementation updates all fibers at spatial index `i`, memory access may be strided.

Therefore test both:

```text
fiber-major:
    [B, Nx]

index-major / interleaved:
    [Nx, B]

tiled:
    [n_tiles, Nx_pad, BLOCK_B]
```

This is especially important for a PTA-style solver.

---

# Resource 4 — arXiv 2509.03933v1 Pipelined TDMA

## Reference

- A Highly Scalable TDMA for GPUs / Pipelined TDMA.
- URL: https://arxiv.org/html/2509.03933v1

## What it contributes

The paper emphasizes the throughput problem of small tridiagonal systems on GPUs:

```text
small systems underutilize GPU
batching improves occupancy
kernel launch overhead matters
batch size has an optimal range
too-large batches can increase temporary memory or reduce pipeline efficiency
```

## AxonScope implication

The solver cannot be optimized in isolation. The dispatcher must feed it large enough batches.

Recommended AxonScope scheduling direction:

```text
B_effective = fibers × amplitudes × electrode configs × model variants
```

and:

```text
Nx buckets:
    32 / 64 / 128

coalesce dispatch groups:
    merge compatible groups into larger JIT calls

async enqueue:
    optional secondary optimization
    not the main strategy

compact outputs:
    avoid full Vm[B, Nt, Nx] when not needed
```

This resource reinforces the dispatch scheduling note: **batch size is a performance parameter**.

---

# Resource 5 — PaScaL_TDMA

## Reference

- PaScaL_TDMA GitHub.
- URL: https://github.com/xccels/PaScaL_TDMA

## What it contributes

PaScaL_TDMA is a Parallel and Scalable Library for Tridiagonal Matrix Algorithm. It targets many tridiagonal systems, including multi-GPU / MPI contexts, and includes CUDA-oriented implementations.

It is not directly reusable in AxonScope because:

```text
AxonScope uses JAX
AxonScope target is small Nx
AxonScope system is 2x2 block-tridiagonal
single-GPU performance is the first target
```

But it is useful as a reference for:

```text
multi-GPU decomposition
pipeline-style TDMA
CUDA FFI backend ideas
how mature TDMA libraries organize memory and communication
```

## AxonScope implication

Do not integrate PaScaL_TDMA directly in Phase 1.

Use it later as inspiration for:

```text
FFI CUDA backend
multi-device batch sharding
Pallas kernel design
```

---

# Resource 6 — Efficient Tree Solver for Hines Matrices on the GPU

## Reference

- Huber, F. (2018). Efficient Tree Solver for Hines Matrices on the GPU.
- arXiv: `1810.12742`
- URL: https://arxiv.org/abs/1810.12742
- PDF: https://arxiv.org/pdf/1810.12742

## What it contributes

This paper solves Hines matrices for neuronal tree morphologies on GPU. Its algorithm is not directly applicable to AxonScope's linear double-cable chain, but its GPU engineering lessons are very relevant.

The paper addresses a similar problem class:

```text
many small structured linear systems
GPU underutilization if each system is assigned too coarsely
need for fine-grained parallelism
need for memory layout that supports coalesced access
need for work balancing when systems have different sizes
```

## Important design lessons

### 1. One whole system per thread/program may underutilize the GPU

For small systems, mapping one full matrix or one full cell to one GPU thread can leave too little parallelism.

AxonScope equivalent:

```text
avoid assuming:
    one fiber = one thread/program

test:
    one warp/block/tile handles multiple fibers
    [Nx_pad, BLOCK_B] tile
    multiple fibers processed together at each spatial index
```

### 2. Interleaved layout matters

The Hines solver uses interleaved data layout so that operations at a similar dependency level access contiguous memory.

AxonScope equivalent:

```text
test [Nx, B] layout
test tiled [Nx_pad, BLOCK_B] layout
```

This reinforces the same conclusion from NVIDIA and PTA resources: layout can dominate the perceived quality of the solver.

### 3. Bucket and balance work

The Hines paper deals with heterogeneous tree sizes and branches. AxonScope has a simpler version of this:

```text
fibers with different Nx
models with different compartment counts
different recording modes
different solver backends
```

Equivalent AxonScope strategy:

```text
Nx buckets = 32 / 64 / 128
avoid mixing very different Nx in one GPU tile
track padded overhead
sort/coalesce compatible jobs
```

### 4. Do not directly port the Hines algorithm

The Hines solver is for branched trees:

```text
parent-child dependencies
branch splitting
tree reduction
```

AxonScope double-cable is:

```text
linear chain
two coupled voltage rails
2x2 block-tridiagonal system
```

So the direct Hines algorithm is not the target. The target is to adopt its GPU layout and work-granularity lessons.

---

# Updated interpretation of the solver problem

Earlier instinct:

```text
The double-cable GPU problem is mainly that Thomas is sequential in Nx.
Therefore PCR/associative scan should be the main route.
```

Updated interpretation after the literature:

```text
For Nx=30-100, the sequential depth of Thomas is short.
The larger bottleneck may be:
    many small systems
    weak batching
    non-coalesced memory
    poor layout
    too many temporaries
    JAX/XLA not generating the ideal kernel
```

Therefore the first serious target should be:

```text
PTA-style batched block-Thomas 2x2
with SoA coefficients
and explicit layout benchmarks
```

PCR/associative/Pallas should be compared against that strong baseline, not against the current possibly layout-suboptimal Thomas implementation.

---

# Updated recommended roadmap

## Phase 0 — Measurement and tracing

Before changing algorithms:

```text
1. Add solver-only benchmark.
2. Add end-to-end double-cable benchmark.
3. Add JAX profiler traces.
4. Separate compile time from runtime.
5. Use block_until_ready for all timings.
6. Enable transfer_guard=log in benchmarks.
7. Record kernel count and effective batch size.
```

Benchmark matrix:

```text
B  = 128, 512, 1024, 2048, 4096, 8192
Nx = 16, 32, 51, 64, 96, 100, 128
dtype = float32, float64
solver = current_thomas, pcr_soa, split_iterative, future PTA
```

---

## Phase 1.1 — PTA-style batched block-Thomas 2x2

### Goal

Implement a stronger Thomas baseline that is GPU-oriented but mathematically identical to the current block Thomas solver.

### Backend name

```python
DoubleCableLinearSolver.PTA_BLOCK_THOMAS
```

### Key implementation principles

Use SoA:

```text
d00, d01, d10, d11
l0, l1
u0, u1
rhs0, rhs1
```

Avoid generic tiny matrices in performance-critical code:

```text
avoid [B, Nx, 2, 2]
prefer scalar arrays
```

Keep the block 2x2 operations explicit:

```text
det = a00 * a11 - a01 * a10
inv00 =  a11 / det
inv01 = -a01 / det
inv10 = -a10 / det
inv11 =  a00 / det
```

### Variants

```text
PTA_BLOCK_THOMAS_BX:
    layout [B, Nx]

PTA_BLOCK_THOMAS_XB:
    layout [Nx, B]

PTA_BLOCK_THOMAS_TILED:
    layout [n_tiles, Nx_pad, BLOCK_B]
```

### Correctness

Must match current Thomas:

```text
float64 max_abs_error < 1e-9
float32 max_abs_error < 1e-5
no activation/threshold difference
```

### Go/no-go

Keep if:

```text
speedup vs current Thomas >= 1.5x for B>=1024, Nx<=100
or end-to-end speedup >= 1.2x with compact outputs
```

---

## Phase 1.1b — Interleaved/tiled layout benchmark

### Goal

Find the best internal layout for the double-cable solver.

### Layouts to test

```text
A. fiber-major:
    [B, Nx]

B. spatial-index-major:
    [Nx, B]

C. tiled:
    [n_tiles, Nx_pad, BLOCK_B]
```

### Why this phase exists

Multiple resources point to the same principle:

```text
memory coalescing can dominate tridiagonal solver speed on GPU
```

If `[Nx, B]` or tiled layout wins by a large margin, integrate that layout into the GPU backend even if AxonScope's public API remains batch-first.

### Go/no-go

Adopt new layout if:

```text
speedup >= 1.3x
packing/unpacking overhead is small
end-to-end improvement remains visible
```

---

## Phase 1.2 — Dispatch coalescing and Nx bucketization

### Goal

Make sure the solver sees large enough batches.

### Scheduler improvements

```text
1. bucket Nx to 32 / 64 / 128
2. coalesce compatible groups
3. keep solver/mode/dtype/recording/Iinj/Vext in the bucket key
4. optionally enqueue multiple groups asynchronously
5. synchronize once or at memory-budget flush points
```

### Why this matters

Small TDMA systems are occupancy-limited. Even a perfect solver can perform poorly if it gets many small calls.

### Default recommendation

```text
coalesce_groups = True
async_groups = False initially
Nx buckets = 32, 64, 128
```

Enable async only after memory-safe benchmarking.

---

## Phase 1.3 — PCR_SOA batch-native

### Goal

Implement/officialize an exact cyclic reduction solver that exposes spatial parallelism.

### Why still test PCR

PCR has:

```text
dependency depth O(log Nx)
more arithmetic than Thomas
more temporary arrays
potentially better GPU parallelism
```

It may beat PTA for larger `Nx` or if batch size is not high enough.

### Required comparison

Compare PCR against the **best PTA backend**, not only against current Thomas.

### Go/no-go

Keep PCR if:

```text
PCR beats best PTA for Nx>=64 or Nx>=96
and numerical agreement is strong
```

---

## Phase 1.4 — Split two-rail iterative solver

### Goal

Use the two-rail form:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

and solve by iterations:

```text
given Ve:
    solve Ti for Vi

given Vi:
    solve Te for Ve
```

### Backends

```text
split_jacobi_fixed_K
split_gauss_seidel_fixed_K
preconditioned_richardson_fixed_K
```

### Why this is attractive

It reuses scalar tridiagonal solves, potentially the already fast single-cable path.

Effective work:

```text
2 scalar tridiagonal solves per iteration per fiber
```

For GPU:

```text
B_effective ≈ 2 * K * B
```

### Exactness

```text
fixed K:
    approximate

iterate to convergence:
    exact numerically but less GPU-friendly
```

### Validation metrics

Do not rely only on voltage norm error. Check:

```text
residual ||Ax-b|| / ||b||
Vm error vs Thomas
activation agreement
threshold error
first spike timing
conduction velocity
recruitment curve
```

### Go/no-go

Keep if:

```text
K=2 or K=4 gives large speedup
and physiological metrics remain acceptable
```

---

## Phase 2 — Associative scan

### Phase 2A — Associative backward substitution

After Thomas forward elimination, the backward pass is affine:

```text
x_i = d_i - C_i x_{i+1}
```

This can be written as:

```text
f_i(x) = A_i x + q_i
```

Affine transform composition is associative:

```text
f(g(x)) = A_f A_g x + A_f q_g + q_f
```

Use `jax.lax.associative_scan(reverse=True)` to parallelize the backward pass.

Expected gain:

```text
modest
low risk
does not remove forward dependency
```

### Phase 2B — Full transfer-matrix associative scan

Rewrite each row as a state transition:

```text
x_{i+1} = -U_i^{-1}D_i x_i - U_i^{-1}L_i x_{i-1} + U_i^{-1}b_i
```

State:

```text
y_i = [x_i, x_{i-1}, 1]
```

Then:

```text
y_{i+1} = M_i y_i
```

Prefix products of `M_i` can be computed by associative scan.

Risks:

```text
U_i conditioning
matrix product instability
float32 stability
heterogeneous NODE/MYSA/FLUT/STIN coefficients
```

Prototype dense 5x5 first, optimize only if stable.

---

## Phase 3 — Pallas / custom kernels

### When to move to Pallas

Only after Phase 1 identifies the best math/layout.

Use Pallas if JAX traces show:

```text
too many kernels
poor fusion
large temporary arrays
bad gather/scatter overhead
layout not expressible cleanly in JAX
```

### Candidate Pallas kernels

```text
pallas_pta_block_thomas_2x2:
    first custom kernel candidate

pallas_pcr_hybrid:
    only if PCR/hybrid wins algorithmically

pallas_tiled_solver:
    [tile, Nx_pad, BLOCK_B]
```

### Why not start with Pallas

Pallas does not change the math. It only gives lower-level control. It is more work and more maintenance.

---

## Optional Phase 4 — FFI / CUDA backend

Use only if:

```text
Pallas is insufficient
the winning solver is stable
there is a clear reason to maintain CUDA/C++ code
```

Potentially useful references:

```text
PfSolve
PaScaL_TDMA
custom CUDA TDMA kernels
```

But FFI should not be a Phase 1 solution.

---

# Proposed backend enum

```python
class DoubleCableLinearSolver(str, Enum):
    THOMAS = "thomas"

    PTA_BLOCK_THOMAS = "pta_block_thomas"
    PTA_BLOCK_THOMAS_BX = "pta_block_thomas_bx"
    PTA_BLOCK_THOMAS_XB = "pta_block_thomas_xb"
    PTA_BLOCK_THOMAS_TILED = "pta_block_thomas_tiled"

    PCR_SOA = "pcr_soa"

    SPLIT_JACOBI = "split_jacobi"
    SPLIT_GAUSS_SEIDEL = "split_gauss_seidel"
    SPLIT_RICHARDSON = "split_richardson"

    ASSOCIATIVE_BACKWARD = "associative_backward"
    ASSOCIATIVE_TRANSFER = "associative_transfer"

    PALLAS_PTA_BLOCK_THOMAS = "pallas_pta_block_thomas"
    PALLAS_PCR_HYBRID = "pallas_pcr_hybrid"

    AUTO = "auto"
```

---

# Benchmark plan

## Solver-only benchmark

Create:

```text
benchmark/solvers/bench_double_cable_linear_solvers.py
```

Backends:

```text
current_thomas
pta_block_thomas_BxNx
pta_block_thomas_NxB
pta_block_thomas_tiled
pcr_soa
split_jacobi_K1/K2/K4/K8
split_gauss_seidel_K1/K2/K4/K8
associative_backward
associative_transfer_dense
```

Sizes:

```text
B  = 128, 512, 1024, 2048, 4096, 8192
Nx = 16, 32, 51, 64, 96, 100, 128
dtype = float32, float64
```

Metrics:

```text
compile time
steady-state time
node-solves/s = B * Nx / time
speedup vs current Thomas
max_abs_error vs current Thomas
max_rel_error vs current Thomas
residual norm
number of kernels
estimated memory traffic
```

## End-to-end benchmark

Create:

```text
benchmark/solvers/bench_double_cable_end_to_end.py
```

Backends:

```text
current_thomas
best_pta_block_thomas
pcr_soa
split_iterative_best
associative_best_if_any
```

Workloads:

```text
B  = 512, 1024, 2048, 4096
Nx = 32, 51, 64, 96, 100
Nt = 500, 1000
recording = observer-only, center, full
Iinj = none, dense_zero, nonzero
Vext = dense, factorized if available
```

Metrics:

```text
total wall time
solver-only time if isolated
kernel wait time
GPU memory
compile count
kernel count
activation agreement
threshold agreement
```

---

# AUTO policy sketch

Initial:

```python
if device == "cpu":
    solver = THOMAS

elif Nx <= 100 and B >= 512:
    solver = PTA_BLOCK_THOMAS_BEST_LAYOUT

elif Nx > 100 and B >= 512:
    solver = PCR_SOA

else:
    solver = THOMAS
```

After benchmarking:

```text
choose PTA vs PCR vs split by table:
    device
    dtype
    Nx bucket
    B bucket
    output mode
```

Do not hardcode theoretical preferences. Use measured backend tables.

---

# Key implementation rules

## 1. Preserve Thomas as oracle

Never remove the current robust Thomas solver.

Use it for:

```text
CPU
small B
validation
fallback
debugging
```

## 2. Use SoA for block 2x2

Prefer:

```text
d00, d01, d10, d11
rhs0, rhs1
```

over:

```text
block[..., 2, 2]
```

in optimized paths.

## 3. Benchmark layout before concluding algorithmic failure

If Thomas is slow, first ask:

```text
is the layout coalesced?
is B large enough?
are shapes bucketed?
are there too many tiny calls?
```

Only then conclude that Thomas is algorithmically insufficient.

## 4. Separate solver-only from end-to-end

Solver-only speedup may not show end-to-end if time is dominated by:

```text
Vm[B, Nt, Nx] output
Vext materialization
dense zero Iinj
host/device transfers
dispatch group overhead
```

## 5. Compact outputs matter

For threshold/recruitment workflows, prefer:

```text
activation observer
first spike time
peak Vm
threshold result
center/probe traces
```

over full trace:

```text
Vm[B, Nt, Nx]
```

---

# Final recommendation

The next practical implementation order should be:

```text
1. Add solver-only benchmark and JAX traces.
2. Implement PTA-style batched block-Thomas 2x2.
3. Benchmark [B,Nx] vs [Nx,B] vs tiled layout.
4. Add Nx bucketization and group coalescing in dispatcher.
5. Officialize PCR_SOA and compare against best PTA baseline.
6. Test split two-rail fixed-K.
7. Only then test associative scan.
8. Move to Pallas only after identifying the best algorithm/layout.
9. Consider FFI/CUDA only if Pallas is insufficient.
```

The main updated lesson from the literature is:

> For short cables (`Nx=30-100`) and many independent fibers (`B>500`), a memory-coalesced, batch-optimized Thomas-family solver may be the most realistic first win. PCR and associative scan are still important, but they should be benchmarked against a strong PTA baseline, not against the current implementation alone.

---

# References

## GPU tridiagonal / TDMA

1. PfSolve / Parallel Factorization Solver for tridiagonal and bidiagonal matrices.
   - DOI: `10.1145/3716171`
   - https://dl.acm.org/doi/10.1145/3716171
   - https://www.research-collection.ethz.ch/items/fab84ea7-24e3-48ec-bd12-85477e1ef4e3

2. Pipelined TDMA for GPUs.
   - arXiv `2509.03933v1`
   - https://arxiv.org/html/2509.03933v1

3. NVIDIA GTC 2009 tridiagonal solver slides.
   - https://www.nvidia.com/content/gtc/documents/1058_gtc09.pdf

4. PaScaL_TDMA.
   - https://github.com/xccels/PaScaL_TDMA

5. Mechanics & Industry 2020 Parallel Thomas Algorithm article.
   - https://www.mechanics-industry.org/articles/meca/full_html/2020/03/mi170262/mi170262.html

## Hines / neuroscience GPU solvers

6. Huber, F. (2018). Efficient Tree Solver for Hines Matrices on the GPU.
   - arXiv `1810.12742`
   - https://arxiv.org/abs/1810.12742
   - https://arxiv.org/pdf/1810.12742

## Axon biology / double-cable motivation

7. Abdollahi, N. and Prescott, S. A. (2024). Impact of Extracellular Current Flow on Action Potential Propagation in Myelinated Axons.
   - Journal of Neuroscience, 44(26):e0569242024.
   - DOI: `10.1523/JNEUROSCI.0569-24.2024`
   - Uploaded PDF in this conversation: `e0569242024.full.pdf`

## AxonScope implementation targets

8. Current double-cable linear solver and PCR prototypes.
   - `src/axonscope/solvers/common.py`
   - https://raw.githubusercontent.com/louisreg/AxonScope/bench-colab/src/axonscope/solvers/common.py

9. Current dispatcher and batch grouping.
   - `src/axonscope/dispatcher/plan.py`
   - `src/axonscope/dispatcher/execution.py`
   - https://raw.githubusercontent.com/louisreg/AxonScope/main/src/axonscope/dispatcher/plan.py
   - https://raw.githubusercontent.com/louisreg/AxonScope/main/src/axonscope/dispatcher/execution.py

## JAX implementation tools

10. JAX asynchronous dispatch.
    - https://docs.jax.dev/en/latest/async_dispatch.html

11. JAX benchmarking guide.
    - https://docs.jax.dev/en/latest/benchmarking.html

12. JAX profiler.
    - https://docs.jax.dev/en/latest/profiling.html

13. `jax.lax.associative_scan`.
    - https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html

14. JAX Pallas.
    - https://docs.jax.dev/en/latest/pallas/index.html
