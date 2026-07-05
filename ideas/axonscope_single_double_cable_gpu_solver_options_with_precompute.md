# AxonScope GPU Solver Options for Single-Cable and Double-Cable Models

## Purpose

This document summarizes the solver options proposed for AxonScope when simulating many axons on GPU.

Target workload:

```text
Nx ≈ 30-100 compartments per fiber
B > 500 fibers
Nt = many time steps
many amplitudes / electrode configurations / stimulation conditions
JAX-based model construction and execution
GPU acceleration desired
```

The document covers both:

```text
single-cable models
double-cable models
```

and separates:

```text
exact solvers
approximate / iterative solvers
GPU layout and batching strategies
Pallas / CUDA implementation options
scheduler-level optimizations
```

Main conclusion:

> For short cables and many fibers, the first production-quality GPU direction should probably be a **many-small-systems Thomas-family solver** with SoA layout, `Nx` bucketing, interleaved/tiled memory, compact outputs, and very large effective batch size.

For double-cable:

```text
PTA-style batched block-Thomas 2x2
```

For single-cable:

```text
PTA-style batched scalar Thomas
```

---

# 1. Design principles

## 1.1 Do not optimize a single fiber

The target GPU workload is not:

```text
solve one cable very fast
```

It is:

```text
solve thousands of short, independent cables per time step
```

Therefore the main parallel axis should be:

```text
B_effective = fibers × amplitudes × electrode configs × model variants
```

Example:

```text
512 fibers × 8 amplitudes = 4096 independent systems
```

This is much better for GPU than launching eight small simulations separately.

---

## 1.2 Use static `Nx` buckets

Use fixed solver families:

```text
Nx_pad = 32
Nx_pad = 64
Nx_pad = 128
```

Example:

```text
Nx = 30-32   -> Nx_pad = 32
Nx = 33-64   -> Nx_pad = 64
Nx = 65-128  -> Nx_pad = 128
```

Padded rows must be electrically neutral:

```text
D = identity
L = 0
U = 0
rhs = 0
```

Then slice outputs back to the true `Nx`.

This reduces recompilation, makes kernels easier to specialize, and enables tiled GPU layouts.

---

## 1.3 Prefer SoA over small dense matrices

For block 2x2 systems, prefer scalar arrays:

```text
d00, d01, d10, d11
l0, l1
u0, u1
rhs0, rhs1
```

rather than:

```text
blocks[..., 2, 2]
```

Reason:

```text
fewer tiny matrix abstractions
fewer temporaries
clearer XLA lowering
easier Pallas/CUDA kernel mapping
better memory layout control
```

---

## 1.4 Test `[Nx, B]` and tiled layouts

The usual user-facing layout may be:

```text
[B, Nx]
```

But for GPU Thomas-style sweeps, the internal solver layout should be tested as:

```text
[Nx, B]
```

because at each spatial index `i`, all fibers are updated in parallel:

```text
rhs[i, :]
```

is contiguous.

A tiled production layout may be:

```text
[n_tiles, Nx_pad, BLOCK_B]
```

Example:

```text
Nx_pad = 64
BLOCK_B = 32 or 64
```

This is the natural layout for Pallas/CUDA kernels.

---

## 1.5 Avoid full trace output in performance mode

For threshold / recruitment / activation workflows, avoid:

```text
Vm[B, Nt, Nx]
```

unless needed for debugging or validation.

Prefer compact observers:

```text
activated[B]
first_spike_time[B]
peak_vm[B]
spike_count[B]
initiation_node[B]
conduction_success[B]
selected_probe_traces[B, Nt_probe]
```

Full traces can dominate memory bandwidth and hide solver improvements.

---

## 1.6 Factorize extracellular drive when possible

Avoid materializing:

```text
Vext[B, Nt, Nx]
```

if the stimulation can be written as:

```text
Vext[b, t, x] = footprint[b, x] * waveform[t]
```

Instead pass:

```text
footprint[Nx, B]
waveform[Nt]
```

and build the RHS inside the time loop.

---

# 2. Single-cable solver options

The single-cable implicit linear system is scalar tridiagonal:

```text
l_i V_{i-1} + d_i V_i + u_i V_{i+1} = rhs_i
```

This is exactly the kind of problem targeted by GPU TDMA / PTA literature.

---

## 2.1 Baseline: current JAX / XLA tridiagonal solver

### Name

```text
JAX_TRIDIAGONAL_REFERENCE
```

or:

```text
SINGLE_THOMAS_REFERENCE
```

### Role

Keep as reference and fallback.

Use for:

```text
CPU
small batch sizes
validation
debugging
fallback if custom backend fails
```

### Pros

```text
simple
robust
already integrated
portable across CPU/GPU/TPU
```

### Cons

```text
generic
may not be optimal for many short systems
may not expose ideal layout/coalescing
```

---

## 2.2 PTA scalar Thomas, `[B, Nx]` layout

### Name

```text
PTA_SCALAR_BX
```

### Idea

Implement scalar Thomas over many systems in batch-first layout:

```text
rhs[B, Nx]
```

### Role

Useful as an intermediate benchmark. It tells whether a custom scalar Thomas beats the current solver even before changing memory layout.

### Pros

```text
easy to integrate
minimal packing/unpacking
same shape as many existing AxonScope arrays
```

### Cons

```text
may not be memory-coalesced for GPU sweeps
not the final preferred layout
```

---

## 2.3 PTA scalar Thomas, `[Nx, B]` layout

### Name

```text
PTA_SCALAR_XB
```

### Idea

Use internal solver layout:

```text
rhs[Nx, B]
lower[Nx] or lower[Nx, B]
diag[Nx] or diag[Nx, B]
upper[Nx] or upper[Nx, B]
```

Forward Thomas:

```text
for i in 0..Nx-1:
    update all B systems at index i
```

Backward Thomas:

```text
for i in Nx-1..0:
    update all B systems at index i
```

### Why it may be faster

At each spatial step:

```text
rhs[i, :]
diag[i, :]
```

are contiguous, which is better for memory coalescing.

### Pros

```text
exact
simple
directly matches scalar TDMA literature
good first production candidate
```

### Cons

```text
requires transpose or direct construction in XB layout
still sequential in Nx
```

### Recommended first test

Benchmark:

```text
current single solver
PTA_SCALAR_BX
PTA_SCALAR_XB
```

for:

```text
B  = 512, 1024, 2048, 4096, 8192, 16384
Nx = 32, 51, 64, 96, 100, 128
```

---

## 2.4 PTA scalar tiled solver

### Name

```text
PTA_SCALAR_TILED
```

### Layout

```text
rhs[n_tiles, Nx_pad, BLOCK_B]
```

Example configurations:

```text
Nx_pad = 32,  BLOCK_B = 64, 128, 256
Nx_pad = 64,  BLOCK_B = 64, 128
Nx_pad = 128, BLOCK_B = 32, 64
```

### Scratch memory

Only needs:

```text
c[Nx_pad, BLOCK_B]
y[Nx_pad, BLOCK_B]
```

This is much lighter than the double-cable block case.

### Pros

```text
very good fit for GPU custom kernels
low scratch memory
easy Pallas/CUDA implementation
strong candidate for production
```

### Cons

```text
packing/unpacking overhead
requires careful bucket scheduling
```

---

## 2.5 Pre-factorized scalar Thomas

### Name

```text
PTA_SCALAR_PREFACTORED
```

### When applicable

If the tridiagonal coefficients do not change over time:

```text
lower, diag, upper constant over t
```

then precompute the Thomas forward coefficients once:

```text
c_i
denom_i
```

At each time step, solve only the RHS-dependent part:

```text
forward RHS update
backward substitution
```

### Advantage

This removes many divisions from the time loop.

### When it may not apply

If the membrane conductance changes every step and enters `diag`, full pre-factorization is not exact.

Still, partial precomputation may help if:

```text
lower/upper are constant
diag = constant axial term + variable membrane term
```

---

## 2.6 Pallas scalar PTA

### Name

```text
PALLAS_PTA_SCALAR
```

### Goal

Convert the winning JAX scalar PTA layout into a custom GPU kernel.

### Mapping

```text
grid = (n_tiles,)
one program handles [Nx_pad, BLOCK_B]
```

Each program:

```text
loads one tile
runs forward Thomas
runs backward Thomas
stores solution
```

### Pros

```text
control of tiling
fewer JAX temporaries
reduced kernel overhead
good fit for short systems
```

### Cons

```text
more maintenance
requires Pallas expertise
```

---

## 2.7 CUDA FFI scalar PTA

### Name

```text
CUDA_FFI_PTA_SCALAR
```

### Goal

If Pallas is insufficient, implement a CUDA C++ kernel inspired by PTA / PfSolve-style many-small-systems solvers.

### Pros

```text
maximum low-level control
can use warp-level optimizations
can tune shared memory and occupancy directly
```

### Cons

```text
highest maintenance cost
requires JAX FFI integration
less portable
```

---

## 2.8 Recommended single-cable path

Implementation order:

```text
1. Keep current solver as reference.
2. Implement PTA_SCALAR_XB in pure JAX.
3. Test PTA_SCALAR_BX vs PTA_SCALAR_XB.
4. Implement PTA_SCALAR_TILED in pure JAX or Pallas.
5. Add pre-factorized variant when coefficients are constant.
6. Move to Pallas if JAX traces show overhead.
7. Move to CUDA FFI only if Pallas is insufficient.
```

Expected winner:

```text
PTA_SCALAR_XB or PTA_SCALAR_TILED
```

For passive/constant coefficient cases:

```text
PTA_SCALAR_PREFACTORED
```

may be the fastest.

---

# 3. Double-cable solver options

The double-cable system is a 2x2 block-tridiagonal system:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = rhs_i

x_i = [Vi_i, Ve_i]^T
```

with local block:

```text
D_i =
[d00_i  d01_i]
[d10_i  d11_i]
```

and off-blocks often diagonal:

```text
L_i =
[l0_i  0   ]
[0     l1_i]

U_i =
[u0_i  0   ]
[0     u1_i]
```

---

## 3.1 Baseline block Thomas

### Name

```text
DOUBLE_THOMAS_REFERENCE
```

### Role

Keep as exact reference.

Use for:

```text
CPU
small B
validation
debugging
fallback
```

### Pros

```text
exact
stable
already implemented
good oracle
```

### Cons

```text
sequential in Nx
may not be layout-optimized for GPU
may rely too much on vmap / scan structure
```

---

## 3.2 PTA block Thomas 2x2, `[B, Nx]`

### Name

```text
PTA_BLOCK_THOMAS_BX
```

### Role

Intermediate benchmark to compare against current implementation.

### Layout

```text
rhs0[B, Nx]
rhs1[B, Nx]
d00[B, Nx] or d00[Nx]
...
```

### Pros

```text
easy to integrate
exact
minimal packing changes
```

### Cons

```text
may not be coalesced
not the preferred final layout
```

---

## 3.3 PTA block Thomas 2x2, `[Nx, B]`

### Name

```text
PTA_BLOCK_THOMAS_XB
```

### Core idea

Use a GPU-oriented X-major layout:

```text
d00[Nx, B]
d01[Nx, B]
d10[Nx, B]
d11[Nx, B]

l0[Nx, B]
l1[Nx, B]
u0[Nx, B]
u1[Nx, B]

rhs0[Nx, B]
rhs1[Nx, B]
```

At each Thomas step `i`, the solver processes:

```text
all B fibers at spatial index i
```

### Forward equations

Given previous modified block `C_{i-1}` and vector `y_{i-1}`:

```text
M_i = D_i - L_i C_{i-1}
r_i = rhs_i - L_i y_{i-1}

C_i = M_i^{-1} U_i
y_i = M_i^{-1} r_i
```

In SoA:

```text
m00 = d00_i - l0_i * c00_prev
m01 = d01_i - l0_i * c01_prev
m10 = d10_i - l1_i * c10_prev
m11 = d11_i - l1_i * c11_prev

r0 = rhs0_i - l0_i * y0_prev
r1 = rhs1_i - l1_i * y1_prev
```

2x2 inverse:

```text
det = m00 * m11 - m01 * m10

inv00 =  m11 / det
inv01 = -m01 / det
inv10 = -m10 / det
inv11 =  m00 / det
```

Modified upper block:

```text
c00 = inv00 * u0_i
c01 = inv01 * u1_i
c10 = inv10 * u0_i
c11 = inv11 * u1_i
```

Modified RHS:

```text
y0 = inv00 * r0 + inv01 * r1
y1 = inv10 * r0 + inv11 * r1
```

Backward:

```text
x0_i = y0_i - c00_i * x0_next - c01_i * x1_next
x1_i = y1_i - c10_i * x0_next - c11_i * x1_next
```

### Pros

```text
exact
stable
close to current Thomas algorithm
much better GPU layout candidate
directly inspired by many-small-systems PTA literature
```

### Cons

```text
still sequential in Nx
requires XB packing or direct XB RHS construction
```

### Recommended first double-cable production candidate

This is the first backend I would seriously optimize.

---

## 3.4 PTA block Thomas tiled

### Name

```text
PTA_BLOCK_THOMAS_TILED
```

### Layout

```text
rhs0[n_tiles, Nx_pad, BLOCK_B]
rhs1[n_tiles, Nx_pad, BLOCK_B]
```

### Scratch per tile

Need to store:

```text
c00
c01
c10
c11
y0
y1
```

Scratch size:

```text
6 × Nx_pad × BLOCK_B × sizeof(dtype)
```

Example float32:

```text
Nx_pad = 64,  BLOCK_B = 32  -> ~49 KB
Nx_pad = 128, BLOCK_B = 16  -> ~49 KB
Nx_pad = 128, BLOCK_B = 32  -> ~98 KB
```

Suggested candidates:

```text
Nx_pad = 32:  BLOCK_B = 64
Nx_pad = 64:  BLOCK_B = 32 or 64
Nx_pad = 128: BLOCK_B = 16 or 32
```

### Pros

```text
best fit for Pallas/CUDA
explicit occupancy/scratch tradeoff
likely production layout if custom kernel is used
```

### Cons

```text
packing/unpacking complexity
must benchmark BLOCK_B carefully
```

---

## 3.5 PCR / cyclic reduction SoA

### Name

```text
PCR_SOA
```

### Idea

Parallel cyclic reduction eliminates neighbors at increasing strides:

```text
stride = 1
stride = 2
stride = 4
stride = 8
...
```

After `ceil(log2(Nx))` stages, rows are independent.

### Pros

```text
exact
parallel in Nx
better theoretical GPU depth
good for larger Nx or smaller B
```

### Cons

```text
more arithmetic
more temporaries
more gather/scatter
may lose to optimized Thomas for Nx=30-100
```

### Important point

PCR should be compared against:

```text
best PTA_BLOCK_THOMAS
```

not against an unoptimized current Thomas implementation.

---

## 3.6 Hybrid PCR + Thomas

### Name

```text
PCR_THOMAS_HYBRID
```

### Idea

Do a few PCR stages to reduce dependency depth, then solve small remaining independent blocks with Thomas.

Example:

```text
Nx_pad = 64

PCR strides:
    1, 2, 4

then local Thomas on blocks of size 8 or 16
```

### Pros

```text
less sequential than Thomas
less arithmetic than full PCR
potentially good compromise for Nx=64/128
```

### Cons

```text
more complex
must be carefully validated
```

### When to test

Only after:

```text
PTA_BLOCK_THOMAS_XB/TILED
PCR_SOA
```

are benchmarked.

---

## 3.7 Split two-rail iterative solver

### Names

```text
SPLIT_JACOBI_FIXED_K
SPLIT_GAUSS_SEIDEL_FIXED_K
SPLIT_RICHARDSON_FIXED_K
```

### Reformulation

Write the double-cable system as:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

Iterative Gauss-Seidel:

```text
given Ve^k:
    solve Ti Vi^{k+1} = bi - Cie Ve^k

given Vi^{k+1}:
    solve Te Ve^{k+1} = be - Cei Vi^{k+1}
```

Jacobi:

```text
given Vi^k, Ve^k:
    solve Ti Vi^{k+1} = bi - Cie Ve^k
    solve Te Ve^{k+1} = be - Cei Vi^k
```

### Why it is attractive

Each iteration becomes:

```text
two scalar tridiagonal solves
```

which can reuse the single-cable PTA solver.

Effective GPU work:

```text
B_effective ≈ 2 × K × B
```

### Exactness

```text
fixed K:
    approximate

iterate to convergence:
    exact numerically
    less GPU-friendly if K is dynamic
```

### Initialization

Use previous time step:

```text
Vi_init = Vi_previous
Ve_init = Ve_previous
```

This may reduce required iterations because consecutive time steps are close.

### Validation metrics

Must check:

```text
residual ||Ax-b|| / ||b||
Vi / Ve / Vm error
activation agreement
threshold error
first spike time
conduction velocity
recruitment curve
```

### Pros

```text
very GPU-friendly
reuses scalar solver
potentially fast for screening
```

### Cons

```text
fixed-K is not exact
convergence may be slow if coupling is strong
```

### Role

Good fast approximate backend, or exact backend if iterated to convergence.

---

## 3.8 Associative backward scan

### Name

```text
ASSOCIATIVE_BACKWARD
```

### Idea

After Thomas forward elimination:

```text
x_i = y_i - C_i x_{i+1}
```

This is an affine transform:

```text
f_i(x) = A_i x + q_i
```

Affine transforms compose associatively:

```text
f(g(x)) = A_f A_g x + A_f q_g + q_f
```

Therefore the backward pass can be parallelized with `jax.lax.associative_scan`.

### Pros

```text
exact
low risk
can improve current Thomas modestly
```

### Cons

```text
forward scan remains sequential
likely modest speedup only
```

---

## 3.9 Full associative transfer scan

### Name

```text
ASSOCIATIVE_TRANSFER
```

### Idea

Rewrite each row as a state transition:

```text
x_{i+1} = -U_i^{-1}D_i x_i - U_i^{-1}L_i x_{i-1} + U_i^{-1}rhs_i
```

Define:

```text
y_i = [x_i, x_{i-1}, 1]
```

Then:

```text
y_{i+1} = M_i y_i
```

Prefix products of `M_i` can be computed via associative scan.

### Pros

```text
exact in principle
parallel prefix depth O(log Nx)
interesting research backend
```

### Cons

```text
can be numerically unstable
requires inverses of U_i or alternative formulation
dense 5x5 prototype may be slow
float32 stability uncertain
```

### Role

Prototype only after PTA and PCR baselines.

---

## 3.10 Pallas block Thomas

### Name

```text
PALLAS_PTA_BLOCK_THOMAS
```

### Goal

Implement the winning PTA layout as a custom Pallas kernel.

### Mapping

```text
grid = (n_tiles,)
one program handles [Nx_pad, BLOCK_B]
```

### Pros

```text
reduced JAX overhead
explicit tile control
better scratch control
closer to GPU literature implementations
```

### Cons

```text
more complex than JAX
still less low-level than CUDA
```

### Recommended first Pallas double-cable kernel

Start with:

```text
PALLAS_PTA_BLOCK_THOMAS
```

not PCR.

---

## 3.11 CUDA FFI block Thomas

### Name

```text
CUDA_FFI_PTA_BLOCK_THOMAS
```

### Goal

A true low-level CUDA implementation inspired by PTA / PfSolve / TDMA literature.

### Conceptual mapping

```text
one CUDA block = one tile
one thread/lane = one fiber in the tile
loop over Nx sequentially
coalesced access across B
scratch in shared memory
```

### Pros

```text
maximum control
can use warp-level instructions
can tune shared memory and occupancy
likely fastest possible exact backend
```

### Cons

```text
highest maintenance
JAX FFI required
CUDA-specific
more difficult CI/testing
```

### Role

Only after JAX/Pallas proves that PTA layout is the right direction.

---

---

# 4. Coefficient precomputation and factorization strategy

Coefficient precomputation should be treated as a major part of the solver design, not as a minor optimization. For many-fiber simulations, the solver should not rebuild geometry, axial coefficients, masks, padding, passive terms, or static RHS contributions at every time step.

The implementation should explicitly separate:

```text
static quantities:
    geometry, layout, axial couplings, padding masks, passive constants

slowly-changing quantities:
    stimulation footprint, waveform, amplitude candidates, observer settings

dynamic quantities:
    active membrane conductances, gating states, nonlinear current terms, RHS
```

The goal is to minimize per-time-step work so the GPU kernel spends most of its time solving or updating the actual dynamic state.

---

## 4.1 Level 0 — Geometry, masks, layout, and bucket precomputation

This applies to both single-cable and double-cable solvers.

Precompute once per execution bucket:

```text
Nx_true
Nx_pad = 32 / 64 / 128
real_compartment_mask[Nx_pad]
node_mask[Nx_pad]
section_type[Nx_pad]
dx[Nx_pad]
area[Nx_pad]
perimeter[Nx_pad]
diameter[Nx_pad]
axial_resistance / axial_conductance terms
left/right boundary condition coefficients
padding rows
recording / observer indices
```

For GPU execution, also precompute the packed layout:

```text
BX layout:
    [B, Nx_pad]

XB layout:
    [Nx_pad, B]

TILED layout:
    [n_tiles, Nx_pad, BLOCK_B]
```

The user-facing API can remain batch-first. The solver backend should receive already-packed arrays in its preferred layout.

Recommended cache object:

```python
@dataclass(frozen=True)
class GeometryLayoutCache:
    nx_true: int
    nx_pad: int
    layout: Literal["BX", "XB", "TILED"]
    block_b: int | None

    real_mask: Array
    node_mask: Array
    section_type: Array

    dx: Array
    area: Array
    perimeter: Array
    diameter: Array

    pack_indices: Array | None = None
    unpack_indices: Array | None = None
```

Padding rule:

```text
single-cable:
    lower = 0
    diag  = 1
    upper = 0
    rhs   = 0

double-cable:
    D = I_2
    L = 0
    U = 0
    rhs = [0, 0]
```

This makes padded rows electrically neutral and easy to slice away after the solve.

---

## 4.2 Level 1 — Static linear-system coefficients

Most of the linear-system structure is static for a fixed axon layout and time step `dt`.

For single-cable, precompute:

```text
lower_static[Nx_pad]
upper_static[Nx_pad]
axial_diag_static[Nx_pad]
cm_over_dt[Nx_pad]
passive_leak_g[Nx_pad]
passive_leak_rhs_const[Nx_pad]
diag_base[Nx_pad] = axial_diag_static + cm_over_dt + passive_leak_g
```

For double-cable, precompute:

```text
l0_static[Nx_pad]
l1_static[Nx_pad]
u0_static[Nx_pad]
u1_static[Nx_pad]

d00_base[Nx_pad]
d01_base[Nx_pad]
d10_base[Nx_pad]
d11_base[Nx_pad]

rhs0_static_terms[Nx_pad]
rhs1_static_terms[Nx_pad]
```

The base block should include static contributions such as:

```text
Cm/dt
passive leak
axial diagonal terms
periaxonal / myelin coupling terms
boundary condition terms
```

Dynamic contributions are added later inside the time loop.

Recommended cache objects:

```python
@dataclass(frozen=True)
class SingleCableLinearSystemCache:
    geometry: GeometryLayoutCache

    lower_static: Array
    diag_base: Array
    upper_static: Array

    cm_over_dt: Array
    passive_g: Array
    passive_rhs_const: Array

    thomas_c: Array | None = None
    thomas_inv_denom: Array | None = None
    can_use_prefactored: bool = False
```

```python
@dataclass(frozen=True)
class DoubleCableLinearSystemCache:
    geometry: GeometryLayoutCache

    l0_static: Array
    l1_static: Array
    u0_static: Array
    u1_static: Array

    d00_base: Array
    d01_base: Array
    d10_base: Array
    d11_base: Array

    rhs0_static_terms: Array
    rhs1_static_terms: Array

    block_thomas_c00: Array | None = None
    block_thomas_c01: Array | None = None
    block_thomas_c10: Array | None = None
    block_thomas_c11: Array | None = None

    block_thomas_inv_m00: Array | None = None
    block_thomas_inv_m01: Array | None = None
    block_thomas_inv_m10: Array | None = None
    block_thomas_inv_m11: Array | None = None

    can_use_prefactored: bool = False
```

---

## 4.3 Level 2 — Full Thomas pre-factorization

Full Thomas pre-factorization is possible when the matrix coefficients are constant over time.

That means:

```text
single-cable:
    lower, diag, upper do not change with t

double-cable:
    L, D, U do not change with t
```

This is common for:

```text
passive cables
linearized fixed-conductance models
some pseudo/effective models
some solver-only benchmark problems
```

It is less common for active axons because active membrane conductances usually change the diagonal terms at each time step.

---

## 4.4 Single-cable full pre-factorization

For scalar Thomas:

```text
lower_i x_{i-1} + diag_i x_i + upper_i x_{i+1} = rhs_i
```

Precompute once:

```text
denom_0 = diag_0
c_0 = upper_0 / denom_0

for i = 1..Nx-1:
    denom_i = diag_i - lower_i * c_{i-1}
    c_i = upper_i / denom_i
```

Store:

```text
c_i
inv_denom_i = 1 / denom_i
```

At each time step, solve only the RHS-dependent part:

```text
y_0 = rhs_0 * inv_denom_0

for i = 1..Nx-1:
    y_i = (rhs_i - lower_i * y_{i-1}) * inv_denom_i

x_last = y_last

for i = Nx-2..0:
    x_i = y_i - c_i * x_{i+1}
```

This removes repeated denominator computations and many divisions from the time loop.

Suggested solver names:

```text
PTA_SCALAR_PREFACTORED_XB
PTA_SCALAR_PREFACTORED_TILED
PALLAS_PTA_SCALAR_PREFACTORED
```

Use:

```python
if cache.can_use_prefactored:
    x = solve_scalar_prefactored(cache, rhs)
else:
    x = solve_scalar_dynamic(cache, diag_dynamic, rhs)
```

Expected impact: potentially large for passive or linear single-cable simulations, because the per-time-step solve becomes mostly multiply-add operations plus one backward substitution.

---

## 4.5 Double-cable full block pre-factorization

For double-cable, full pre-factorization is possible if the full block matrix is static.

The forward block-Thomas recurrence computes:

```text
M_i = D_i - L_i C_{i-1}
C_i = M_i^{-1} U_i
```

If `D_i`, `L_i`, and `U_i` are constant over time, then `C_i` and `M_i^{-1}` can be precomputed.

Store:

```text
C_i:
    c00_i
    c01_i
    c10_i
    c11_i

M_i^{-1}:
    inv00_i
    inv01_i
    inv10_i
    inv11_i
```

At each time step, solve only the dynamic RHS:

```text
r_i = rhs_i - L_i y_{i-1}
y_i = M_i^{-1} r_i
```

Then backward substitute:

```text
x_i = y_i - C_i x_{i+1}
```

Suggested solver names:

```text
PTA_BLOCK_THOMAS_PREFACTORED_XB
PTA_BLOCK_THOMAS_PREFACTORED_TILED
PALLAS_PTA_BLOCK_THOMAS_PREFACTORED
```

When to use:

```text
passive double-cable
fixed linearized double-cable
solver-only validation cases
not generally valid for fully active MRG-like nodes
```

Expected impact: useful but less broadly applicable than single-cable pre-factorization. It is still valuable for passive benchmark cases and for validating solver overhead independently of active membrane dynamics.

---

## 4.6 Level 3 — Partial precomputation for active membranes

Active models often change diagonal terms at every time step:

```text
single-cable:
    diag_i(t) = diag_base_i + g_active_i(t)
    rhs_i(t)  = rhs_base_i + active_rhs_i(t) + stimulus_rhs_i(t)

double-cable:
    d00_i(t) = d00_base_i + g_active_i(t)
    rhs0_i(t) = rhs0_base_i + active_rhs_i(t) + stimulus_rhs0_i(t)
```

Often only node compartments are active:

```text
NODE:
    dynamic active conductance

MYSA / FLUT / STIN:
    passive or effective passive
```

This suggests a hybrid strategy:

```text
precompute all static arrays
update only node-local dynamic diagonal/RHS terms
avoid recomputing passive internodal rows
```

Implementation pattern:

```python
diag = cache.diag_base
rhs = cache.rhs_base

active_g, active_rhs = membrane_model(...)

diag = diag + node_mask * active_g
rhs = rhs + node_mask * active_rhs
rhs = rhs + stimulus_rhs
```

For double-cable:

```python
d00 = cache.d00_base + node_mask * active_g
d01 = cache.d01_base
d10 = cache.d10_base
d11 = cache.d11_base

rhs0 = cache.rhs0_static_terms + active_rhs + stimulus_rhs0
rhs1 = cache.rhs1_static_terms + stimulus_rhs1
```

Optimization target:

```text
avoid building full dense dynamic arrays when only a few compartments are active
```

But for small `Nx`, dense masked updates may still be faster than sparse/scattered updates. Benchmark both:

```text
dense masked update
node-only indexed update
```

---

## 4.7 Level 4 — RHS precomputation and factorized stimulation

The RHS often contains:

```text
previous voltage term
passive reversal term
active ionic current term
extracellular stimulation term
intracellular stimulation term
```

Separate static, dynamic, and factorized pieces.

Static RHS terms:

```text
passive_rhs_const = g_leak * E_leak
resting offsets if used
fixed source terms
```

Dynamic RHS terms:

```text
Cm/dt * V_previous
active current linearization terms
gating-dependent terms
```

Factorized extracellular drive:

```text
Vext[b, t, x] = footprint[b, x] * waveform[t]
```

Represent internally as:

```text
footprint[Nx, B]
waveform[Nt]
```

Then inside the time loop:

```python
vext_t = waveform[t] * footprint
stim_rhs_t = build_rhs_from_vext(vext_t)
```

Do not materialize:

```text
Vext[B, Nt, Nx]
```

unless unavoidable.

For threshold sweeps, include amplitudes in the batch:

```text
B_effective = fibers × amplitudes
```

If waveform is normalized:

```python
vext_t = amplitude_batch[None, :] * waveform[t] * footprint
```

---

## 4.8 Interaction with tiled layout

Precompute caches directly in solver layout.

Preferred:

```text
cache stored as [Nx_pad, B]
or [n_tiles, Nx_pad, BLOCK_B]
```

Avoid repeatedly doing this inside the time loop:

```python
jnp.swapaxes(...)
reshape(...)
pad(...)
```

Packing should occur once per batch/bucket.

Recommended packing flow:

```text
1. Build public AxonScope arrays.
2. Group into execution bucket.
3. Pad to Nx_pad.
4. Pack to solver layout:
       [Nx_pad, B]
       or [n_tiles, Nx_pad, BLOCK_B]
5. Build LinearSystemCache in that layout.
6. Run time loop using cached layout.
7. Unpack only compact outputs or selected traces.
```

---

## 4.9 Suggested API additions

Solver cache builders:

```python
def build_single_cable_linear_system_cache(
    axon_batch,
    *,
    dt,
    nx_pad,
    layout: Literal["BX", "XB", "TILED"],
    block_b: int | None,
    dtype,
) -> SingleCableLinearSystemCache:
    ...
```

```python
def build_double_cable_linear_system_cache(
    axon_batch,
    *,
    dt,
    nx_pad,
    layout: Literal["BX", "XB", "TILED"],
    block_b: int | None,
    dtype,
) -> DoubleCableLinearSystemCache:
    ...
```

Solver entry points:

```python
def solve_single_cable_step(
    cache: SingleCableLinearSystemCache,
    state,
    dynamic_terms,
    stimulus_terms,
    solver: SingleCableLinearSolver,
):
    ...
```

```python
def solve_double_cable_step(
    cache: DoubleCableLinearSystemCache,
    state,
    dynamic_terms,
    stimulus_terms,
    solver: DoubleCableLinearSolver,
):
    ...
```

Cache capability flags:

```python
cache.can_use_prefactored
cache.has_active_membrane
cache.has_dynamic_diagonal
cache.has_factorized_stimulus
cache.layout
cache.nx_pad
```

---

## 4.10 Benchmark matrix for precomputation

Add solver benchmarks that explicitly compare:

```text
dynamic coefficients
partial precomputation
full pre-factorization
factorized Vext
dense Vext
```

Single-cable benchmark cases:

```text
passive constant coefficients:
    expect pre-factorization to help strongly

active nodes only:
    partial precompute

fully active every compartment:
    less precompute benefit
```

Double-cable benchmark cases:

```text
passive double-cable:
    full block pre-factorization possible

MRG-like active nodes:
    partial precompute

pathological / active internodes:
    less precompute benefit
```

Metrics:

```text
solver-only time
time-step time
end-to-end time
memory traffic
number of dynamic arrays allocated
kernel count
speedup vs no-cache baseline
```

---

## 4.11 Go / no-go criteria

Single-cable pre-factorization:

```text
keep if speedup >= 1.3x for passive/linear cases
```

Double-cable block pre-factorization:

```text
keep if speedup >= 1.2x in passive/linear double-cable cases
```

Partial precomputation for active models:

```text
keep if end-to-end speedup >= 10-20%
or memory allocation / kernel count decreases clearly
```

Factorized Vext:

```text
keep if Vext memory decreases substantially
and end-to-end speed improves or allows larger B_effective
```

---

## 4.12 Updated implementation order including precomputation

```text
Step 1:
    Implement LinearSystemCache objects for single and double cable.

Step 2:
    Move static geometry / axial / passive coefficients into caches.

Step 3:
    Implement PTA_SCALAR_XB and PTA_BLOCK_THOMAS_XB using caches.

Step 4:
    Add full scalar pre-factorization for constant-coefficient single-cable.

Step 5:
    Add full block pre-factorization for passive double-cable.

Step 6:
    Add partial active-node dynamic updates.

Step 7:
    Add factorized Vext RHS construction.

Step 8:
    Add tiled cache layout for Pallas/CUDA-ready backends.

Step 9:
    Benchmark all precompute levels independently.
```

---

## 4.13 Practical conclusion

Coefficient precomputation is not optional. It should be part of the core GPU solver design.

For single-cable:

```text
full Thomas pre-factorization can be a major win when coefficients are static
partial precompute still helps active models
```

For double-cable:

```text
static geometry and block coefficients must be cached
full block pre-factorization is useful for passive/linear cases
active-node models mostly benefit from partial precompute
```

Combined with:

```text
SoA layout
[Nx, B] or tiled packing
Nx buckets
compact observers
factorized Vext
```

this becomes the realistic production direction for AxonScope GPU simulation.


---

# 5. Scheduler / batching options for both single and double cable

## 4.1 Execution bucket key

Group simulations by:

```text
mode: single / double
solver backend
Nx_pad
dtype
recording mode
Iinj kind
Vext representation
membrane signature
geometry signature
```

---

## 4.2 Coalesce groups

Instead of launching many small groups:

```text
group A: B=256, Nx_pad=64
group B: B=512, Nx_pad=64
group C: B=128, Nx_pad=64
```

merge into:

```text
one bucket: B=896, Nx_pad=64
```

---

## 4.3 Async enqueue

Optional:

```text
enqueue multiple independent groups
block_until_ready once or at memory flush points
```

This may reduce host/device bubbles but does not guarantee true simultaneous GPU execution.

---

## 4.4 Preferred priority

```text
1. coalesce larger batches
2. bucket Nx
3. compact outputs
4. factorize Vext
5. optional async groups
```

---

# 6. Recommended implementation order

## Phase 0 — benchmarking and tracing

Add:

```text
benchmark/solvers/bench_single_cable_linear_solvers.py
benchmark/solvers/bench_double_cable_linear_solvers.py
benchmark/solvers/profile_solver_jax.py
```

Use:

```text
block_until_ready
compile/runtime separation
JAX profiler
transfer_guard=log
persistent compilation cache
```

---

## Phase 1 — single-cable PTA

Implement:

```text
PTA_SCALAR_XB
PTA_SCALAR_TILED
PTA_SCALAR_PREFACTORED if possible
```

Why first?

```text
simpler than double-cable
directly matches TDMA literature
validates layout and scheduler design
```

---

## Phase 2 — double-cable PTA block Thomas

Implement:

```text
PTA_BLOCK_THOMAS_XB
PTA_BLOCK_THOMAS_TILED
```

This is the main exact double-cable production candidate.

---

## Phase 3 — scheduler integration

Implement:

```text
Nx buckets
group coalescing
B_effective batching
compact observers
factorized Vext
```

---

## Phase 4 — PCR / split iterative comparison

Implement and compare:

```text
PCR_SOA
PCR_THOMAS_HYBRID
SPLIT_GAUSS_SEIDEL_FIXED_K
```

Compare against best PTA backend, not current baseline only.

---

## Phase 5 — associative scan prototypes

Implement:

```text
ASSOCIATIVE_BACKWARD
ASSOCIATIVE_TRANSFER
```

Keep only if they beat the best PTA/PCR options or provide useful insight.

---

## Phase 6 — Pallas

Implement:

```text
PALLAS_PTA_SCALAR
PALLAS_PTA_BLOCK_THOMAS
```

Only after JAX prototypes identify the winning layout.

---

## Phase 7 — CUDA FFI

Implement:

```text
CUDA_FFI_PTA_SCALAR
CUDA_FFI_PTA_BLOCK_THOMAS
```

only if Pallas is insufficient and performance justifies maintenance.

---

# 7. Go / no-go criteria

## Single-cable

Keep `PTA_SCALAR_XB` if:

```text
speedup vs current single solver >= 1.3x
```

Keep `PTA_SCALAR_TILED` if:

```text
speedup vs PTA_SCALAR_XB >= 1.2x
```

Implement Pallas if:

```text
JAX PTA is faster but trace shows temporary/kernel overhead
```

---

## Double-cable

Keep `PTA_BLOCK_THOMAS_XB` if:

```text
speedup vs current double Thomas >= 1.5x for B>=1024, Nx<=100
```

Keep tiled layout if:

```text
tiled beats XB by >= 1.2x
```

Keep PCR if:

```text
PCR beats best PTA for Nx>=64 or Nx>=96
```

Keep split iterative if:

```text
K=2 or K=4 is much faster
and threshold/recruitment/conduction metrics remain acceptable
```

Move to CUDA FFI if:

```text
Pallas or JAX proves algorithm/layout is good
but still leaves clear low-level performance on the table
```

---

# 8. Expected best choices

## Single-cable likely winner

```text
PTA_SCALAR_XB
or
PTA_SCALAR_TILED / PALLAS_PTA_SCALAR
```

If coefficients are constant:

```text
PTA_SCALAR_PREFACTORED
```

may be best.

---

## Double-cable likely winner

For exact production:

```text
PTA_BLOCK_THOMAS_XB
or
PTA_BLOCK_THOMAS_TILED / PALLAS_PTA_BLOCK_THOMAS
```

For fast approximate screening:

```text
SPLIT_GAUSS_SEIDEL_FIXED_K
```

For larger `Nx`:

```text
PCR_SOA
or
PCR_THOMAS_HYBRID
```

---

# 9. Hardware implications

These solvers benefit from:

```text
high memory bandwidth
large VRAM for large B_effective
good CUDA/JAX/Pallas support
```

Recommended platforms for benchmarking:

```text
local development:
    RTX 4090 / RTX 5090 / RTX PRO 6000 depending on memory needs

cloud reference:
    A100 / H100 / H200

TPU:
    worth testing mainly for JAX-pure split iterative, PCR, associative scan
    less obvious for custom CUDA-style PTA
```

The optimized PTA path is most naturally NVIDIA/CUDA-oriented.

---

# 10. References

## GPU tridiagonal / TDMA literature

1. PfSolve / Parallel Factorization Solver for tridiagonal and bidiagonal matrices.
   - DOI: `10.1145/3716171`
   - https://dl.acm.org/doi/10.1145/3716171
   - https://www.research-collection.ethz.ch/items/fab84ea7-24e3-48ec-bd12-85477e1ef4e3

2. Mechanics & Industry 2020, Parallel Thomas Algorithm for GPUs.
   - https://www.mechanics-industry.org/articles/meca/full_html/2020/03/mi170262/mi170262.html

3. NVIDIA GTC 2009 tridiagonal solver slides.
   - https://www.nvidia.com/content/gtc/documents/1058_gtc09.pdf

4. Pipelined TDMA for GPUs.
   - arXiv `2509.03933v1`
   - https://arxiv.org/html/2509.03933v1

5. PaScaL_TDMA.
   - https://github.com/xccels/PaScaL_TDMA

## Hines / neuronal GPU solvers

6. Huber, F. (2018). Efficient Tree Solver for Hines Matrices on the GPU.
   - arXiv `1810.12742`
   - https://arxiv.org/abs/1810.12742
   - https://arxiv.org/pdf/1810.12742

## Axon biology / double-cable motivation

7. Abdollahi, N. and Prescott, S. A. (2024). Impact of Extracellular Current Flow on Action Potential Propagation in Myelinated Axons.
   - Journal of Neuroscience, 44(26):e0569242024.
   - DOI: `10.1523/JNEUROSCI.0569-24.2024`
   - Uploaded PDF in this conversation: `e0569242024.full.pdf`

## JAX implementation tools

8. JAX asynchronous dispatch.
   - https://docs.jax.dev/en/latest/async_dispatch.html

9. JAX benchmarking guide.
   - https://docs.jax.dev/en/latest/benchmarking.html

10. JAX profiler.
   - https://docs.jax.dev/en/latest/profiling.html

11. JAX associative scan.
   - https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html

12. JAX Pallas.
   - https://docs.jax.dev/en/latest/pallas/index.html

---

# 11. Final recommendation

If only one path could be implemented first:

```text
Single-cable:
    PTA_SCALAR_XB
    then PTA_SCALAR_TILED / Pallas

Double-cable:
    PTA_BLOCK_THOMAS_XB
    then PTA_BLOCK_THOMAS_TILED / Pallas

Scheduler:
    Nx bucketization
    group coalescing
    compact observers
    factorized Vext
```

This approach is less theoretically elegant than PCR or associative scan, but it is the most aligned with the actual workload:

```text
many independent short systems
small Nx
large B
GPU memory/coalescing critical
exact double-cable reference still needed
```
