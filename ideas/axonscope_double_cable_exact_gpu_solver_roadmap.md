# AxonScope Double-Cable GPU Solver Roadmap

## Objective

Make the **exact double-cable solver** scale better on GPU for the target regime:

```text
Nx = 30-100 compartments per fiber
B  > 500 fibers
Nt = many time steps
main use cases = threshold, activation, recruitment, conduction validation
```

This roadmap focuses on rewriting the **linear solve** used inside each implicit double-cable time step. It does **not** replace the double-cable model with a pseudo-single-cable approximation.

The scientific reason to keep an exact double-cable path is that recent work by Abdollahi & Prescott (2024) shows that axial, submyelin, transmyelin, and extramyelin current pathways materially affect conduction velocity, conduction reliability, energy efficiency, demyelination sensitivity, and ephaptic effects. In particular, extracellular boundary conditions change how much current reaches the next node versus how much leaks through myelin/extracellular pathways.

---

## Current AxonScope snapshot

Updated on 2026-06-16 after checking `src/axonscope/solvers/`,
`benchmark/hotpaths/`, `tests/unit/solvers/`, and
`benchmark/pseudo_double/`.

Current exact double-cable solver selection is already wired through:

```text
BatchOptions.double_cable_block_solver
benchmark/hotpaths/run.py --double-cable-block-solver
```

The currently implemented choices are:

| Choice | Current behavior |
| --- | --- |
| `auto` | CPU/default backends resolve to `thomas`; GPU-like backends resolve to `pcr_adaptive`. |
| `thomas` | Exact block-Thomas scan. |
| `pcr` | Exact matrix-layout PCR variant. |
| `pcr_soa` | Exact struct-of-arrays PCR variant. |
| `pcr_adaptive` | `pcr_soa` for `B <= 4096`, matrix-layout `pcr` for larger batches. This threshold is calibrated from the 2026-06-16 P100 solver-only sweep and should be revisited with more devices. |

Do not add more solver-choice names to public surfaces until they exist in code,
have Thomas-vs-candidate tests, and have benchmark evidence. Benchmark-only
names may exist in solver-focused runners while gathering that evidence, but
must stay out of `BatchOptions.double_cable_block_solver` and `auto`.

Pseudo-double and pseudo-MRG modes are now on standby. They live only under
`benchmark/pseudo_double/` as validation harness candidates and must not be
used as exact double-cable solver options or selected by `auto`.

---

## Current bottleneck

The current double-cable solve is a 2x2 block-tridiagonal system at each implicit time step:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = r_i

x_i = [Vi_i, Ve_i]^T

D_i = 2x2 diagonal block
L_i = lower 2x2 off-block, diagonal in the Vi/Ve channels
U_i = upper 2x2 off-block, diagonal in the Vi/Ve channels
```

The current exact solver is essentially a specialized block Thomas algorithm:

```text
forward elimination:     i = 0 -> Nx-1
backward substitution:   i = Nx-1 -> 0
```

This is excellent for CPU and acceptable for small workloads, but it exposes little spatial parallelism to the GPU. For GPU, the current solver mostly parallelizes across `B`, not across `Nx`.

For `B > 500`, that can be enough to see some GPU benefit, but it is not ideal. The goal of this roadmap is to expose more parallel work over:

```text
B x Nx
```

inside every time step.

---

## Core proposals

We will implement and compare exact double-cable solver backends. The current
code already implements the first group:

```text
implemented now:
    thomas
    pcr
    pcr_soa
    pcr_adaptive
    auto
```

Candidate future names should stay roadmap-only until implemented:

```python
class DoubleCableLinearSolver(str, Enum):
    THOMAS = "thomas"
    PCR = "pcr"
    PCR_SOA = "pcr_soa"
    PCR_ADAPTIVE = "pcr_adaptive"
    PCR_HYBRID = "pcr_hybrid"
    SPLIT_JACOBI = "split_jacobi"
    SPLIT_GAUSS_SEIDEL = "split_gauss_seidel"
    SPLIT_RICHARDSON = "split_richardson"
    ASSOCIATIVE_BACKWARD = "associative_backward"
    ASSOCIATIVE_TRANSFER = "associative_transfer"
    PALLAS_THOMAS = "pallas_thomas"
    PALLAS_PCR_HYBRID = "pallas_pcr_hybrid"
    AUTO = "auto"
```

The current implementation uses Literal string choices in `options.py`, not this
enum. Promote to an enum only if it improves the public/solver boundary after
the option set stabilizes.

The roadmap is phased:

```text
Phase 0: profiling and solver-only baseline
Phase 1: simple exact JAX direct backends
         - clean API
         - PCR_SOA
         - batch-native layout
         - padding Nx -> 32/64/128
         - hybrid PCR/Thomas optional
Phase 1.5: two-rail split iterative solver
         - rewrite one double-cable solve as two coupled scalar cable solves
         - split Jacobi, split Gauss-Seidel, fixed-K Richardson
         - reuse the fast single-cable/tridiagonal GPU path
Phase 2: associative scan exact solvers
         - backward associative scan
         - full transfer-matrix associative scan
Phase 3: Pallas custom kernels
         - pallas_thomas_small_nx
         - pallas_hybrid_pcr_thomas_small_nx
```

---

# Phase 0 — Baseline and tracing

## Goal

Before rewriting the solver, establish a reliable performance and correctness baseline.

This phase answers:

```text
1. Is the solve itself the bottleneck?
2. Is the GPU idle between small kernels?
3. Is full Vm output dominating?
4. Is dense zero Iinj being materialized?
5. Is current double-cable scaling limited by Nx scans?
6. What is the crossover point for B and Nx?
```

## Actions

### 0.1 Add a solver-only benchmark

Status on 2026-06-16: implemented as:

```text
benchmark/solvers/bench_double_cable_linear_solvers.py
```

This benchmark does not build full axon models. It directly generates
deterministic, well-conditioned 2x2 block-tridiagonal systems with shapes:

```text
B  = 1, 8, 128, 512, 1024, 2048, 4096
Nx = 16, 32, 51, 64, 96, 100, 128
dtype = float32, float64
solver = thomas, pcr, pcr_soa, pcr_adaptive
```

It measures:

```text
compile time
first run time
steady-state min/median/p95 time
node-solves/s = B * Nx / time
max_abs_error vs Thomas float64
max_rel_error vs Thomas float64
max/median block residual norm
```

All timings call `jax.block_until_ready(out)`. Future candidate solver names
such as `pcr_hybrid`, associative scans, split solvers, and Pallas variants
should be added here only after the backend exists and has Thomas-equivalence
tests.

### 0.2 Add end-to-end double-cable benchmark

Status on 2026-06-16: implemented as:

```text
benchmark/solvers/bench_double_cable_end_to_end.py
benchmark/solvers/colab_double_cable_end_to_end.ipynb
```

The runner builds homogeneous MRG-like double-cable batches, prepares runtime
arrays, materializes dense `Vext`, optionally materializes dense `Iinj`, and
runs `DoubleCableBatchKernel` directly. This keeps the axes needed for Phase
0.2 without relying on public APIs that cannot force dense-zero current input.

Configured matrix:

```text
B  = configurable; Colab defaults include 512, 1024, 2048
Nx = configurable target compartments; Colab defaults include 51, 64, 96
Nt = configurable; Colab defaults include 500, 1000
recording = none, center, full
Iinj = none, dense_zero, nonzero
Vext = dense
solver = auto, thomas, pcr_adaptive, plus explicit variants as needed
```

Reported metrics:

```text
setup time
runtime preparation time
dense Vext materialization time
dense Iinj materialization time
kernel enqueue median/min
kernel wait median/min
total setup time
total setup + median kernel time
case wall time
trace location
Vm output bytes
input Vext bytes
input Iinj bytes
actual MRG Nx
```

Future extension: add factorized `Vext` once the exact double-cable kernel can
consume it without first materializing a dense `(B, Nt, Nx)` tensor.

### 0.3 Add JAX profiler traces

Status on 2026-06-16: linear-solver tracing is implemented as:

```text
benchmark/solvers/profile_double_cable_linear_solvers.py
```

Status on 2026-06-17: the main linear-solver runner also supports
`--jax-trace`, and Kaggle exposes a focused P100 preset:

```text
linear_pcr_soa_trace
```

This preset traces `pcr`, `pcr_soa`, and `pcr_adaptive` at `B=2048/4096` and
`Nx=51/96`, skips the Thomas64 reference to keep the profiler focused, and
packages `jax_traces/` inside the downloaded Kaggle output archive.

P100 result on 2026-06-17:

```text
run: benchmark/results/kaggle/20260617_214032_linear_pcr_soa_trace_NvidiaTeslaP100
solvers: pcr, pcr_soa, pcr_adaptive
B: 2048, 4096
Nx: 51, 96
dtype: float32
```

The trace confirms that `pcr_soa` is the better pure-JAX exact PCR layout for
these focused cases. It is `1.09x-1.38x` faster than matrix-layout `pcr` on
steady medians, and cuts device fusion events from `31-48` for `pcr` to `7-13`
for `pcr_soa`. Matrix `pcr` spends time in many `loop_slice_fusion_*` kernels;
`pcr_soa` mostly spends time in `loop_select_subtract_fusion_*`. Treat the
profiler host/solve scopes as noisy because `jax.profiler` overhead dominates
them; use the steady medians and device event counts as the actionable signal.

Decision: do not optimize matrix-layout PCR first. Continue with
batch-native `pcr_soa` and target its per-stage `where`/boundary-mask/gather
behavior before more threshold tuning.

Run at minimum:

```bash
python benchmark/solvers/profile_double_cable_linear_solvers.py \
  --solver thomas \
  --batch-size 1024 \
  --nx 64 \
  --trace-dir /tmp/as-thomas-B1024-Nx64

python benchmark/solvers/profile_double_cable_linear_solvers.py \
  --solver pcr_soa \
  --batch-size 1024 \
  --nx 64 \
  --trace-dir /tmp/as-pcr-B1024-Nx64
```

Kaggle P100 command:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_trace \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

End-to-end double-cable traces that include time stepping, stimulation inputs,
and recording policies remain part of Phase 0.2.

## Success criteria

Phase 0 is complete when we have:

```text
1. Solver-only timings.
2. End-to-end timings.
3. JAX trace for at least Thomas and PCR_SOA.
4. Correctness baseline using Thomas float64.
5. A written table of crossover points.
```

---

# Phase 1 — Simple exact JAX solver backends

## Goal

Make a clean, exact, JAX-native solver stack before attempting more complex algorithms.

This phase should deliver immediate improvements and create the infrastructure needed for Phase 2 and Phase 3.

---

## Phase 1A — Clean and stabilize solver selection API

Status on 2026-06-16: partially done. `BatchOptions.double_cable_block_solver`
already exists, hotpath CLI passthrough exists, and tests cover
`auto`/`thomas`/`pcr`/`pcr_soa`/`pcr_adaptive`. The remaining work is to decide
whether this stays as Literal string choices or becomes a typed enum after the
option set stabilizes.

### Files to modify

```text
src/axonscope/solvers/common.py
src/axonscope/solvers/batch_kernels.py
src/axonscope/solvers/options.py
benchmark/hotpaths/run.py
```

### Possible enum after stabilization

```python
class DoubleCableLinearSolver(str, Enum):
    THOMAS = "thomas"
    PCR = "pcr"
    PCR_SOA = "pcr_soa"
    PCR_ADAPTIVE = "pcr_adaptive"
    PCR_HYBRID = "pcr_hybrid"
    SPLIT_JACOBI = "split_jacobi"
    SPLIT_GAUSS_SEIDEL = "split_gauss_seidel"
    SPLIT_RICHARDSON = "split_richardson"
    ASSOCIATIVE_BACKWARD = "associative_backward"
    ASSOCIATIVE_TRANSFER = "associative_transfer"
    AUTO = "auto"
```

Do not introduce `PCR_HYBRID`, split, associative, or Pallas names into
`BatchOptions` until those backends exist and tests show numerical equivalence
to Thomas. Pallas variants can be added later in Phase 3.

### Add dispatch

```python
def solve_double_cable_linear_system(
    *,
    a00,
    a01,
    a10,
    a11,
    off0,
    off1,
    rhs0,
    rhs1,
    solver: DoubleCableLinearSolver,
):
    if solver == DoubleCableLinearSolver.THOMAS:
        return solve_block_tridiagonal_2x2_scalar(...)

    if solver == DoubleCableLinearSolver.PCR:
        return solve_block_tridiagonal_2x2_pcr(...)

    if solver == DoubleCableLinearSolver.PCR_SOA:
        return solve_block_tridiagonal_2x2_pcr_soa(...)

    if solver == DoubleCableLinearSolver.PCR_HYBRID:
        return solve_block_tridiagonal_2x2_pcr_hybrid_soa(...)

    if solver == DoubleCableLinearSolver.SPLIT_JACOBI:
        return solve_double_cable_two_rail_split(..., method="jacobi")

    if solver == DoubleCableLinearSolver.SPLIT_GAUSS_SEIDEL:
        return solve_double_cable_two_rail_split(..., method="gauss_seidel")

    if solver == DoubleCableLinearSolver.SPLIT_RICHARDSON:
        return solve_double_cable_two_rail_richardson(...)

    if solver == DoubleCableLinearSolver.ASSOCIATIVE_BACKWARD:
        return solve_block_tridiagonal_2x2_assoc_backward(...)

    if solver == DoubleCableLinearSolver.ASSOCIATIVE_TRANSFER:
        return solve_block_tridiagonal_2x2_transfer_scan(...)

    raise ValueError(...)
```

### Current AUTO policy

Current policy:

```python
def choose_double_cable_solver(B: int, Nx: int, device: str) -> DoubleCableLinearSolver:
    if device == "cpu":
        return THOMAS

    if device in {"cuda", "gpu", "metal", "rocm"}:
        return PCR_ADAPTIVE

    return THOMAS
```

Inside the current batch kernel, `PCR_ADAPTIVE` resolves by batch size:

```python
if B <= 4096:
    return PCR_SOA
return PCR
```

This policy should become data-driven after solver-only and end-to-end
benchmarks cover enough `B`, `Nx`, device, output-policy, and dtype cases.

---

## Phase 1B — Make PCR_SOA official

The branch already contains a SoA parallel cyclic reduction solver. Make it an official backend.

### Algorithm

Parallel cyclic reduction eliminates neighbors at exponentially increasing strides:

```text
stage 0: stride = 1
stage 1: stride = 2
stage 2: stride = 4
stage 3: stride = 8
...
```

Each row remains active at each stage. After `ceil(log2(Nx))` stages, every row is independent:

```text
D_i_final x_i = rhs_i_final
```

Then each row does a local 2x2 solve.

### Why this is GPU-friendly

Thomas has dependency depth:

```text
O(Nx)
```

PCR has dependency depth:

```text
O(log Nx)
```

For `Nx=64`, that means roughly 6 parallel stages instead of about 128 sequential forward/backward steps.

### Implementation rules

Use SoA arrays:

```text
d00, d01, d10, d11      [Nx] or [B, Nx]
lower0, lower1          [Nx] or [B, Nx]
upper0, upper1          [Nx] or [B, Nx]
rhs0, rhs1              [B, Nx]
```

Avoid:

```text
[B, Nx, 2, 2]
```

in performance-critical code unless it is only a prototype.

### Correctness tests

Add:

```text
tests/solvers/test_double_cable_linear_solvers.py
```

Test:

```text
Thomas vs PCR_SOA
Nx = 2, 3, 4, 8, 16, 32, 51, 64, 96, 100
dtype = float64, float32
random diagonally-dominant systems
physical systems from real axon builder
```

Tolerance:

```text
float64 max_abs_error < 1e-9
float32 max_abs_error < 1e-5
```

For physical voltage traces:

```text
Vm max_abs_error < 1e-4 mV or physiologically negligible
activation/threshold unchanged
```

---

## Phase 1C — Batch-native PCR_SOA

Status on 2026-06-16: implemented as
`solve_block_tridiagonal_2x2_pcr_soa_batched(...)`. The solver accepts shared
`[Nx]` coefficients or per-row `[B, Nx]` coefficients and operates directly on
batch-first right-hand sides `[B, Nx]`. The solver-focused benchmark uses this
batch-native path for `pcr_soa` / `pcr_adaptive` when they resolve to SoA, and
`DoubleCableBatchKernel` uses it for array-output double-cable chunks when the
resolved kernel solver is `pcr_soa` and `B >= 2048`. The P100 E2E run showed
that the batch-native route improves large batches but regresses `B=512`, so
small batches keep the previous per-fiber `vmap` route for now.

Remaining work: thread the same batch-aware solve through the observer-only
path. The observer scan still evaluates each fiber under `vmap` before calling
the one-fiber block solver.

## Problem

If PCR is called as:

```python
jax.vmap(solve_one_fiber_pcr)(...)
```

XLA may do a good job, but we do not fully control layout and broadcasting.

The target workload is large batch and small Nx:

```text
B  > 500
Nx = 30-100
```

Therefore the solver should explicitly operate on `[B, Nx]`.

## Add function

```python
def solve_block_tridiagonal_2x2_pcr_soa_batched(
    a00, a01, a10, a11,  # [Nx] or [B,Nx]
    off0, off1,          # [Nx-1] or [B,Nx-1]
    rhs0, rhs1,          # [B,Nx]
) -> tuple[Array, Array]:
    ...
```

Internally broadcast `[Nx]` coefficients to `[B, Nx]` only virtually where possible.

## Layout

Preferred layout:

```text
rhs0, rhs1 = [B, Nx]
```

Within PCR stage, every operation touches `B x Nx` elements.

If profiling shows poor memory coalescing, test transposed layout:

```text
rhs0, rhs1 = [Nx, B]
```

But start with `[B, Nx]`, because this matches batch-first AxonScope patterns.

Status on 2026-06-16: the benchmark-only exact candidate
`solve_block_tridiagonal_2x2_pcr_soa_batched_transposed(...)` /
`pcr_soa_transposed` was added to test this layout without changing public
solver options or `auto`. It accepts the same `[B, Nx]` RHS shape as
`pcr_soa` and transposes internally to `[Nx, B]`.

Local smoke on 2026-06-16 passed for `B=2`, `Nx=45/89`, `float32`, with max
absolute error about `7.8e-08` versus Thomas64. It was faster than batch-first
`pcr_soa` in this tiny CPU/local smoke, but the go/no-go decision needs P100
solver-only evidence.

Kaggle P100 output recovered under `benchmark/results/kaggle/linear` from the
`20260616_223754_linear_NvidiaTeslaP100` run measured `B=128..4096`,
`Nx=32/51/64/96`, `float32`, five repeats. `pcr_soa_transposed` stayed
numerically aligned with Thomas64 (`~1.4e-07` max absolute error), but was not a
general speed win over batch-first `pcr_soa`: `8/20` cases faster and geomean
`1.047x` slower. Decision: keep benchmark-only/standby and do not route it
through `auto`.

Focused JAX trace on 2026-06-17 (`linear_pcr_soa_trace`, P100) showed the
remaining `pcr_soa` device time concentrated in `loop_select_subtract_fusion_*`
kernels:

```text
B=2048, Nx=51: pcr_soa 0.731 ms vs pcr 1.009 ms, 7 vs 31 device fusion events
B=2048, Nx=96: pcr_soa 1.211 ms vs pcr 1.564 ms, 8 vs 40 device fusion events
B=4096, Nx=51: pcr_soa 1.315 ms vs pcr 1.437 ms, 7 vs 31 device fusion events
B=4096, Nx=96: pcr_soa 1.972 ms vs pcr 2.591 ms, 13 vs 48 device fusion events
```

This keeps Phase 1C open as an implementation-optimization phase: reduce
per-stage mask/select work and neighbor gathers in the existing batch-first SoA
solver before revisiting solver-policy thresholds.

Status on 2026-06-17: added two benchmark-only candidates:

```text
solve_block_tridiagonal_2x2_pcr_soa_batched_nomask(...) / pcr_soa_nomask
solve_block_tridiagonal_2x2_pcr_soa_batched_shift(...)  / pcr_soa_shift
```

`pcr_soa_nomask` removes explicit boundary `where` masks from each PCR stage
and relies on the invariant that invalid lower/upper couplings are already zero
at the start of each stride. `pcr_soa_shift` also replaces clamped neighbor
gathers with static slice/concat shifts and identity/zero fills. Local targeted
tests passed against masked `pcr_soa` and vmapped Thomas; local solver-only
smoke kept the same Thomas64 error/residual envelope. A local HLO smoke at
`B=8`, `Nx=13` reduced `pcr_soa_shift` gather/select counts from `104/105` to
`0/0`, replacing them with static slices/concats. Next evidence gate is the P100
`linear_pcr_soa_nomask_focus` preset:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_nomask_focus \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

P100 result on 2026-06-17:

```text
run: benchmark/results/kaggle/20260617_220929_linear_pcr_soa_nomask_focus_NvidiaTeslaP100
B=2048, Nx=51: pcr_soa_nomask 0.995x runtime vs pcr_soa; pcr_soa_shift 2.051x
B=2048, Nx=96: pcr_soa_nomask 1.001x runtime vs pcr_soa; pcr_soa_shift 1.662x
B=4096, Nx=51: pcr_soa_nomask 0.983x runtime vs pcr_soa; pcr_soa_shift 1.946x
B=4096, Nx=96: pcr_soa_nomask 1.025x runtime vs pcr_soa; pcr_soa_shift 1.535x
```

`pcr_soa_nomask` was effectively neutral (`2/4` wins, geomean `1.001x`
runtime vs `pcr_soa`). `pcr_soa_shift` was slower in every focused case
(`1.786x` geomean runtime vs `pcr_soa`), even though local HLO removed gathers
and selects. The static slice/concat replacement is therefore not a useful P100
GPU optimization. Decision: do not route either candidate through `auto`; keep
`pcr_soa_nomask` as a benchmark-only neutral probe and close/standby
`pcr_soa_shift`.

Status on 2026-06-18: after reviewing the official JAX Advanced Guides, added
two benchmark-only candidates for JAX-level layout/memory control:

```text
pcr_soa_layout_auto
pcr_soa_ref
```

`pcr_soa_layout_auto` keeps the exact same batch-native SoA PCR algebra as
`pcr_soa` and changes only the compilation contract of the benchmark wrapper:
`jax.jit(..., in_shardings=Format(Layout.AUTO),
out_shardings=Format(Layout.AUTO))`. The goal is to let XLA choose device-local
layouts for the focused GPU solve without changing solver numerics or public
routing. Benchmark rows now also record compact input/output `major_to_minor`
layout summaries from the compiled executable, so the P100 run can tell us
whether the compiler requested anything different from the default batch-first
layout.

`pcr_soa_ref` keeps the same exact SoA PCR stage algebra but stores the main
PCR work arrays in internal `jax.new_ref` buffers and mutates them stage by
stage. The goal is to test whether XLA can shorten live ranges or reuse buffers
more effectively on GPU without changing the solver policy.

Evidence gate:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_layout_focus \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

Decision rule: keep both candidates benchmark-only unless one gives a clear
P100 steady-state speedup versus baseline `pcr_soa` without increasing compile
fragility or numerical error. If they only change compile time or are neutral,
close them as JAX-layout/memory diagnostics and move on to deeper solver-body
changes.

P100 result on 2026-06-18:

```text
run: benchmark/results/kaggle/20260618_202917_linear_pcr_soa_layout_focus_NvidiaTeslaP100
B=1024, Nx=51: layout_auto 1.050x runtime vs pcr_soa; ref 1.006x
B=1024, Nx=96: layout_auto 1.012x runtime vs pcr_soa; ref 1.043x
B=2048, Nx=51: layout_auto 1.099x runtime vs pcr_soa; ref 1.093x
B=2048, Nx=96: layout_auto 0.963x runtime vs pcr_soa; ref 1.008x
B=4096, Nx=51: layout_auto 0.989x runtime vs pcr_soa; ref 1.034x
B=4096, Nx=96: layout_auto 1.020x runtime vs pcr_soa; ref 1.018x
```

Summary:

```text
pcr_soa_layout_auto: 2/6 wins, 1.021x geomean runtime vs pcr_soa
pcr_soa_ref:         0/6 wins, 1.033x geomean runtime vs pcr_soa
compiled layouts:    identical to baseline, inputs [0, 1], output [0, 1, 2]
```

Decision: no routing change. Keep both candidates benchmark-only for
reproducibility, but do not spend more Kaggle runs on JAX layout/ref controls
unless a future JAX/XLA release changes layout inference. The next useful work
should change the PCR_SOA stage body itself rather than only its compilation
contract or buffer representation.

---

## Phase 1C.1 — Batch-native Thomas baseline

Status on 2026-06-16: implemented and tested on P100 as
`solve_block_tridiagonal_2x2_scalar_batched(...)` and exposed only to the
solver-only benchmark as `thomas_batched`. This is the exact same block-Thomas
algebra as `thomas`, but scans once over `Nx` with batch lanes instead of using
an outer `vmap` over fibers. It is not a public `BatchOptions` solver choice
and is not routed by `auto`.

Local tests cover shared coefficients, batched coefficients, and `Nx=1` against
the vmapped Thomas reference. Next evidence needed: Kaggle P100 solver-only
`linear` comparison of `thomas_batched` versus `thomas`.

Local smoke on 2026-06-16 passed for `B=2`, `Nx=45/89`, `float32`, with max
absolute error about `4.6e-08` versus Thomas64. It was slightly faster than the
current vmapped Thomas path on this CPU/local smoke, but that is not GPU
performance evidence.

Kaggle P100 `20260616_222231_linear_NvidiaTeslaP100` measured `B=128..4096`,
`Nx=32/51/64/96`, `float32`, five repeats. `thomas_batched` stayed numerically
aligned with Thomas64 (`~1.4e-07` max absolute error), but was not a steady-state
GPU win over the current vmapped `thomas`: `8/20` cases faster and geomean
`1.009x` slower. Compile time improved (`0.885x` geomean), but runtime does not
justify routing this path into `auto`. Decision: keep benchmark-only/standby.

---

## Phase 1D — Padding Nx to 32/64/128

Status on 2026-06-16: helper and benchmark-only candidate implemented and
tested on P100.
`double_cable_power_bucket(...)`,
`pad_double_cable_system_to_power_bucket(...)`, and
`solve_block_tridiagonal_2x2_pcr_soa_batched_padded(...)` keep padded rows as
identity equations and slice the real rows after the solve. Local tests cover
shared and batched coefficients for `Nx=45 -> 64` and `Nx=89 -> 128`.
`pcr_soa_padded` is available only in the solver-only benchmark and Kaggle
`linear` matrix; it is not a public `BatchOptions.double_cable_block_solver`
choice and is not routed by `auto`.

Local smoke on 2026-06-16 passed for `B=2`, `Nx=45/89`, `float32`, comparing
`pcr_soa_padded` to the Thomas64 reference with max absolute error about
`7.8e-08`. This CPU/local smoke is not GPU performance evidence; the padded
path was slower than unpadded at this tiny batch size. The next useful result is
the Kaggle P100 solver-only `linear` matrix.

Kaggle P100 `20260616_220653_linear_NvidiaTeslaP100` measured `B=128..4096`,
`Nx=32/51/64/96`, `float32`, five repeats. `pcr_soa_padded` stayed numerically
aligned with Thomas64 (`~1.4e-07` max absolute error), but was not a general
speed win over unpadded batch-native `pcr_soa`: `6/20` cases faster and
geomean `1.086x` slower. Decision: keep padding as benchmark-only/standby and
do not route it through `auto`.

## Why

The target `Nx` is small and bounded. Padding enables static stage counts and fewer recompilations.

```text
Nx <= 32   -> Nx_pad = 32
Nx <= 64   -> Nx_pad = 64
Nx <= 128  -> Nx_pad = 128
```

For example:

```text
Nx=51 -> Nx_pad=64
Nx=96 -> Nx_pad=128
```

## Padding rule

For padded rows:

```text
D_i = I
L_i = 0
U_i = 0
rhs_i = 0
```

Then slice the final result:

```python
x0 = x0[:, :Nx]
x1 = x1[:, :Nx]
```

## Add helper

```python
def pad_double_cable_system_to_power_bucket(..., bucket: int):
    ...
```

## Tests

Assert padded and unpadded solutions match for real rows.

---

## Phase 1E — Hybrid PCR + Thomas

Status on 2026-06-16: benchmark-only exact candidates implemented and tested on
P100 as
`solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(...)` with
`pcr_soa_hybrid_4`, `pcr_soa_hybrid_8`, and `pcr_soa_hybrid_16`. The
implementation runs PCR stages until remaining couplings jump by the requested
stride, then solves the independent residual chains defined by `i % stride`
with exact batch-native 2x2 block Thomas. These variants are not public
`BatchOptions` choices and are not routed by `auto`.

Local smoke on 2026-06-16 passed for `B=2`, `Nx=45/89`, `float32`, with max
absolute error about `7.8e-08` versus Thomas64 for all three hybrid variants.
CPU/local timing was mixed (`hybrid_4` faster than `pcr_soa` at `Nx=89` but
slower at `Nx=45`); the go/no-go decision needs the Kaggle P100 solver-only
`linear` matrix.

Kaggle P100 `20260616_225915_linear_NvidiaTeslaP100` measured `B=128..4096`,
`Nx=32/51/64/96`, `float32`, five repeats. All hybrid variants stayed
numerically aligned with Thomas64 (`~1.4e-07` max absolute error), but failed
the go/no-go by a wide margin: `hybrid_4` won `0/20` cases and was `3.405x`
slower than batch-native `pcr_soa` geomean; `hybrid_8` was `3.828x` slower;
`hybrid_16` was `4.274x` slower. Decision: keep all hybrid variants
benchmark-only/standby and do not route them through `auto`.

## Motivation

PCR exposes parallelism but does more arithmetic than Thomas. For `Nx=30-100`, full PCR may not always win. Hybrid PCR reduces global dependency depth, then solves small independent blocks with Thomas.

## Candidate variants

```text
PCR_HYBRID_BLOCK_4
PCR_HYBRID_BLOCK_8
PCR_HYBRID_BLOCK_16
```

Example for `Nx_pad = 64`:

```text
PCR stages until stride = 8
then solve independent residual blocks of size 8 or 16
```

## Go/no-go

Keep hybrid if:

```text
solver-only speedup > 1.2x vs PCR_SOA
and numerical error is unchanged
```

Drop or postpone if:

```text
implementation complexity increases sharply
or XLA lowering is worse than full PCR
```

---

## Phase 1 deliverables

By the end of Phase 1:

```text
1. Solver option contract is documented and either remains Literal-based or is
   promoted to an enum deliberately.
2. THOMAS, PCR, PCR_SOA, and PCR_ADAPTIVE are official exact backends.
3. Solver-only benchmark compares THOMAS, PCR, PCR_SOA, and PCR_ADAPTIVE.
4. End-to-end benchmark confirms whether PCR-family solvers help real
   double-cable workloads.
5. Batch-native PCR_SOA and padding buckets are implemented only if the
   solver-only baseline still justifies them.
6. AUTO policy is updated only from benchmark evidence.
```

Working rule: prioritize implementing roadmap solver backends and reusable
solver primitives before spending time on fine-grained heuristic thresholds.
Small routing thresholds such as batch-size crossovers should be recorded as
benchmark-backed calibration after a substantive backend exists, not treated as
the main optimization work.

## Phase 1 expected outcome

Conservative target:

```text
1.2x-2x solver-only speedup on B>=1024, Nx=64
```

Optimistic target:

```text
2x-4x solver-only speedup if Thomas scan was the dominant GPU bottleneck
```

End-to-end may be lower if output materialization or Vext construction dominates.

---


# Phase 1.5 — Two-rail split iterative solver

## Goal

Test the idea of treating one double-cable system as **two coupled scalar cables** rather than one 2x2 block-tridiagonal cable.

This phase is motivated by the observation that the double-cable matrix can be reordered as:

```text
Ti Vi + Cie Ve = bi
Cei Vi + Te Ve = be
```

where:

```text
Ti = scalar tridiagonal axonal/intracellular cable operator
Te = scalar tridiagonal periaxonal/extracellular cable operator
Cie, Cei = local diagonal coupling between the two rails
```

If `Vi` and `Ve` were independent, each rail would be a normal single-cable solve. They are not independent, but they can be solved iteratively using the scalar tridiagonal solver that already scales well on GPU.

This does not replace the exact direct block solver. It adds an iterative backend that is exact only if iterated to convergence. Fixed small iteration counts are approximate but may be accurate enough for threshold/recruitment if the residual is small.

Status on 2026-06-16: Phase 1.5 is implemented as solver-only,
benchmark-only candidates:

```text
split_jacobi_4
split_jacobi_8
split_jacobi4_gs1
split_gs_2
split_gs_3
split_gs_4
split_gs_8
split_richardson_4
```

Implemented primitives:

```text
split_double_cable_block_system_soa(...)
solve_tridiagonal_batched(...)
solve_double_cable_split_jacobi_batched(...)
solve_double_cable_split_gauss_seidel_batched(...)
solve_double_cable_split_richardson_batched(...)
double_cable_block_residual_norm(...)
```

These names are not public `BatchOptions.double_cable_block_solver` choices and
are not used by `auto`. The Kaggle `linear` preset now runs only the official
direct solvers plus these split candidates, leaving prior non-winning
`thomas_batched`, padded, transposed, and hybrid candidates in standby.

Local CPU smoke on 2026-06-16 (`B=2`, `Nx=45/89`, `float32`) compiled and ran
all Phase 1.5 candidates. `split_jacobi_8`, `split_gs_4`, and `split_gs_8`
matched the exact baseline with max error/residual around `1e-7`.
`split_jacobi_4` was close but less accurate (`~6e-6` residual).
`split_richardson_4` was not accurate enough in this smoke (`~1e-3` residual).

Kaggle P100 `20260616_233228_linear_NvidiaTeslaP100` measured the focused
solver-only `linear` matrix (`B=128..4096`, `Nx=32/51/64/96`, `float32`):

```text
split_gs_4:
    max_residual ~2.1e-7
    max_abs_error ~1.6e-7
    13/20 wins vs pcr_soa
    0.893x geomean runtime vs pcr_soa overall
    0.687x geomean runtime vs pcr_soa for B>=2048, winning 8/8

split_jacobi_4:
    16/20 wins vs pcr_soa
    0.818x geomean runtime vs pcr_soa overall
    but max_residual ~7.1e-6 and max_abs_error ~2.5e-6

split_jacobi_8:
    exact-like residual/error, but 1.149x geomean runtime vs pcr_soa

split_gs_8:
    exact-like residual/error, but 1.319x geomean runtime vs pcr_soa

split_richardson_4:
    not accurate enough; max_residual ~1.1e-3
```

Interim decision: keep `split_gs_4` as the first clean Phase 1.5 candidate,
but test lower-K cleanup variants before spending an E2E/physiology run. Keep
`split_jacobi_4` benchmark-only as a possible approximate physiology
experiment. Put `split_jacobi_8`, `split_gs_8`, and `split_richardson_4` in
standby.

Follow-up local smoke on 2026-06-16 added:

```text
split_gs_2
split_gs_3
split_jacobi4_gs1
```

Local CPU smoke (`B=2`, `Nx=45/89`, `float32`) showed `split_gs_3` and
`split_jacobi4_gs1` at near-exact residual/error levels (`~4e-7`) while running
faster than local `split_gs_4`. `split_gs_2` was too approximate locally
(`~5e-5` residual).

Kaggle P100 `20260616_235328_linear_split_focus_NvidiaTeslaP100` measured the
focused follow-up (`B=1024/2048/4096`, `Nx=32/51/64/96`, `float32`):

```text
split_gs_3:
    max_residual ~6.5e-7
    max_abs_error ~3.1e-7
    11/12 wins vs pcr_soa
    0.648x geomean runtime vs pcr_soa
    12/12 wins vs split_gs_4
    0.818x geomean runtime vs split_gs_4

split_gs_4:
    stricter residual/error, max_residual ~2.1e-7
    10/12 wins vs pcr_soa
    0.792x geomean runtime vs pcr_soa

split_jacobi4_gs1:
    same residual/error level as split_gs_3
    slower than split_gs_3 and split_gs_4 on this panel

split_gs_2:
    fastest but not accurate enough; max_residual ~6.2e-5
```

Interim decision at this point: carry `split_gs_3` forward to E2E/physiology
validation as the main Phase 1.5 candidate, with `split_gs_4` as the stricter
residual fallback. Keep `split_jacobi4_gs1`, `split_gs_2`, `split_jacobi_4`,
`split_jacobi_8`, `split_gs_8`, and `split_richardson_4`
benchmark-only/standby. This interim decision was superseded by the E2E
agreement validation below.

E2E focus status on 2026-06-17: `split_gs_3` and `split_gs_4` are wired into
the end-to-end benchmark through an internal benchmark-only kernel override.
`BatchOptions.double_cable_block_solver` and `auto` remain unchanged. The E2E
split path is intentionally limited to array-output recordings (`center` or
`full`) so it exercises the batch-native array kernel, not the observer-only
per-fiber path. Local smoke passed for `B=2`, actual `Nx=45`, `Nt=3`,
`recording=center`, `Iinj=none`.

Kaggle P100 `20260617_105250_e2e_split_focus_NvidiaTeslaP100` measured
`pcr_adaptive`, `split_gs_3`, and `split_gs_4` for `B=1024/2048/4096`,
`target_Nx=51/96`, `Nt=500`, `recording=center`, and `Iinj=none`:

```text
median kernel time, split_gs_3 vs pcr_adaptive:
    all cases: 5/6 wins, 0.721x geomean runtime = 1.39x speedup
    B>=2048: 4/4 wins, 0.604x geomean runtime = 1.66x speedup
    B>=2048 and actual_Nx=89: 2/2 wins, 0.514x runtime = 1.94x speedup

median kernel time, split_gs_3 vs split_gs_4:
    6/6 wins, 0.792x geomean runtime = 1.26x speedup

total_with_inputs, split_gs_3 vs pcr_adaptive:
    all cases: 1.46x geomean speedup
    B>=2048: 1.05x geomean speedup
    B>=2048 and actual_Nx=89: 1.09x geomean speedup
```

Validation update on 2026-06-17: added
`benchmark/solvers/validate_double_cable_solver_agreement.py` to compare
recorded `Vm` traces, peak-voltage errors, activation agreement, and
first-threshold-crossing timing against exact public solvers. A local held-out
smoke (`B=2`, target `Nx=51`, actual `Nx=45`, `Nt=3`, `dt=0.05 ms`,
`recording=center`, `Iinj=none`) found that `split_gs_3` and `split_gs_4`
diverge from `pcr_adaptive` by about `77 mV` at the center trace and introduce
false activations at the public `-20 mV` activation threshold. The exact
controls in the same harness (`pcr_soa`/`pcr_adaptive` versus `thomas`) stayed
close at about `0.0014 mV` max absolute error on this tiny local smoke.

Decision: abandon split iterative approaches (`split_jacobi_*`,
`split_gs_*`, `split_richardson_*`, and mixed Jacobi/GS cleanup) for the
current double-cable GPU optimization pass. The physical MRG double-cable
systems are too strongly coupled for a few split iterations, and the E2E
total-with-input gains were not strong enough to justify pursuing more split
variants. Keep existing split code benchmark-only for historical
reproducibility until a later cleanup removes failed candidates. Do not route
any split solver into `BatchOptions` or `auto`.

---

## Why this is attractive on GPU

For each double-cable solve, instead of one block-tridiagonal 2x2 solve, perform repeated scalar tridiagonal solves:

```text
K split iterations -> 2*K scalar tridiagonal solves
```

For a batch of fibers:

```text
B double-cable systems -> 2*B scalar cable systems per iteration
```

With `B=1024` and `K=4`, the GPU sees approximately:

```text
8192 scalar tridiagonal solves per time step
```

This may exploit the existing single-cable GPU path better than a custom block solver, especially for `Nx=30-100`.

---

## Backends to add

Add these solver modes:

```python
DoubleCableLinearSolver.SPLIT_JACOBI
DoubleCableLinearSolver.SPLIT_GAUSS_SEIDEL
DoubleCableLinearSolver.SPLIT_RICHARDSON
```

Add config:

```python
@dataclass(frozen=True)
class SplitIterativeSolverConfig:
    iterations: int = 4
    method: Literal["jacobi", "gauss_seidel", "richardson"] = "gauss_seidel"
    relaxation: float = 1.0
    init: Literal["previous_timestep", "zero", "rhs_guess", "thomas_warmup"] = "previous_timestep"
    residual_check: bool = True
    residual_every_n_steps: int = 50
    residual_tolerance: float = 1e-5
    fallback_solver: DoubleCableLinearSolver = DoubleCableLinearSolver.THOMAS
```

The `iterations` value must be static for JIT. Do not implement dynamic convergence loops in the main GPU path initially.

---

## Phase 1.5A — Matrix reordering

Add a helper that extracts the two scalar cable operators from the existing block-tridiagonal coefficients.

From the block system:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = b_i
x_i = [Vi_i, Ve_i]^T
```

where:

```text
D_i = [[d00_i, d01_i],
       [d10_i, d11_i]]
```

construct:

```text
Ti: lower_vi, diag_vi=d00, upper_vi
Te: lower_ve, diag_ve=d11, upper_ve
Cie = d01
Cei = d10
bi = rhs0
be = rhs1
```

This reordering is only valid if off-block coupling between rails remains diagonal/local, which is true for the current specialized double-cable structure. Fail loudly if a future model introduces off-diagonal rail coupling in `L_i` or `U_i`.

Add:

```python
def split_double_cable_block_system_soa(...):
    return Ti, Te, Cie, Cei, bi, be
```

---

## Phase 1.5B — Split Jacobi

Jacobi iteration updates both rails from the previous iterate:

```text
Vi^{k+1} = solve(Ti, bi - Cie * Ve^k)
Ve^{k+1} = solve(Te, be - Cei * Vi^k)
```

Advantages:

```text
1. The two scalar solves are independent inside each iteration.
2. They can be concatenated into one 2B batch.
3. Maximum GPU parallelism.
```

Disadvantages:

```text
1. Convergence may be slower than Gauss-Seidel.
2. More sensitive to strong coupling.
```

Implementation shape:

```python
def split_jacobi_fixed_k(Ti, Te, Cie, Cei, bi, be, Vi0, Ve0, K):
    Vi = Vi0
    Ve = Ve0
    for _ in range(K):
        rhs_i = bi - Cie * Ve
        rhs_e = be - Cei * Vi

        # Preferred: concatenate the two scalar cable batches.
        Vi_new, Ve_new = solve_two_scalar_batches_together(Ti, Te, rhs_i, rhs_e)

        Vi = Vi_new
        Ve = Ve_new
    return Vi, Ve
```

---

## Phase 1.5C — Split Gauss-Seidel

Gauss-Seidel uses the updated `Vi` immediately when solving `Ve`:

```text
Vi^{k+1} = solve(Ti, bi - Cie * Ve^k)
Ve^{k+1} = solve(Te, be - Cei * Vi^{k+1})
```

Advantages:

```text
1. Usually converges faster than Jacobi.
2. May need fewer iterations.
```

Disadvantages:

```text
1. The two rail solves are sequential inside each iteration.
2. Less parallelism than Jacobi, but still uses scalar tridiagonal solves.
```

Implementation shape:

```python
def split_gauss_seidel_fixed_k(Ti, Te, Cie, Cei, bi, be, Vi0, Ve0, K):
    Vi = Vi0
    Ve = Ve0
    for _ in range(K):
        Vi = tridiagonal_solve(Ti, bi - Cie * Ve)
        Ve = tridiagonal_solve(Te, be - Cei * Vi)
    return Vi, Ve
```

---

## Phase 1.5D — Preconditioned Richardson

This treats the full block system as:

```text
A x = b
```

with block-diagonal preconditioner:

```text
M = diag(Ti, Te)
```

Each iteration:

```text
r^k = b - A x^k
z^k = M^{-1} r^k
x^{k+1} = x^k + omega * z^k
```

Applying `M^{-1}` costs two scalar tridiagonal solves.

Advantages:

```text
1. More general and easier to monitor with residuals.
2. Can use relaxation `omega` for stability.
3. Natural stepping stone toward Krylov methods.
```

Disadvantages:

```text
1. May require tuning `omega`.
2. Fixed-K Richardson may converge slowly.
```

Start with:

```text
omega = 0.5, 0.75, 1.0
K = 1, 2, 4, 8
```

---

## Initialization strategy

The best initial guess is usually the previous time step:

```text
Vi0 = Vi_previous_timestep
Ve0 = Ve_previous_timestep
```

Reason: implicit time steps are close in time, so the new solution should often be close to the previous state.

Fallback initializations to test:

```text
zero
rhs_guess: Vi0 = bi / diag(Ti), Ve0 = be / diag(Te)
thomas_warmup: use exact solver every N steps as a refresh
```

Do not use `zero` as the main benchmark unless testing worst-case convergence.

---

## Residual checking

Even with fixed `K`, compute the residual periodically in debug/validation mode:

```text
residual = ||A x - b|| / (||b|| + eps)
```

Add function:

```python
def double_cable_block_residual_norm(..., Vi, Ve) -> Array:
    ...
```

For production, residual checks can be disabled or run every `N` time steps.

Suggested thresholds:

```text
float64 solver-only residual < 1e-9 for convergence-mode tests
float32 fixed-K residual < 1e-5 to 1e-4 for candidate production tests
```

But physiological outputs matter most:

```text
threshold error
activation agreement
first spike time error
conduction velocity error
```

---

## Exactness modes

Support two conceptual modes:

```text
fixed_k:
    K is static.
    Very JIT/GPU-friendly.
    Approximate unless K is high enough.

converged_debug:
    Run until residual < tolerance outside the main compiled path, or with a bounded max K.
    Used only to determine how many iterations are needed.
```

Do not put an unbounded dynamic convergence loop inside the main JIT path.

---

## Benchmark matrix for split solvers

Solver-only:

```text
B:       512, 1024, 2048, 4096
Nx:      32, 51, 64, 96, 100
K:       1, 2, 4, 8, 12
method:  jacobi, gauss_seidel, richardson
init:    previous_timestep, rhs_guess, zero
```

Compare against:

```text
THOMAS
PCR_SOA
PCR_ADAPTIVE
```

Metrics:

```text
solver time
speedup vs THOMAS
speedup vs PCR_SOA
residual norm
max_abs(Vi - Vi_exact)
max_abs(Ve - Ve_exact)
max_abs(Vm - Vm_exact)
```

End-to-end:

```text
activation agreement
threshold error
first spike time error
conduction velocity error
recruitment curve error
```

---

## Go/no-go criteria

Keep split iterative as a serious backend if:

```text
K <= 4 gives threshold error < 1% and activation agreement is exact or near-exact
and solver-only speedup > PCR_SOA or THOMAS
```

Keep only as optional approximate backend if:

```text
K = 8 is needed but still faster end-to-end for B>=1024
and errors are physiologically negligible
```

Drop or postpone if:

```text
K > 8 is needed for stability
or residuals stagnate
or threshold/recruitment drift is unacceptable
or two scalar solves * K is slower than PCR_SOA
```

---

## Where Phase 1.5 fits in the roadmap

This phase should be implemented after Phase 1A/1B because it needs clean solver dispatch and benchmarks, but it can run before Phase 2 associative scan and Phase 3 Pallas.

Recommended order:

```text
1. Add solver dispatch. [done]
2. Make PCR_SOA official. [done]
3. Add split_double_cable_block_system_soa. [done]
4. Implement split_jacobi_fixed_k. [done as benchmark-only]
5. Implement split_gauss_seidel_fixed_k. [done as benchmark-only]
6. Add residual checker. [done]
7. Benchmark vs Thomas/PCR. [done for solver-only P100]
8. Only then decide whether associative scan or Pallas is still necessary.
```

This phase may become the highest-value practical path if the existing scalar tridiagonal GPU solver is already highly optimized.

---


# Phase 2 — Associative scan exact solvers

## Goal

Explore exact parallel prefix formulations using `jax.lax.associative_scan`.

JAX `associative_scan` performs a parallel scan when the binary operation is associative. This is useful if we can express part or all of the block-tridiagonal solve as composition of associative transforms.

There are two variants:

```text
Phase 2A: Thomas forward + associative backward
Phase 2B: full transfer-matrix associative scan
```

---

## Phase 2A — Associative backward substitution

## Idea

After Thomas forward elimination, the backward pass has the form:

```text
x_i = d_i - C_i x_{i+1}
```

This is an affine transform:

```text
f_i(x) = A_i x + q_i

A_i = -C_i
q_i = d_i
```

Composition of affine transforms is associative:

```text
f(g(x)) = A_f A_g x + A_f q_g + q_f
```

Therefore the backward substitution can be parallelized with an associative scan in reverse.

## Add function

```python
def solve_block_tridiagonal_2x2_assoc_backward_soa(
    a00, a01, a10, a11,
    off0, off1,
    rhs0, rhs1,
):
    # 1. Same Thomas forward elimination as current solver.
    # 2. Replace reverse lax.scan with lax.associative_scan over affine transforms.
    ...
```

## Expected benefit

This does **not** remove the forward dependency. It only parallelizes the backward pass.

Expected speedup:

```text
modest: 5-25% solver-only
```

But the implementation risk is low.

## Go/no-go

Keep if:

```text
speedup >= 10% on GPU for B>=1024, Nx>=64
no numerical regression
code remains simple
```

Drop if:

```text
XLA lowering is worse than reverse scan
or performance gain is negligible
```

## Current status — 2026-06-17

Implemented as benchmark-only `assoc_backward` via
`solve_block_tridiagonal_2x2_assoc_backward_batched(...)`.

Local smoke:

```text
B=2, Nx=45/89, float32
solvers: thomas, thomas_batched, assoc_backward, pcr_soa
max_abs_error_vs_thomas64 for assoc_backward: ~5.0e-08
max_block_residual_norm for assoc_backward: ~9.3e-08
```

P100 result:

```text
run: 20260617_112515_linear_assoc_focus_NvidiaTeslaP100
max_abs_error_vs_thomas64: ~1.0e-07
max_block_residual_norm: ~1.3e-07
vs thomas: 9/9 wins, 0.706x geomean runtime, 1.42x speedup
vs thomas_batched: 9/9 wins, 0.696x geomean runtime, 1.44x speedup
vs pcr_soa: 3/9 wins, 1.313x geomean runtime, 0.76x speedup
vs pcr_soa at B=4096: 3/3 wins, 0.792x geomean runtime, 1.26x speedup
```

JAX 0.10.2 P100 retest:

```text
run: benchmark/results/kaggle/20260618_182820_linear_assoc_focus_NvidiaTeslaP100
jax: 0.10.2
jaxlib: 0.10.2
max_abs_error_vs_thomas64 for assoc_backward: ~1.0e-07
max_block_residual_norm for assoc_backward: ~1.3e-07
vs thomas_batched: 9/9 wins, 0.722x geomean runtime, 1.385x speedup
vs pcr_soa: 1/9 wins, 1.570x geomean runtime, 0.637x speedup
only pcr_soa win: B=4096, Nx=96, 0.995x runtime, ~1.005x speedup
```

Decision: `assoc_backward` is a successful Thomas-family optimization, but it
is not a better general exact backend than `pcr_soa`/`pcr_adaptive`. The JAX
0.10.2 retest removed the earlier broad `B=4096` advantage and left only one
tiny win at `B=4096`, `Nx=96`. Keep it benchmark-only/standby. Revisit only if
future workloads specifically require a Thomas-family fallback or if a new
trace shows PCR_SOA is blocked by a regime where associative backward is
consistently faster.

---

## Phase 2B — Full transfer-matrix associative scan

## Idea

Rewrite each local equation as a state transition.

Given:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = b_i
```

If `U_i` is invertible:

```text
x_{i+1} = -U_i^{-1} D_i x_i - U_i^{-1} L_i x_{i-1} + U_i^{-1} b_i
```

Define state:

```text
y_i = [x_i, x_{i-1}, 1]
```

Then:

```text
y_{i+1} = M_i y_i
```

where `M_i` is a 5x5 affine matrix because each `x_i` is length 2.

Products of matrices are associative:

```text
(M3 M2) M1 = M3 (M2 M1)
```

So we can compute prefix products with `lax.associative_scan`.

## Prototype implementation

Start with dense 5x5 matrices for clarity:

```python
def solve_block_tridiagonal_2x2_transfer_scan_dense(
    a00, a01, a10, a11,
    off0, off1,
    rhs0, rhs1,
):
    ...
```

Use shapes:

```text
M: [B, Nx-1, 5, 5]
```

For `B=1024, Nx=64`, this is:

```text
1024 * 63 * 25 * 4 bytes ≈ 6.5 MB
```

which is acceptable for a prototype.

Then optimize to SoA only after correctness and stability are established.

## Boundary handling

Use prefix products to express every pair `(x_i, x_{i-1})` in terms of initial unknown `x_0`.

Then use the final boundary equation to solve for `x_0`.

Conceptually:

```text
x_{N-2} = P_{N-2} x_0 + p_{N-2}
x_{N-1} = P_{N-1} x_0 + p_{N-1}
```

Final equation:

```text
L_{N-1} x_{N-2} + D_{N-1} x_{N-1} = b_{N-1}
```

so:

```text
(L_{N-1} P_{N-2} + D_{N-1} P_{N-1}) x_0
=
b_{N-1} - L_{N-1} p_{N-2} - D_{N-1} p_{N-1}
```

Then reconstruct all `x_i` in parallel from prefix products.

## Risks

This is exact algebraically, but may be numerically less stable than Thomas/PCR because it resembles a transfer-matrix or shooting method.

Risks:

```text
1. U_i may be singular or poorly conditioned.
2. Matrix products may amplify errors.
3. float32 may be unstable.
4. Heterogeneous NODE/MYSA/FLUT/STIN coefficients may worsen conditioning.
5. Dense 5x5 implementation may be too slow unless optimized.
```

## Required safeguards

Add diagnostics:

```text
det(U_i)
condition proxy for U_i
condition proxy for final 2x2 boundary solve
NaN/Inf checks in debug mode
float32 vs float64 comparison
```

If `U_i` is not safe, derive a symmetric alternative based on invertible `L_i`, or use a Riccati/Green's function formulation.

## Go/no-go

Continue to SoA optimization only if dense prototype gives:

```text
max_abs_error float64 < 1e-8 vs Thomas
float32 physiologically stable
solver-only speed comparable to PCR_SOA
```

Stop if:

```text
float32 diverges
condition issues appear in physical systems
dense version is already much slower than PCR and unlikely to optimize enough
```

## Current status — 2026-06-17

Implemented as benchmark-only diagnostic `assoc_transfer_dense` via
`solve_block_tridiagonal_2x2_assoc_transfer_dense_batched(...)`.

Result:

```text
well-conditioned artificial systems: matches Thomas in local tests
benchmark-like float32 systems: numerically unstable due transfer amplification
```

Do not include this candidate in Kaggle focus runs and do not optimize it to
SoA unless a stabilized transfer/Riccati formulation is derived. It remains a
diagnostic prototype, not a routing candidate.

---

## Phase 2 deliverables

```text
1. ASSOCIATIVE_BACKWARD backend.
2. ASSOCIATIVE_TRANSFER dense prototype.
3. Correctness tests vs Thomas and PCR.
4. Stability report.
5. Performance report on B>=1024, Nx=32/64/128.
6. Decision: keep associative transfer, optimize it, or shelve it.
```

## Phase 2 expected outcome

`ASSOCIATIVE_BACKWARD`:

```text
low risk, modest speedup
```

`ASSOCIATIVE_TRANSFER`:

```text
high risk, potentially strong speedup
```

The likely winner may still be PCR_SOA, but associative scan is worth testing because it may reduce work relative to full PCR if XLA lowers it well.

---

# Phase 3 — Pallas custom kernels

## Goal

If JAX-native PCR or associative scan leaves performance on the table, implement small-Nx, large-B custom kernels.

Pallas should be used only after Phase 1 and Phase 2 identify the best mathematical algorithm. Pallas should not be the first step.

## Why Pallas may help

Pallas can give direct control over:

```text
1. GPU tiling over B and Nx.
2. Memory layout.
3. Register/scratch usage.
4. Avoiding tiny 2x2 matrix abstractions.
5. Reducing temporary arrays.
6. Reducing gather/scatter overhead in PCR.
7. Reducing kernel count if XLA does not fuse well.
```

Pallas does not magically parallelize Thomas. It only lets us implement the chosen algorithm more directly.

---

## Phase 3A — Pallas Thomas small-Nx baseline

## Goal

Determine whether custom kernels help even without changing the algorithm.

Implement:

```text
pallas_thomas_2x2_small_nx
```

Mapping:

```text
grid = (ceil(B / BLOCK_B),)
BLOCK_B = 32, 64, or 128
Nx_pad = 32, 64, or 128
```

Each Pallas program handles:

```text
BLOCK_B fibers
full Nx_pad cable
one linear solve per fiber
```

Use scalarized 2x2 operations.

## Expected benefit

If JAX/XLA overhead is significant:

```text
1.2x-2x solver-only speedup
```

If the main bottleneck is pure Thomas dependency depth:

```text
small or no speedup
```

## Go/no-go

Proceed to Pallas PCR/hybrid if:

```text
pallas_thomas >= 1.3x vs jax_thomas
or trace shows lower kernel count / better occupancy
```

If Pallas Thomas is not faster, Pallas PCR may still help, but the bar is higher.

## Current status — 2026-06-17

Implemented as benchmark-only `pallas_thomas_128` via
`solve_block_tridiagonal_2x2_pallas_thomas_batched(...)`.

Local smoke:

```text
B=128, Nx=16, float32
solvers: thomas, thomas_batched, assoc_backward, pallas_thomas_128, pcr_soa
max_abs_error_vs_thomas64 for pallas_thomas_128: ~5.9e-08
max_block_residual_norm for pallas_thomas_128: ~1.2e-07
```

Local execution used Pallas `interpret=True` on CPU, so timing is not GPU
evidence. The first useful decision point is the P100 `linear_pallas_focus`
run. Keep this candidate benchmark-only and do not add any public routing.

First Kaggle attempt:

```text
run: 20260617_114922_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas measurement
cause: Kaggle JAX 0.7.2 no longer exposes jax.experimental.pallas.MemoryRef
fix: compatibility shim falls back to jax._src.pallas.core.MemoryRef
```

Second Kaggle attempt:

```text
run: 20260617_115323_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas measurement
cause: JAX 0.7.2 private MemoryRef uses signature (shape, dtype)
fix: compatibility shim accepts both two- and three-argument MemoryRef forms
```

Third Kaggle attempt:

```text
run: 20260617_115814_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas measurement
cause: JAX 0.7.2 exposes pallas.triton.CompilerParams, not TritonCompilerParams
fix: remove explicit Triton compiler params and use Pallas defaults for spike
```

Fourth Kaggle attempt:

```text
run: 20260617_120625_linear_pallas_focus_NvidiaTeslaP100
status: failed during Pallas kernel tracing
cause: direct scratch/output indexing writes hit swap.abstract_eval in JAX 0.7.2
fix: use explicit pl.store/pl.load for scratch and output refs
```

Fifth Kaggle attempt:

```text
run: 20260617_121201_linear_pallas_focus_NvidiaTeslaP100
status: failed during Pallas kernel tracing
cause: scratch refs in JAX 0.7.2 do not expose .shape
fix: compute batch indices from input block shape and reuse for pl.store/pl.load
```

Sixth Kaggle attempt:

```text
run: 20260617_121708_linear_pallas_focus_NvidiaTeslaP100
status: failed during Pallas kernel tracing
cause: explicit pl.store on scratch ref reaches Pallas swap abstract eval, then
       fails with IndexError: tuple index out of range
decision: no Pallas timing was recorded; put Phase 3A in standby
```

Decision: stop spending Kaggle runs on `pallas_thomas_128`. The candidate is
correct in local Pallas `interpret=True` mode, but the current implementation is
not compatible with Kaggle's JAX/Pallas `0.7.2` tracing path. Revisit only if
we can reproduce against the Kaggle JAX/Pallas version locally, or if the
kernel is rewritten against the current Pallas indexing API. Return to
exact JAX solver candidates and dense-input/Vext work; split iterative
approaches are now closed.

JAX 0.10.1 local retest on 2026-06-17:

```text
environment: Python 3.12.13, jax 0.10.1, jaxlib 0.10.1
backend: CPU
local Pallas mode: interpret=True
```

The Pallas compatibility shim was updated for the current API:

```text
1. jax.experimental.pallas no longer exports pl.load/pl.store.
   The spike now falls back to jax._src.pallas.primitives.load/store.
2. MemoryRef now expects a shaped abstract value. Scratch refs use
   MemorySpace.ANY(shape, dtype), with fallbacks for older signatures.
```

Local Pallas smoke now passes again:

```text
B=128, Nx=8, float32, pallas_thomas_128 vs Thomas
max_abs_error Vi: ~5.96e-08
max_abs_error Ve: ~2.98e-08
```

Small solver benchmark on CPU interpret mode:

```text
B=128, Nx=16, float32
thomas:            median ~0.221 ms
pallas_thomas_128: median ~0.483 ms
pcr_soa:           median ~2.549 ms
```

This CPU timing is not GPU evidence because non-interpreted Pallas is not
supported on the CPU backend. The next useful decision point is a fresh GPU
`linear_pallas_focus` run after committing the JAX 0.10 compatibility shim.

Kaggle P100 retry after the JAX 0.10.1 local update:

```text
run: 20260617_211635_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas measurement
environment reached: Python 3.12, Tesla P100, jax/jaxlib 0.10.1, backend gpu
cause: Kaggle kept preinstalled jax_cuda12_plugin 0.7.2 while the project
       installed jaxlib 0.10.1, causing a PJRT plugin ABI mismatch before the
       benchmark generated its first case.
fix: update the Kaggle wrapper to install jax[cuda12]==<installed jax version>
     after the project install, so the CUDA plugin/PJRT packages match jaxlib.
```

The P100 needs the CUDA 12 JAX extra rather than CUDA 13 because P100 is SM 6.0
and CUDA 13 JAX wheels require newer GPUs. Re-run `linear_pallas_focus` after
committing the Kaggle CUDA-plugin fix before making a final Phase 3A decision.

Second Kaggle P100 retry after the JAX 0.10.1 local update:

```text
run: 20260617_212151_linear_pallas_focus_NvidiaTeslaP100
status: failed during Pallas lowering
progress: the Kaggle wrapper installed jax-cuda12-plugin 0.10.1, selected the
          P100 GPU backend, and measured thomas/thomas_batched/assoc_backward
          for the first case.
cause: Pallas Mosaic GPU lowering rejects scratch MemorySpace.ANY.
fix: make the Pallas scratch MemoryRef prefer MemorySpace.DEFAULT, with ANY
     retained only as a fallback for older/interpreter paths.
```

Local `interpret=True` Pallas tests still pass after the memory-space change.
Re-run `linear_pallas_focus` once this fix is committed.

Third Kaggle P100 retry after the JAX 0.10.1 local update:

```text
run: 20260617_212605_linear_pallas_focus_NvidiaTeslaP100
status: failed during Pallas lowering
cause: MemorySpace.DEFAULT lowers to gmem for scratch allocation, and Mosaic GPU
       rejects gmem scratch just like any.
fix: allocate GPU scratch with jax.experimental.pallas.mosaic_gpu.SMEM(...);
     keep the generic Pallas MemoryRef fallback for local interpret=True and
     older API paths.
```

Local `interpret=True` Pallas tests still pass after switching the GPU scratch
path to SMEM. Re-run `linear_pallas_focus` once this fix is committed.

Fourth Kaggle P100 retry after the JAX 0.10.1 local update:

```text
run: 20260617_213002_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas measurement
progress: Mosaic GPU accepted the SMEM scratch allocation path.
cause: the full-Nx Pallas Thomas design stores 6 forward coefficients for
       block_b=128 and Nx=51 in shared memory, leading to
       smem_bytes=419848 > max_smem_bytes=49152 on P100.
decision: keep pallas_thomas_128 benchmark-only/standby. Do not spend more
          Kaggle runs on this full-Nx scratch design.
```

Future Pallas work should be a redesign rather than another compatibility
patch. Plausible directions are much smaller block sizes, recomputing or
streaming the backward coefficients to bound scratch, or moving directly to a
PCR/hybrid Pallas kernel whose scratch scales with stages rather than
`block_b * Nx`.

Bounded-SMEM retry on 2026-06-18:

```text
candidate: pallas_thomas_16
implementation: same exact Pallas Thomas kernel with BLOCK_B=16
scratch estimate at Nx=96, float32: 16 * 96 * 6 * 4 = 36,864 bytes
P100 max shared memory: 49,152 bytes
local validation: B=16, Nx=8, float32, interpret=True, matches Thomas64
Kaggle P100 run: 20260618_183720_linear_pallas_focus_NvidiaTeslaP100
result: failed before Pallas timing at B=1024, Nx=51
Mosaic GPU lowering: smem_bytes=60424 > max_smem_bytes=49152
```

The naive explicit scratch estimate was too optimistic because Mosaic GPU adds
substantial implicit shared-memory use during lowering. `pallas_thomas_16`
therefore joins `pallas_thomas_128` in standby.

Last Thomas-family Pallas retry before moving to PCR/hybrid:

```text
candidate: pallas_thomas_4
implementation: same exact Pallas Thomas kernel with BLOCK_B=4
scratch estimate at Nx=96, float32: 4 * 96 * 6 * 4 = 9,216 bytes
P100 max shared memory: 49,152 bytes
local validation: B=16, Nx=8, float32, interpret=True, matches Thomas64
local smoke: local_pallas_blocks_smoke, pallas_thomas_4/8/16 all max_abs_err
             4.575e-08 vs Thomas64
```

`pallas_thomas_8` is also exposed as a benchmark-only intermediate probe, but
linear extrapolation from the `128` and `16` failures suggests it may still be
too close to the P100 SMEM ceiling at `Nx=96`. The `linear_pallas_focus` preset
therefore uses `pallas_thomas_4` first to get a complete timing sweep. If `4`
is correct but obviously dominated by launch/program overhead, skip further
Thomas block-size tuning and move to Phase 3B Pallas PCR/hybrid.

First `pallas_thomas_4` P100 attempt:

```text
run: 20260618_184529_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: Mosaic GPU gmem->smem block copies must transfer a byte count divisible
       by the 128-byte warpgroup size. For BLOCK_B=4 and Nx=51, each float32
       input block transfers 4 * 51 * 4 = 816 bytes.
fix: pad Pallas internal storage lengths to multiples of 8 for main, edge,
     rhs, and output block specs, while keeping the Thomas loop bounded by the
     real Nx and slicing the returned output back to Nx.
local validation: local_pallas_padded_smoke at B=16, Nx=51, float32 matched
                  Thomas64 for pallas_thomas_4/8/16 with max_abs_err 6.493e-08
```

Second `pallas_thomas_4` P100 attempt after padding:

```text
run: 20260618_185101_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: Mosaic GPU could not infer the layout of the `jnp.arange` iota used to
       build vectorized batch indices for scratch/output stores.
fix: remove the artificial batch-index iota and use explicit Pallas
     `pl.ds(row, 1)` / `pl.ds(component, 1)` slices for scratch and output
     load/store helpers.
local validation: local_pallas_ds_smoke at B=16, Nx=51, float32 matched
                  Thomas64 for pallas_thomas_4/8/16 with max_abs_err 6.493e-08
```

Third `pallas_thomas_4` P100 attempt after replacing vectorized batch indices:

```text
run: 20260618_185700_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: the first coefficient access, `a00_ref[:, 0]`, lowers to a strided SMEM
       load with only 4 elements for BLOCK_B=4. Mosaic GPU requires this
       `load_strided` path to have a number of elements that is a multiple of
       128.
constraint: BLOCK_B=128 satisfies the strided-load width but exceeds P100 SMEM
            for the full-Nx Thomas scratch/input-block design; small BLOCK_B
            satisfies SMEM but violates the strided-load layout.
decision: close/standby pallas_thomas_4/8/16/128 for P100. Do not spend more
          Kaggle runs on Thomas-Pallas unless the kernel is redesigned around
          global-memory streaming or a non-columnar layout. If we continue
          Pallas, move to Phase 3B PCR/hybrid with these layout constraints
          designed in from the start.
```

---

## Phase 3B — Pallas hybrid PCR/Thomas

## Goal

Implement the most likely production GPU solver for `Nx=30-100`.

Candidate:

```text
pallas_hybrid_pcr_thomas_2x2
```

Initial Phase 3B spike implemented on 2026-06-18:

```text
candidate: pallas_pcr_128
implementation: benchmark-only Pallas PCR stage kernel
program shape: 128 fibers x 1 cable column
grid per stage: (batch_size / 128, Nx)
reason: preserve Mosaic GPU's 128-element strided-load width while avoiding
        the full-cable SMEM footprint that closed Thomas-Pallas.
solver flow: initialize SoA PCR arrays in JAX, run one Pallas stage kernel per
             PCR stride, then finish with the independent 2x2 solve.
local validation:
  - tests/unit/solvers/test_pallas_kernels.py passes in interpret=True
  - local_pallas_pcr128_smoke at B=128, Nx=8 matched pcr_soa/Thomas
  - local_pallas_pcr128_nx51_smoke at B=128, Nx=51 matched pcr_soa/Thomas
status: ready for first Kaggle P100 compile/timing via linear_pallas_focus
```

First P100 attempt:

```text
run: 20260618_192128_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: Mosaic GPU allows explicit GMEM block specs only for full-array trivial
       mappings. `pallas_pcr_128` used explicit GMEM for blocked outputs
       shaped (128, 1), which is rejected.
fix: keep explicit GMEM only for full-array input refs; let blocked outputs use
     the standard Pallas output block specs.
```

Second P100 attempt:

```text
run: 20260618_192526_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: `pl.load(...)` on dynamic GMEM refs lowered to Mosaic `masked_load`,
       which is not implemented for this Pallas GPU path.
fix: use direct ref indexing (`ref[index]`) for stage input loads instead of
     the compatibility `_pallas_load` helper.
local validation: local_pallas_pcr128_direct_index_smoke at B=128, Nx=51,
                  float32 matched pcr_soa/Thomas with max_abs_err 1.021e-07
```

Third P100 attempt:

```text
run: 20260618_193013_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: output stores used `value[:, None]`, which lowered through
       `broadcast_in_dim` on `WGStridedFragLayout(shape=(128,), vec_size=1)`;
       Mosaic GPU reports that layout as unsupported for this broadcast.
fix: keep `(128, 1)` output block specs but write with direct ref assignment
     `ref[:, 0] = value`, avoiding the explicit broadcast.
local validation: local_pallas_pcr128_direct_store_smoke at B=128, Nx=51,
                  float32 matched pcr_soa/Thomas with max_abs_err 1.021e-07
```

Fourth P100 attempt:

```text
run: 20260618_193442_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: Mosaic async-copy lowering requires every non-last GMEM stride to be a
       multiple of 16 bytes. The stage output buffers had shape
       (batch_size, 51), so the batch-row stride was 51 * 4 = 204 bytes.
fix: pad internal Pallas PCR work arrays to a four-column multiple. Padded
     columns are kept as identity diagonal / zero off-diagonal / zero RHS
     sentinels in every stage, and final outputs are sliced back to real Nx.
local validation: local_pallas_pcr128_padded_stride_smoke at B=128, Nx=51,
                  float32 matched pcr_soa/Thomas with max_abs_err 1.021e-07
```

Fifth P100 attempt and documentation check:

```text
run: 20260618_194249_linear_pallas_focus_NvidiaTeslaP100
status: failed before Pallas timing at B=1024, Nx=51
cause: after row-stride padding, Mosaic async-copy lowering rejected the
       minormost output block width: a `(128, 1)` float32 block copies only
       32 bits along the last dimension, but the lowering requires 128 bits.
fix: group four cable columns per Pallas program. The stage now uses
     `(128 fibers x 4 cable columns)`, preserving the 128-fiber batch width
     while making the float32 minormost block exactly 16 bytes / 128 bits.
local validation: local_pallas_pcr128_block4_smoke at B=128, Nx=51, float32
                  matched pcr_soa/Thomas with max_abs_err 1.021e-07
doc check: official JAX Pallas docs state that Mosaic GPU support is only for
           Hopper and newer GPUs. Kaggle P100 is Pascal, so it remains outside
           the documented target. However, a user Colab T4 smoke notebook
           compiles/runs Pallas successfully, so T4 should be treated as an
           empirical target rather than ruled out by the docs alone.
decision: pause Kaggle P100 Pallas runs. Keep `pallas_pcr_128` as a
          benchmark-only candidate and validate next on T4/Colab or Kaggle T4
          before deciding whether the Pallas PCR line is worth more work.
```

T4 validation attempt and notebook comparison:

```text
run: 20260618_200242_linear_pallas_focus_NvidiaTeslaT4
status: failed before Pallas timing at B=1024, Nx=51
progress: Kaggle provisioned Tesla T4 devices and non-Pallas controls ran:
          pcr_soa median 0.529 ms for B=1024, Nx=51, float32.
cause: current JAX 0.10.x Mosaic GPU lowering emitted
       `nvvm.cp.async.bulk.wait_group`, which NVVM reports as unsupported on
       T4 `sm_75`.
notebook check: `Copy of Pallas on GPU Demo.ipynb` is a real T4 Pallas smoke,
                but it pins an older stack:
                `jax==0.4.16.dev20230831`,
                `jaxlib==0.4.16.dev20230831+cuda12.cudnn89`,
                `jax_triton`, and `triton-nightly`.
                Its debug IR is Triton (`tt.func`), not the current Mosaic GPU
                lowering path used by this repo's JAX 0.10.x environment.
decision: close current-stack P100/T4 Mosaic-Pallas benchmarking. Reopen only
          for a Hopper+ Mosaic GPU, or as a separate legacy Triton/Pallas
          notebook spike if T4 custom kernels remain attractive.
```

## Phase 3B.1 — CuTe DSL / CUTLASS JAX custom-call scout

Status on 2026-06-18: reviewed the official JAX CuTe DSL guide and added a
standalone smoke harness:

```text
benchmark/cute_dsl/run_cute_dsl_smoke.py
benchmark/cute_dsl/cute_dsl_jax_kernels.py
```

The smoke follows the guide's vector-add `@cute.kernel` +
`cutlass.jax.cutlass_call` pattern and checks dependencies plus GPU compute
capability before importing the CuTe kernel module. It exits with
`status=skipped` by default when the runtime is incompatible, so it can be used
as a first gate on future GPU runtimes without breaking local CPU checks.

Hardware decision:

```text
CuTe DSL documented minimum: SM 8.0+ (Ampere)
Kaggle P100:                 SM 6.0 -> unsupported
Kaggle T4:                   SM 7.5 -> unsupported
Potential targets:           L4, A100, H100, or newer
```

Local result:

```text
command: python benchmark/cute_dsl/run_cute_dsl_smoke.py --n 512
status:  skipped
reason:  JAX backend is 'cpu', expected 'gpu'
```

Decision: do not spend Kaggle P100/T4 runs on CuTe DSL. Reopen only on an
Ampere-or-newer runtime, first by passing the standalone vector-add smoke, then
by attempting a single PCR-stage custom call. CuTe remains more plausible than
current-stack Mosaic-Pallas for non-Hopper custom kernels, but it is a separate
hardware/toolchain spike rather than the next P100 solver optimization.

## Phase 3B.2 — Triton standalone double-cable scout

Status on 2026-06-18: added a standalone Triton exact block-Thomas benchmark:

```text
benchmark/triton_solver/bench_double_cable_triton.py
benchmark/triton_solver/triton_double_cable_kernels.py
```

The candidate is intentionally simple:

```text
solver: triton_block_thomas
layout: row-major [B, Nx] SoA tensors
forward: one Triton program per fiber, sequential block-Thomas elimination
backward: one Triton program per fiber, sequential substitution
scratch: modified 2x2 upper block + modified RHS, stored globally
```

This is not yet a JAX custom call. The point is to test whether a small
hand-written Triton double-cable kernel is even in the right performance
neighborhood. The Kaggle preset runs a same-GPU JAX baseline first:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_triton_focus \
  --machine-shape NvidiaTeslaT4 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

Decision rule: if `triton_block_thomas` does not clearly beat `pcr_soa` on
steady-state median kernel time, close the Triton line. If it does beat
`pcr_soa`, the next step is either a JAX custom-call integration path or a
Triton PCR-stage kernel that better matches the current exact solver policy.

First T4 attempt on 2026-06-18:

```text
run: benchmark/results/kaggle/20260618_204657_linear_triton_focus_NvidiaTeslaT4
status: failed before Triton timing
baseline: JAX pcr_soa completed on Tesla T4
B=1024, Nx=51: 0.463 ms
B=1024, Nx=96: 1.045 ms
B=2048, Nx=51: 0.993 ms
B=2048, Nx=96: 1.693 ms
B=4096, Nx=51: 1.734 ms
B=4096, Nx=96: 2.990 ms
cause: `bench_double_cable_triton.py` was executed as a file from Kaggle's
       checkout, so `benchmark.triton_solver` was not importable.
fix: add repo-root/src `sys.path` bootstrap, matching the other benchmark
     entry points.
decision: rerun required; no Triton performance evidence yet.
```

Second T4 attempt on 2026-06-18:

```text
run: benchmark/results/kaggle/20260618_205135_linear_triton_focus_NvidiaTeslaT4
status: completed
gpu: 2x Tesla T4 provisioned by Kaggle; benchmark used default visible device
jax baseline: pcr_soa, JAX 0.10.2 CUDA backend
candidate: standalone Triton exact block-Thomas, two kernels plus global scratch

B=1024, Nx=51: pcr_soa 0.477 ms, triton_block_thomas 0.217 ms, speedup 2.199x
B=1024, Nx=96: pcr_soa 0.920 ms, triton_block_thomas 0.338 ms, speedup 2.720x
B=2048, Nx=51: pcr_soa 1.030 ms, triton_block_thomas 0.356 ms, speedup 2.892x
B=2048, Nx=96: pcr_soa 1.719 ms, triton_block_thomas 0.627 ms, speedup 2.743x
B=4096, Nx=51: pcr_soa 1.738 ms, triton_block_thomas 0.622 ms, speedup 2.795x
B=4096, Nx=96: pcr_soa 3.112 ms, triton_block_thomas 1.104 ms, speedup 2.820x

geomean speedup vs pcr_soa: 2.684x
speedup range: 2.199x-2.892x
max dense64-smoke abs error: ~3.97e-08
max block residual norm: ~3.60e-07

decision: keep Triton alive. This is the first custom-kernel candidate with a
          clear solver-only win over JAX pcr_soa on the focused full-batch
          linear systems. Do not route it publicly yet: this is a standalone
          Torch/Triton benchmark, not a JAX-integrated solver. The next gate is
          integration overhead and data ownership: a JAX custom-call path,
          DLPack zero-copy bridge, or narrow E2E prototype must preserve most
          of the solver-only gain without CPU copies.
```

PCR_SOA quick scout added before committing to Thomas-only Triton work:

```text
candidate: triton_pcr_soa
status: completed benchmark-only on Kaggle T4
implementation: global-memory SoA PCR with init/stage/final Triton kernels
stage count: ceil(log2(Nx)), one kernel launch per stride
purpose: answer whether a Triton PCR_SOA-style implementation is immediately
         competitive with triton_block_thomas before investing in integration.
expected risk: more launches and much larger global-memory work arrays than
               block Thomas; useful only if parallelism across Nx offsets that
               overhead on T4/P100-like GPUs.

run: benchmark/results/kaggle/20260618_210243_linear_triton_focus_NvidiaTeslaT4
status: completed
gpu: 2x Tesla T4 provisioned by Kaggle; benchmark used default visible device
jax baseline: pcr_soa, JAX 0.10.2 CUDA backend

B=1024, Nx=51: pcr_soa 0.534 ms, triton_block_thomas 0.209 ms, triton_pcr_soa 0.494 ms
B=1024, Nx=96: pcr_soa 0.903 ms, triton_block_thomas 0.351 ms, triton_pcr_soa 0.604 ms
B=2048, Nx=51: pcr_soa 1.035 ms, triton_block_thomas 0.369 ms, triton_pcr_soa 0.561 ms
B=2048, Nx=96: pcr_soa 1.833 ms, triton_block_thomas 0.621 ms, triton_pcr_soa 0.970 ms
B=4096, Nx=51: pcr_soa 1.794 ms, triton_block_thomas 0.624 ms, triton_pcr_soa 0.949 ms
B=4096, Nx=96: pcr_soa 3.091 ms, triton_block_thomas 1.127 ms, triton_pcr_soa 1.828 ms

triton_block_thomas geomean speedup vs JAX pcr_soa: 2.747x
triton_pcr_soa geomean speedup vs JAX pcr_soa: 1.619x
triton_pcr_soa runtime vs triton_block_thomas: 1.697x geomean slower
triton_pcr_soa runtime range vs triton_block_thomas: 1.521x-2.364x slower
triton_pcr_soa max dense64-smoke abs error: ~7.62e-08
triton_pcr_soa max block residual norm: ~5.79e-07

decision: keep triton_pcr_soa benchmark-only/standby. It proves that Triton can
          beat JAX pcr_soa for the PCR algebra too, but it is not competitive
          with the simpler block-Thomas Triton kernel on the focused full-batch
          cases. Focus Triton work on block-Thomas integration rather than PCR
          kernel tuning.
```

Block-Thomas integration gate added after the PCR_SOA scout:

```text
candidate: triton_block_thomas_jax_bridge
status: implemented experimental bridge, awaiting Kaggle timing
implementation:
  - accepts eager JAX arrays
  - converts inputs to Torch tensors through DLPack
  - runs the same exact Triton block-Thomas kernels
  - converts outputs back to JAX through DLPack
scope: benchmark/integration measurement only; not usable inside jax.jit
      and not public solver routing.

Updated `linear_triton_focus` now compares:
  1. JAX pcr_soa baseline
  2. pure Torch/Triton triton_block_thomas
  3. eager JAX -> DLPack/Torch -> Triton -> DLPack/JAX bridge

decision rule: if the bridge preserves most of the pure Triton speedup, build
               a narrow E2E prototype around this data boundary. If the bridge
               loses too much time, skip DLPack/Python routing and investigate
               a deeper integration path before touching production dispatch.
```

First bridge attempt on 2026-06-18:

```text
run: benchmark/results/kaggle/20260618_213014_linear_triton_focus_NvidiaTeslaT4
status: failed before bridge timing
cause: `src/axonscope/solvers/triton_thomas.py` was not tracked in Git, so
       Kaggle cloned commit `ac9f9c3` without the bridge module.

pure Triton Thomas still ran before the failure:
B=1024, Nx=51: 0.203 ms
B=1024, Nx=96: 0.344 ms
B=2048, Nx=51: 0.359 ms
B=2048, Nx=96: 0.619 ms
B=4096, Nx=51: 0.610 ms
B=4096, Nx=96: 1.105 ms

fix: track and commit `src/axonscope/solvers/triton_thomas.py`, then rerun the
     same `linear_triton_focus` preset.
```

Use static buckets:

```text
Nx_pad = 32
Nx_pad = 64
Nx_pad = 128
```

For each bucket, unroll or statically structure:

```text
PCR stages:
    stride = 1
    stride = 2
    stride = 4
optional:
    stride = 8

then local Thomas on residual blocks
```

## Why hybrid

Full PCR may do too much arithmetic for small Nx. Thomas is too sequential. Hybrid may be optimal for small cables.

## Variants to benchmark

```text
hybrid_block_4
hybrid_block_8
hybrid_block_16
full_pcr
```

## Go/no-go

Keep Pallas hybrid if:

```text
solver-only speedup >= 2x vs best JAX backend
or end-to-end speedup >= 1.5x on real double-cable workloads
```

Drop or postpone if:

```text
solver-only gain < 20%
end-to-end unchanged
implementation becomes fragile
numerical mismatch is hard to debug
```

---

## Phase 3C — Pallas full PCR

Only implement full Pallas PCR if hybrid is inconclusive or if JAX PCR is clearly bottlenecked by temporaries/gathers.

Expected value:

```text
strong control over PCR stages
less HBM traffic
fewer temporaries
potentially lower kernel count
```

But it is more work than JAX PCR and should be justified by trace evidence.

---

## Phase 3 deliverables

```text
1. pallas_thomas_small_nx backend.
2. pallas_hybrid_pcr_thomas backend if baseline justifies it.
3. Tests vs Thomas/PCR.
4. Performance report.
5. AUTO policy update.
```

---

# Cross-cutting optimizations

These are not the core solver rewrite, but they matter for end-to-end speed.

## 1. No dense zero Iinj

If intracellular stimulation is absent:

```text
do not materialize Iinj[B, Nt, Nx] = 0
```

Specialize the scan inputs so the solver sees no dense zero tensor.

## 2. Compact outputs

For performance claims, avoid full:

```text
Vm[B, Nt, Nx]
```

unless the real workload requires it.

Prefer:

```text
activation observer
threshold observer
center/probe recording
peak Vm
first spike time
```

## 3. Factorized extracellular drive

If possible, avoid materializing:

```text
Vext[B, Nt, Nx]
```

when it is really:

```text
footprint[B, Nx] * waveform[Nt]
```

Build the RHS inside the time scan from factorized inputs.

## 4. Static shapes

For repeated GPU calls, bucket:

```text
Nx -> 32/64/128
B  -> maybe padded batch buckets
```

This reduces recompilation and improves kernel specialization.

---

# Validation plan

## Solver-only validation

For every solver backend:

```text
Thomas float64 = reference
```

Test systems:

```text
1. random diagonally dominant systems
2. systems generated from real double-cable axons
3. extreme MRG-like boundary conditions
4. small Nx edge cases
5. padded Nx cases
```

Required tests:

```text
Nx = 2, 3, 4, 8, 16, 32, 51, 64, 96, 100, 128
B  = 1, 8, 512, 2048
dtype = float64, float32
```

Tolerances:

```text
float64 max_abs_error < 1e-9
float32 max_abs_error < 1e-5
no NaN/Inf
```

## Simulation validation

For real axon simulations:

```text
1. Vm trace error at nodes.
2. peak nodal Vm error.
3. first spike time error.
4. conduction velocity error.
5. activation yes/no agreement.
6. threshold agreement.
7. recruitment curve agreement.
```

Because all solvers are exact linear algebra alternatives, the physiological outputs should match Thomas to numerical tolerance.

## Scientific validation

Keep the exact double-cable path because the Abdollahi & Prescott paper shows that current flow through axial, submyelin, transmyelin, and extramyelin pathways changes conduction velocity and propagation reliability. If solver rewrites alter those outputs beyond numerical tolerance, the backend must be considered incorrect.

---

# Benchmark matrix

## Solver-only matrix

```text
B:      1, 8, 128, 512, 1024, 2048, 4096
Nx:     16, 32, 51, 64, 96, 100, 128
dtype:  float32, float64
solver: thomas, pcr_soa, pcr_hybrid, split_jacobi, split_gauss_seidel, split_richardson, assoc_backward, assoc_transfer, pallas_*
```

Metrics:

```text
compile time
steady-state time
node-solves/s
speedup vs Thomas
max_abs_error
max_rel_error
memory allocated
```

## End-to-end matrix

```text
B:      512, 1024, 2048, 4096
Nx:     32, 51, 64, 96, 100
Nt:     500, 1000
output: none, observer, center, full
Iinj:   none, dense zero, nonzero
Vext:   dense, factorized
solver: thomas, pcr_soa, split_best, assoc_best, pallas_best
```

Metrics:

```text
total wall time
compile time
steady-state time
GPU memory
kernel count
GPU occupancy from trace
activation/threshold agreement
```

---

# AUTO policy after benchmarking

Current:

```python
if device == "cpu":
    solver = THOMAS
elif device in {"cuda", "gpu", "metal", "rocm"}:
    solver = PCR_ADAPTIVE
else:
    solver = THOMAS
```

After Phase 1:

```python
if device == "cpu":
    solver = THOMAS
elif benchmark_table.supports("pcr_adaptive", device, B, Nx, output_policy):
    solver = PCR_ADAPTIVE
else:
    solver = THOMAS
```

After Phase 1.5:

```python
if split_best is stable and K <= 4 and faster than PCR_SOA:
    use split_best for the validated B/Nx/dt region
elif split_best needs K > 8 or changes physiology:
    keep it as experimental only
else:
    use PCR_ADAPTIVE, PCR_SOA, PCR, or THOMAS
```

After Phase 2:

```python
if ASSOCIATIVE_TRANSFER stable and faster:
    use ASSOCIATIVE_TRANSFER for its winning region
elif ASSOCIATIVE_BACKWARD is slightly faster and stable:
    use ASSOCIATIVE_BACKWARD for Thomas-like region
else:
    use split_best, PCR_ADAPTIVE, PCR_SOA, PCR, or THOMAS according to the
    benchmark table
```

After Phase 3:

```python
if pallas_hybrid wins for Nx bucket and B threshold:
    solver = PALLAS_PCR_HYBRID
elif pallas_thomas wins for small Nx:
    solver = PALLAS_THOMAS
else:
    solver = best JAX backend
```

All AUTO decisions must be benchmark-backed and stored in a small table.

---

# Recommended implementation order

## Week 1 — clean API + baseline

```text
1. Keep the existing `BatchOptions.double_cable_block_solver` contract clean.
2. Add or extend solver-only benchmarks.
3. Add JAX trace script or hotpath preset.
4. Confirm tests compare Thomas, PCR, PCR_SOA, and PCR_ADAPTIVE.
5. Run baseline on CPU and GPU.
```

## Week 2 — Phase 1 PCR

```text
1. Re-run `pcr_adaptive` end-to-end after the SoA/matrix split.
2. Add batch-native PCR_SOA only if solver-only evidence still justifies it.
3. Add Nx padding buckets only if they reduce recompilation or improve kernels.
4. Update AUTO policy only from benchmark-backed crossover data.
5. Decide whether hybrid PCR is worth implementing.
```

## Week 3 — Phase 1 hybrid / cleanup

```text
1. Add PCR_HYBRID if PCR full is promising but too expensive.
2. Add no-Iinj specialization if not already clean.
3. Ensure compact outputs for fair solver benchmarking.
4. Update benchmark reports.
```

## Week 3.5 — Phase 1.5 two-rail split solver

```text
1. Add split_double_cable_block_system_soa. [done]
2. Implement split_jacobi_fixed_k. [done as split_jacobi_4/8]
3. Implement split_gauss_seidel_fixed_k. [done as split_gs_2/3/4/8]
4. Implement residual checker. [done]
5. Test K = 4, 8 with rhs_guess initialization in solver-only benchmark. [done]
6. Compare speed and residual/error vs Thomas and PCR_SOA. [done]
7. Run physiology/E2E validation only if solver-only residuals and speed are credible. [E2E focus wired; Kaggle next]
8. Decide whether split_best should enter AUTO policy.
```

## Week 4 — Phase 2 associative scan

```text
1. Implement associative backward substitution.
2. Benchmark and decide whether to keep it.
3. Implement dense 5x5 transfer scan prototype.
4. Validate stability float64 and float32.
5. Decide whether to optimize transfer scan or shelve it.
```

## Week 5+ — Phase 3 Pallas

```text
1. Implement pallas_thomas_small_nx.
2. Benchmark against JAX Thomas and JAX PCR.
3. If promising, implement pallas_hybrid_pcr_thomas.
4. Add tests and AUTO policy integration.
5. Keep Pallas behind an optional backend flag until stable.
```

---

# Expected final backend ranking

Likely final ranking for the target regime:

```text
1. PALLAS_PCR_HYBRID          best if custom kernel pays off
2. SPLIT_GAUSS_SEIDEL/JACOBI  possible practical winner if K<=4 and scalar solves are very fast
3. PCR_SOA_BATCHED            best robust pure-JAX direct candidate
4. ASSOCIATIVE_TRANSFER       possible winner if stable and XLA lowers well
5. ASSOCIATIVE_BACKWARD       modest improvement over Thomas
6. THOMAS                     CPU/default fallback
```

---

# Main risks

## Risk 1 — PCR does more FLOPs

PCR reduces dependency depth but increases arithmetic. For `Nx=30-100`, full PCR may not always win.

Mitigation:

```text
test hybrid PCR/Thomas
use AUTO policy
do not remove Thomas
```

## Risk 2 — associative transfer instability

Transfer-matrix methods can amplify errors.

Mitigation:

```text
prototype dense first
validate float64 and float32
add condition diagnostics
keep PCR as robust fallback
```

## Risk 3 — Split iterative convergence

The two-rail split solver is exact only when iterated to convergence. Fixed-K versions may introduce small linear-solve errors that can accumulate over time.

Mitigation:

```text
compare to Thomas every time
monitor residuals
use previous-timestep initialization
start with conservative K=4 or K=8
fall back to PCR/Thomas for validation or difficult regimes
never claim exactness for fixed-K unless residual and physiology match
```

## Risk 4 — Pallas complexity

Pallas kernels are harder to maintain.

Mitigation:

```text
only implement after JAX baselines
keep behind backend flag
write extensive solver-only tests
keep JAX fallback
```

## Risk 5 — end-to-end bottleneck is not solver

If output or Vext construction dominates, solver improvements may not show end-to-end.

Mitigation:

```text
profile first
use compact outputs
specialize no-Iinj
factorize Vext
report solver-only and end-to-end separately
```

---

# Definition of success

## Minimum success

```text
PCR_SOA official backend
correctness vs Thomas
GPU solver-only speedup >= 1.5x for B>=1024, Nx=64
end-to-end speedup measurable with compact outputs
```

## Strong success

```text
best backend gives >= 2x solver-only speedup
end-to-end double-cable speedup >= 1.5x
no physiological differences vs Thomas beyond numerical tolerance
AUTO policy chooses correct backend by B/Nx/device
if split iterative wins, K <= 4 with stable residuals and unchanged activation/threshold outputs
```

## Excellent success

```text
Pallas or associative backend gives >= 3x solver-only speedup
end-to-end double-cable GPU becomes clearly favorable for B>500
all threshold/recruitment workflows run through exact double-cable without pseudo approximation
```

---

# Sources

## AxonScope code

- `src/axonscope/solvers/common.py` on branch `bench-colab` contains the current scalar block-Thomas solver and existing PCR/PCR_SOA prototypes.
  - https://raw.githubusercontent.com/louisreg/AxonScope/bench-colab/src/axonscope/solvers/common.py

## JAX tools

- `jax.lax.associative_scan`
  - https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html

- JAX Pallas
  - https://docs.jax.dev/en/latest/pallas/index.html
  - https://docs.jax.dev/en/latest/pallas/quickstart.html

- `jax.profiler.trace`
  - https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html

## Scientific motivation

- Abdollahi, N. and Prescott, S. A. (2024). *Impact of Extracellular Current Flow on Action Potential Propagation in Myelinated Axons*. Journal of Neuroscience.
  - Uploaded file: `e0569242024.full.pdf`
  - DOI page: https://doi.org/10.1523/JNEUROSCI.0569-24.2024

Key paper takeaways used in this roadmap:

```text
1. The adapted MRG model includes nodes, paranodes, juxtaparanodes, and internodes.
2. Condition 1 grounds the extramyelin layer and represents the conventional absorptive-boundary double-cable model.
3. Conditions 2 and 3 disconnect internodal extramyelin compartments from ground and vary longitudinal extramyelin resistance.
4. Conduction velocity differs because extracellular conditions change how much axial current reaches the next node.
5. Transmyelin current is high when extramyelin is grounded or longitudinally low-resistance, slowing conduction.
6. Multifiber Condition 2* validates the dense-fascicle interpretation.
7. Myelin thickness effects depend strongly on extracellular boundary conditions.
```
