# AxonScope Double-Cable GPU Optimization Plan

**Target workload:** double-cable fiber simulations with `Nx = 30–100` compartments and `B > 500` fibers.  
**Primary goal:** make double-cable GPU execution scale well for small-to-medium cable lengths by exploiting the batch axis, reducing dense input/output materialization, and only moving to more complex spatial-parallel solvers if the simpler batched approach is insufficient.

This plan is deliberately written as an implementation sequence. Each step has a goal, files likely to touch, concrete tests, metrics, and a go/no-go decision.

---

## 0. Current understanding and optimization target

### 0.1 What is currently known

The current evidence from the `bench-colab` branch suggests:

- Single-cable GPU scaling is already comparatively good.
- Double-cable GPU scaling is weaker, especially around `B <= 600`.
- For double-cable extracellular long runs, CPU wins at `n=100`, GPU crosses over around `n=300`, and GPU wins clearly at `n=600`.
- The suspected hot path is the `Nt` time scan plus the per-time-step forward and reverse `Nx` scans inside `solve_block_tridiagonal_2x2_scalar`.
- For double-cable, the GPU currently gets mostly batch-axis parallelism, not much spatial parallelism.
- True factorized extracellular forcing, `waveform[Nt] * footprint[B, Nx]`, is deferred in the public API roadmap, but it is still a strong internal benchmark/prototype target.
- Dense zero `Iinj[B, Nt, Nx]` materialization has already been identified as something to avoid.

Relevant current files:

- `todo.md`
- `src/axonscope/solvers/common.py`
- `src/axonscope/solvers/batch_kernels.py`
- `src/axonscope/dispatcher/execution.py`
- `benchmark/hotpaths/`

### 0.2 The target regime changes the solver strategy

For `Nx = 30–100`, the most important GPU optimization is not necessarily a mathematically elegant parallel solver over `Nx`. The cable is short. A parallel cyclic reduction solver may reduce dependency depth, but it may also increase arithmetic, memory traffic, and implementation complexity.

For this workload, prioritize:

1. Large batch utilization.
2. Avoiding dense zero inputs.
3. Avoiding full retained `Vm[B, Nt, Nx]` outputs unless explicitly requested.
4. Using an explicitly batched double-cable Thomas solver with `B` as the main GPU axis.
5. Specializing for small static `Nx` families, such as 32, 64, and 128.
6. Testing PCR/cyclic reduction only after the simpler batched Thomas path has been exhausted.

---

## 1. Establish a canonical benchmark matrix for the real workload

### Goal

Create one stable benchmark that exactly represents the intended production regime: `Nx = 30–100`, `B > 500`, double-cable, extracellular stimulation, mostly trace-free or compact-output workflows.

Do not optimize against random existing examples. Optimize against this matrix.

### Files likely to touch

- `benchmark/hotpaths/run.py`
- `benchmark/hotpaths/workloads.py` or equivalent workload registry
- `benchmark/hotpaths/README.md`
- `benchmark/hotpaths/COLAB.md`
- Optional: `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`

### Add a workload family

Add a named benchmark family such as:

```text
small_nx_large_batch_double_cable_gpu
```

It should sweep:

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096
Nt = 1000 by default, or duration=10 ms and dt=0.01 ms
recording = none, center, full
Iinj = None
Vext = dense precomputed initially
backend = JAX CPU, JAX GPU
precision = current default, plus float32 if not already default
```

### Required benchmark variants

Start with these variants:

```text
A. double-cable extracellular, dense Vext, Iinj=None, recording=full
B. double-cable extracellular, dense Vext, Iinj=None, recording=center
C. double-cable extracellular, dense Vext, Iinj=None, recording=none
D. double-cable extracellular, dense Vext, Iinj=None, activation observer only
E. solver-only precomputed double-cable RHS, no dispatch, no input preparation
```

Variant `E` is essential. It separates solver throughput from:

- public API overhead,
- dispatch planning,
- extracellular input preparation,
- result packaging,
- Python object splitting.

### Metrics to collect

For every run, record at least:

```text
total wall time
kernel enqueue time
kernel wait time
input preparation time
result packaging time
compile/warmup classification
B
Nx
Nt
recording mode
observer mode
Iinj mode: none / dense / sparse
Vext mode: dense / factorized / zero
output bytes
input bytes
node_steps_per_second = B * Nt * Nx / kernel_wait_time
fiber_steps_per_second = B * Nt / kernel_wait_time
```

### Acceptance criteria

This step is complete when:

- The workload can be run locally on CPU.
- The workload can be run on Colab GPU.
- Every result row includes `B`, `Nx`, `Nt`, recording mode, observer mode, and kernel timing.
- Solver-only timing can be compared against end-to-end timing.

### Go/no-go decision

Do not start solver rewrites before this benchmark exists. Otherwise, there is no reliable way to know whether a solver change improved the real workload or just changed overhead elsewhere.

---

## 2. Re-run the current double-cable baseline on the target matrix

### Goal

Build a clean baseline before changing solver code.

### Command shape

The exact command depends on the current benchmark runner, but the intended invocation should be equivalent to:

```bash
python -m benchmark.hotpaths.run \
  --workload small_nx_large_batch_double_cable_gpu \
  --backend jax \
  --device cpu,gpu \
  --warmup 2 \
  --repeat 5 \
  --jax-log-compiles
```

Also run the solver-only variant:

```bash
python -m benchmark.hotpaths.run \
  --workload small_nx_large_batch_double_cable_solver_only \
  --backend jax \
  --device cpu,gpu \
  --warmup 2 \
  --repeat 10
```

### Required output table

For each `Nx` and `B`, produce a table like:

```text
Nx | B | CPU total | GPU total | CPU kernel_wait | GPU kernel_wait | GPU speedup | node_steps/s
```

### Things to look for

Look for these patterns:

1. GPU only wins once `B` is large.
2. GPU speedup improves as `B` increases.
3. GPU speedup does not improve much as `B` increases, which may indicate memory/output bottlenecks.
4. Solver-only speedup is good, but end-to-end speedup is poor, which means preparation or packaging is the issue.
5. Solver-only speedup is poor, which means the solver itself is the issue.

### Acceptance criteria

This step is complete when there is a baseline table for:

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096
recording = none, center, full
solver-only = yes/no
```

### Go/no-go decision

If `recording=full` is slow but `recording=none` is fast, prioritize compact outputs and observers before touching the solver.

If `solver-only` is slow on GPU even for `B >= 2048`, prioritize the solver layout and kernel structure.

---

## 3. Verify and benchmark the zero-Iinj specialization

### Goal

Make sure double-cable runs with no intracellular current do not create, pass, or scan over dense zero arrays of shape `B x Nt x Nx`.

This is a low-risk, high-priority optimization because the intended workload is extracellular double-cable stimulation, often with `Iinj=None`.

### Files likely to touch

- `src/axonscope/solvers/batch_kernels.py`
- `src/axonscope/backends/jax/group_runner.py`
- Any batch lowering code that constructs `intracellular_current_density_mid`
- Benchmark workload metadata

### Desired behavior

When there is no intracellular stimulation:

```text
Iinj_abs_mid = None
```

should stay `None` all the way into the double-cable kernel.

The kernel should not receive:

```text
jnp.zeros((B, Nt, Nx))
```

or an equivalent dense zero input.

### Implementation checklist

- [ ] Audit all paths that call the double-cable batch kernel.
- [ ] Confirm retained-output double-cable runs do not materialize dense zero `Iinj`.
- [ ] Confirm observer-only double-cable runs do not materialize dense zero `Iinj`.
- [ ] Add benchmark metadata such as `i_inj_mode="none"` and `i_inj_dense_bytes_skipped`.
- [ ] Add a unit test or benchmark assertion that no dense zero `Iinj` shape is recorded for the no-Iinj case.

### Test matrix

Run:

```text
Nx = 32, 51, 96
B  = 512, 1024, 2048
recording = none, center, full
Iinj = None
Vext = dense
```

Compare before/after:

```text
input preparation time
kernel enqueue time
kernel wait time
total wall time
peak memory if available
```

### Acceptance criteria

- No dense zero `Iinj[B, Nt, Nx]` is materialized or passed into the backend for double-cable no-Iinj runs.
- Metadata explicitly reports that the dense input was skipped.
- GPU total time improves or stays neutral.
- CPU total time improves or stays neutral.
- Numerical results match the baseline within tolerance.

### Go/no-go decision

If this improves total time by more than 5–10%, keep it immediately.

If it barely changes total time but reduces memory, keep it anyway because it is structurally correct and helps larger `B`.

---

## 4. Make compact output and observer-only double-cable execution a first-class path

### Goal

Avoid retaining full voltage traces when the final workload only needs activation, threshold, recruitment, conduction status, or compact summary values.

For `B > 500`, `Nt = 1000`, and `Nx = 100`, retaining one `float32` voltage tensor costs:

```text
B * Nt * Nx * 4 bytes
= 500 * 1000 * 100 * 4
= 200 MB
```

Double-cable may carry additional state and intermediate arrays, so full output can dominate memory traffic.

### Files likely to touch

- `src/axonscope/dispatcher/execution.py`
- `src/axonscope/solvers/batch_kernels.py`
- `src/axonscope/backends/jax/group_runner.py`
- `src/axonscope/results/`
- Observer-related files
- Tests around `Recording.none()` and observations

### Current blocker to check

The current dispatcher rejects batched double-cable groups when observers are present. That means double-cable observer-only workflows may fall back to scalar execution or otherwise fail to use the fast batch backend.

### Implementation checklist

- [ ] Identify exactly why `observers and group.mode == "double"` is disabled.
- [ ] Add support for the simplest double-cable activation observer first.
- [ ] Do not support every observer at once.
- [ ] Start with homogeneous, unpadded double-cable groups.
- [ ] Return compact batched observations without splitting into one huge Python object per fiber during the hot path.
- [ ] Keep full `Vm` recording as an explicit debug/analysis mode, not the default for studies.

### First observer to support

Start with a minimal activation observer:

```text
Inputs tracked per fiber:
- max Vm at selected node or nodes
- first threshold crossing time if any
- activated boolean
- optional crossing compartment index
```

Output shape should be compact:

```text
activated: [B]
first_crossing_t: [B]
max_vm: [B] or [B, P] for P probes
```

Do not return:

```text
Vm: [B, Nt, Nx]
```

unless explicitly requested.

### Test matrix

Run:

```text
Nx = 32, 51, 96
B  = 512, 1024, 2048, 4096
recording = none
observer = activation
Iinj = None
Vext = dense
```

Compare against:

```text
recording = center
recording = full
scalar observer path if currently forced
```

### Acceptance criteria

- Double-cable observer-only runs stay batched.
- No full `Vm[B, Nt, Nx]` output is retained.
- Results match post-hoc activation computed from full traces on small validation cases.
- `results.split_batch` or equivalent packaging overhead remains small.
- GPU speedup improves at `B >= 512`.

### Go/no-go decision

If observer-only double-cable removes a large fraction of total time, prioritize study API integration before deeper solver work.

If solver-only remains the dominant cost, continue to the batched solver steps.

---

## 5. Build an explicit batched block-Thomas solver for small `Nx`, large `B`

### Goal

Replace or supplement the current scalar-shaped 2x2 block-tridiagonal solve with a solver whose primary internal layout is explicitly batched.

The key idea is:

```text
for each spatial index i:
    update all B fibers in parallel
```

The scan over `Nx` remains sequential, but each scan step is a large vectorized operation over `B`.

This is likely the best first solver change for `Nx = 30–100` and `B > 500`.

### Files likely to touch

- `src/axonscope/solvers/common.py`
- `src/axonscope/solvers/batch_kernels.py`
- New optional file: `src/axonscope/solvers/double_cable_batched.py`
- Tests under `tests/`
- Benchmarks under `benchmark/hotpaths/`

### Proposed function

Add a function with an explicit batched contract:

```python
def solve_block_tridiagonal_2x2_batched(
    a00,  # [Nx] or [B, Nx]
    a01,  # [Nx] or [B, Nx]
    a10,  # [Nx] or [B, Nx]
    a11,  # [Nx] or [B, Nx]
    off0, # [Nx-1] or [B, Nx-1]
    off1, # [Nx-1] or [B, Nx-1]
    rhs0, # [B, Nx]
    rhs1, # [B, Nx]
):
    """Return x0, x1 with shape [B, Nx]."""
```

### Initial implementation strategy

Use `jax.lax.scan` over `Nx`, but carry vectors of shape `[B]`:

```text
forward carry:
    c00_prev: [B]
    c01_prev: [B]
    c10_prev: [B]
    c11_prev: [B]
    d0_prev:  [B]
    d1_prev:  [B]

forward output per i:
    c00_i: [B]
    c01_i: [B]
    c10_i: [B]
    c11_i: [B]
    d0_i:  [B]
    d1_i:  [B]
```

Then run reverse scan with carry:

```text
next0: [B]
next1: [B]
```

and produce:

```text
x0: [B, Nx]
x1: [B, Nx]
```

### Layout variants to benchmark

Implement or at least benchmark these two layouts:

```text
Variant A: rhs0/rhs1 stored as [B, Nx]
Variant B: rhs0/rhs1 stored as [Nx, B]
```

For `lax.scan` over `Nx`, `[Nx, B]` can be more natural because each scan slice is contiguous as `[B]`.

However, the rest of the code may naturally use `[B, Nt, Nx]`. Test both rather than guessing.

### Correctness tests

Add tests comparing:

```text
current solve_block_tridiagonal_2x2_scalar vmapped over B
new solve_block_tridiagonal_2x2_batched
```

Test cases:

```text
Nx = 2, 3, 8, 32, 51, 96
B  = 1, 2, 17, 512
random well-conditioned coefficients
real double-cable coefficients
float32
float64 if supported
```

Assertions:

```text
max_abs_error < tolerance
max_rel_error < tolerance
no NaNs
same shape
same dtype
```

Use a stricter tolerance for float64 and a realistic tolerance for float32.

### Solver-only benchmark

Benchmark only this function with precomputed coefficients and RHS:

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096, 8192
repeats = enough to stabilize timing
```

Compare:

```text
current vmap(scalar solver)
new batched solver [B, Nx]
new batched solver [Nx, B]
```

### Acceptance criteria

Keep the new solver if:

- Correctness matches the existing solver.
- It improves solver-only GPU time by at least 15–20% for `B >= 512`, or
- It improves GPU memory behavior and does not regress time, or
- It gives a clearer path for Pallas/Triton specialization.

### Go/no-go decision

If the explicit batched Thomas solver improves GPU throughput significantly, continue specializing it before attempting PCR.

If it does not improve solver-only performance, inspect generated HLO/profile. The current vmap path may already be equivalent, and the next step should be either layout specialization or a custom kernel.

---

## 6. Integrate the explicit batched solver into the double-cable time step

### Goal

Make the double-cable production kernel use the explicit batched solver for homogeneous or compatible groups.

### Files likely to touch

- `src/axonscope/solvers/batch_kernels.py`
- `src/axonscope/solvers/common.py`
- New solver module if created
- Tests for batch kernel equivalence

### Integration approach

Start with the narrowest safe case:

```text
mode = double-cable
geometry_shared = true
has_padding = false
Iinj = None
recording = none or center
Vext = dense
Nx in target range
```

Do not immediately replace all double-cable execution paths.

Add an internal switch such as:

```text
double_cable_solver="current" | "batched_thomas"
```

or a private feature flag in benchmark code.

### Validation cases

Run small deterministic simulations comparing:

```text
old double-cable kernel
new double-cable kernel
```

Use:

```text
Nx = 8, 32, 51
B = 2, 5, 32
Nt = 10, 100
recording = full
```

Check:

```text
Vm traces
Vi/Ve if exposed internally
activation observer if available
no NaNs
same boundary behavior
```

### Performance benchmark

Run the canonical matrix:

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096
recording = none, center, full
Iinj = None
Vext = dense
```

### Acceptance criteria

- No correctness regression.
- New solver path is faster or neutral for `B >= 512`.
- No unacceptable compile-time increase.
- No output API change.

### Go/no-go decision

If the production integration improves solver-only but not end-to-end timing, the bottleneck has moved to input/output. Continue to factorized forcing and compact output before deeper solver work.

---

## 7. Specialize for static small-`Nx` families

### Goal

Exploit the fact that the real workload uses `Nx = 30–100`, not arbitrary large `Nx`.

The GPU code can be more efficient if shapes are stable and fall into a small number of static families.

### Candidate shape families

Use padded internal sizes:

```text
actual Nx <= 32   -> internal Nx_pad = 32
actual Nx <= 64   -> internal Nx_pad = 64
actual Nx <= 128  -> internal Nx_pad = 128
```

For the current target range, this covers all cases.

### Why this may help

Static padded sizes may:

- reduce recompilation variety,
- improve XLA optimization,
- simplify future Pallas/Triton kernels,
- make memory layout predictable,
- allow manual unrolling or partial unrolling,
- avoid many shape-specialized compiled executables for small `Nx` changes.

### Files likely to touch

- Dispatch planning / batch grouping code
- Padding utilities
- Batch kernels
- Recording selector handling
- Observer masks
- Benchmarks

### Important warning

Padding double-cable is only safe if masks are handled correctly.

The padded compartments must not affect:

```text
activation decisions
recorded probes
boundary conditions
solver coefficients
summary statistics
```

Do not add broad public padding until correctness is fully tested.

### Prototype approach

Start solver-only, not full public dispatch.

Create a benchmark-only padded solver path:

```text
rhs[B, Nx] -> pad to rhs[B, Nx_pad]
coefficients -> pad with safe identity rows or masked rows
solve
slice output back to Nx
```

Then compare to the unpadded solver.

### Test matrix

```text
Nx actual = 30, 32, 51, 63, 64, 96, 100
Nx_pad    = 32, 64, 128
B         = 512, 1024, 2048, 4096
```

### Acceptance criteria

- Padded outputs match unpadded outputs on real compartments.
- Padded compartments do not influence real compartments.
- Compile cache behavior improves or stays manageable.
- Runtime improves or stays neutral.

### Go/no-go decision

If padding hurts runtime due to extra work, do not use it for JAX `lax.scan` kernels yet. Keep the idea for future Pallas/Triton kernels.

---

## 8. Add factorized extracellular forcing internally

### Goal

Avoid dense materialization of:

```text
Vext[B, Nt, Nx]
```

when the drive is naturally:

```text
waveform[Nt] * footprint[B, Nx]
```

For stimulation studies, this can be a major memory and transfer improvement.

### Files likely to touch

- Extracellular lowering code
- `src/axonscope/solvers/batch_kernels.py`
- Typed drive / footprint code
- Benchmark workload builders
- Future Phase 8 study API code

### Prototype contract

Do this first as an internal benchmark-only path:

```python
def run_double_cable_factorized_drive(
    footprint,   # [B, Nx]
    waveform,    # [Nt]
    runtime,
    recording,
    observers,
):
    ...
```

Inside the time scan:

```python
vext_t = waveform[t] * footprint
```

Then build RHS for that time step without constructing `Vext[B, Nt, Nx]`.

### Add multi-amplitude support

For threshold/recruitment, support:

```text
waveform[A, Nt] or amplitude[A] * waveform[Nt]
footprint[B, Nx]
```

Then the effective batch can become:

```text
B_effective = B_fibers * A_amplitudes
```

This is important because `B = 500` is only moderately large for GPU. If each threshold iteration evaluates 4–16 amplitude candidates, the GPU sees:

```text
500 fibers * 8 amplitudes = 4000 lanes
```

which is much more favorable.

### Benchmark variants

Compare:

```text
A. dense Vext[B, Nt, Nx]
B. factorized waveform[Nt] * footprint[B, Nx]
C. factorized multi-amplitude waveform[A, Nt] * footprint[B, Nx]
```

For:

```text
Nx = 32, 51, 96
B  = 512, 1024, 2048
A  = 1, 4, 8, 16
recording = none
observer = activation
```

### Metrics

Collect:

```text
input bytes
peak memory if available
kernel wait time
total time
node_steps/s
effective_lane_steps/s = B * A * Nt / time
```

### Acceptance criteria

- Factorized drive matches dense drive numerically.
- Dense `Vext[B, Nt, Nx]` is not created in the factorized path.
- Memory use decreases substantially.
- Runtime improves or stays neutral.
- Multi-amplitude path improves GPU utilization for `B around 500`.

### Go/no-go decision

If factorized forcing improves end-to-end time more than solver changes, prioritize the study/reuse API around factorized drives before implementing PCR.

---

## 9. Increase effective batch size in threshold and recruitment workflows

### Goal

Use the GPU efficiently even when the number of physical fibers is only around 500.

For small `Nx`, GPU utilization may require a larger batch than the physical fiber count. The clean solution is to batch over study dimensions as well.

### Effective batch dimensions

Treat this as the real GPU batch:

```text
B_effective = n_fibers * n_amplitudes * n_conditions * n_electrode_configs
```

Examples:

```text
512 fibers * 8 amplitudes = 4096 lanes
512 fibers * 4 amplitudes * 2 electrode configs = 4096 lanes
1000 fibers * 4 amplitudes = 4000 lanes
```

### Implementation strategy

For threshold search, do not run one amplitude per kernel call if avoidable.

Instead, evaluate a bracket or candidate set per call:

```text
amplitude_candidates = [a0, a1, a2, ..., a7]
```

Then reduce compact observer results to update thresholds.

### Files likely to touch

- Future study API implementation
- Benchmark workload builder
- Factorized drive path
- Observer output containers

### First benchmark-only prototype

Build a benchmark that simulates:

```text
B fibers
A amplitudes
same footprint[B, Nx]
same waveform[Nt]
activation observer only
```

Return:

```text
activated[B, A]
first_crossing_t[B, A]
```

Do not retain traces.

### Test matrix

```text
B = 512, 1024
A = 1, 2, 4, 8, 16
Nx = 32, 51, 96
```

### Acceptance criteria

- GPU throughput improves as `A` increases up to a useful point.
- The per-amplitude marginal cost is lower when amplitudes are batched than when they are run as separate calls.
- Output remains compact.
- Threshold logic can consume compact activation results.

### Go/no-go decision

If amplitude batching gives strong speedups, this may be more important than a new spatial solver for the real product workflow.

---

## 10. Profile the explicit batched solver before writing a custom kernel

### Goal

Determine whether JAX/XLA is generating good code for the explicit batched Thomas solver.

Do this before investing in Pallas, Triton, CUDA FFI, or PCR.

### What to inspect

Use available tools to inspect:

```text
kernel count
kernel duration
fusion behavior
memory bandwidth
register pressure if available
HLO shape/layout
compile time
```

### Questions to answer

1. Is the forward scan one fused region or many small kernels?
2. Is the reverse scan one fused region or many small kernels?
3. Are transposes inserted between `[B, Nx]` and `[Nx, B]` layouts?
4. Is time spent in memory reads/writes or arithmetic?
5. Does increasing `B` improve occupancy?
6. Does padding to 64 or 128 improve or hurt?

### Acceptance criteria

This step is complete when there is a short profiling note answering the six questions above.

### Go/no-go decision

If XLA generates efficient fused code and runtime is acceptable, avoid custom kernels.

If XLA generates poor layout/transposition or many small kernels, consider Pallas/Triton.

---

## 11. Prototype a small-`Nx`, large-`B` custom kernel only if needed

### Goal

Create a specialized GPU kernel for the target regime if the JAX implementation cannot achieve good performance.

This is not the first optimization. Do it only after the explicit batched solver and factorized-output changes are measured.

### Candidate technologies

Possible implementation paths:

```text
Pallas kernel
Triton kernel
CUDA custom call / FFI
cuSPARSE batched or interleaved routines if applicable
```

### Kernel design target

Specialize for:

```text
Nx_pad = 32, 64, 128
B large
float32 first
homogeneous/shared coefficients first
Iinj = None first
observer-only or compact output first
```

### First custom-kernel scope

Do not implement the whole simulator first.

Start with solver-only:

```text
input: coefficients + rhs0[B, Nx] + rhs1[B, Nx]
output: x0[B, Nx] + x1[B, Nx]
```

Then integrate into one time step. Then integrate into the full `Nt` scan.

### Acceptance criteria

A custom kernel is worth keeping only if it improves the explicit JAX batched solver by at least:

```text
>= 25% solver-only speedup
or
>= 15% end-to-end speedup
```

for the real target matrix:

```text
Nx = 32, 51, 96
B >= 512
```

### Go/no-go decision

If the custom kernel gives only marginal gains, do not maintain it. The maintenance cost is too high unless the speedup is clear.

---

## 12. Test block-PCR or cyclic reduction after the batched Thomas path

### Goal

Reduce the sequential dependency depth over `Nx` if and only if the batched Thomas solver remains the bottleneck.

### Why this is not step 1

For `Nx = 30–100`, PCR has a tradeoff:

```text
Thomas:
    work  = O(Nx)
    depth = O(Nx)
    memory = simple

PCR / cyclic reduction:
    work  = O(Nx log Nx)
    depth = O(log Nx)
    memory = higher
    implementation = more complex
```

At `Nx = 51`, the reduced dependency depth may help, but the extra work and memory traffic may cancel it out.

### Prototype scope

Start solver-only:

```text
2x2 block tridiagonal
small static Nx
B large
float32
shared or homogeneous coefficients first
```

### Compare against

```text
current scalar/vmap solver
explicit batched Thomas solver
custom small-Nx Thomas kernel if implemented
```

### Test matrix

```text
Nx = 32, 51, 64, 96, 128
B  = 512, 1024, 2048, 4096, 8192
```

### Acceptance criteria

PCR is worth integrating only if:

- It beats explicit batched Thomas for the real target range.
- It does not introduce unacceptable numerical instability.
- It does not make compile time or memory usage unacceptable.
- It improves end-to-end workloads, not only isolated synthetic solves.

### Go/no-go decision

If PCR wins only at `Nx >= 128`, do not prioritize it for the stated workload. Keep it as a future long-cable optimization.

---

## 13. Add backend selection rules for double-cable workloads

### Goal

Avoid sending small workloads to GPU when CPU is faster, and avoid sending large workloads to CPU when GPU is faster.

### Current evidence to encode

The existing benchmark evidence indicates roughly:

```text
B = 100: CPU faster
B = 300: GPU crossover region
B = 600: GPU clearly faster
```

This must be recalibrated after the optimizations above.

### Proposed policy inputs

Backend choice should consider:

```text
mode: single / double
Nx
B
Nt
recording mode
observer mode
Iinj mode
Vext mode
device availability
precision
compile cache warm/cold state if known
```

### Initial heuristic

After new benchmarks, define a conservative heuristic such as:

```text
if mode == double and device == gpu:
    if recording == full and B < threshold_full[Nx]:
        prefer CPU or warn
    if recording == none/observer and B >= threshold_compact[Nx]:
        prefer GPU
```

Use measured thresholds, not guesses.

### Files likely to touch

- `src/axonscope/performance.py`
- Solver/backend options
- Dispatch planning
- Benchmark metadata
- Documentation

### Acceptance criteria

- Backend selection is based on measured benchmark rows.
- Users can override it.
- Benchmark metadata records what was selected and why.
- Tiny workloads do not accidentally pay GPU compile/enqueue overhead.

### Go/no-go decision

Do not make automatic backend selection too clever before Phase 8 study workflows exist. First expose enough metadata and manual control.

---

## 14. Keep correctness validation ahead of performance claims

### Goal

Make every optimization provably equivalent to the current implementation for small deterministic cases.

### Required correctness tests

For each new solver or kernel path, compare against the existing path using:

```text
small Nx
small B
short Nt
full recording
fixed seed or deterministic input
float32 and float64 if supported
```

Suggested cases:

```text
Nx = 3, 8, 32, 51
B  = 1, 2, 5, 32
Nt = 2, 10, 100
```

### Signals to compare

Compare as many as are internally available:

```text
Vm
Vi
Ve
activation observer outputs
first crossing times
max Vm
final state
```

### Numerical tolerances

Use tolerances appropriate to dtype:

```text
float64: stricter
float32: realistic but not loose
```

Also check:

```text
no NaNs
no infs
same boundary behavior
same response for zero stimulus
same response for constant stimulus
```

### Acceptance criteria

No performance optimization should be considered complete until it has:

- unit tests,
- at least one benchmark row,
- correctness comparison to the old path,
- benchmark metadata documenting the selected path.

---

## 15. Recommended execution order

This is the most important part of the plan. Execute in this order.

### Step 1 — Add the canonical benchmark matrix

Do this first.

Deliverable:

```text
small_nx_large_batch_double_cable_gpu benchmark family
```

Success condition:

```text
Can generate CPU/GPU tables for Nx=32/51/64/96 and B=512/1024/2048/4096.
```

---

### Step 2 — Re-run the current baseline

Do not change solver code yet.

Deliverable:

```text
baseline_current_double_cable_small_nx_large_batch.md or benchmark artifact
```

Success condition:

```text
You know whether solver-only, input preparation, output retention, or result packaging dominates.
```

---

### Step 3 — Verify zero-Iinj stays sparse/absent

Deliverable:

```text
No dense zero Iinj[B,Nt,Nx] for double-cable no-Iinj runs.
```

Success condition:

```text
Benchmark metadata proves skipped dense zero input.
```

---

### Step 4 — Unblock double-cable observer-only batching

Deliverable:

```text
Double-cable activation observer works in batched mode for homogeneous unpadded groups.
```

Success condition:

```text
No retained Vm[B,Nt,Nx] is needed for activation/recruitment-style outputs.
```

---

### Step 5 — Implement solver-only explicit batched block-Thomas

Deliverable:

```text
solve_block_tridiagonal_2x2_batched(... rhs[B,Nx] ...) -> x[B,Nx]
```

Success condition:

```text
Matches current solver and improves or clarifies GPU solver-only performance.
```

---

### Step 6 — Integrate batched block-Thomas into the production double-cable kernel

Deliverable:

```text
Feature-flagged production path for compatible double-cable groups.
```

Success condition:

```text
End-to-end double-cable GPU timing improves for B>=512 without correctness regression.
```

---

### Step 7 — Test layout and small static Nx families

Deliverable:

```text
Benchmark comparison for [B,Nx] vs [Nx,B], plus optional Nx_pad=32/64/128.
```

Success condition:

```text
Chosen layout is evidence-based.
```

---

### Step 8 — Prototype factorized extracellular drive

Deliverable:

```text
Internal path using waveform[Nt] * footprint[B,Nx], no dense Vext[B,Nt,Nx].
```

Success condition:

```text
Dense and factorized paths match numerically; factorized path reduces memory and/or time.
```

---

### Step 9 — Batch amplitudes for threshold/recruitment

Deliverable:

```text
Activation observer output shaped [B,A] for A amplitude candidates.
```

Success condition:

```text
B_effective = B*A improves GPU utilization and reduces per-amplitude cost.
```

---

### Step 10 — Profile before custom kernels

Deliverable:

```text
Short profiling report: kernel count, fusion, layout, memory, compile time.
```

Success condition:

```text
You know whether Pallas/Triton is justified.
```

---

### Step 11 — Prototype Pallas/Triton only if justified

Deliverable:

```text
Solver-only custom kernel for Nx_pad=32/64/128 and B large.
```

Success condition:

```text
At least 25% solver-only speedup or 15% end-to-end speedup.
```

---

### Step 12 — Test PCR/cyclic reduction last

Deliverable:

```text
Solver-only block-PCR benchmark against explicit batched Thomas.
```

Success condition:

```text
PCR wins in the actual Nx=30–100, B>500 regime, not only for larger Nx.
```

---

## 16. Decision table

Use this table to decide what to do after each benchmark round.

| Observation | Interpretation | Next action |
| --- | --- | --- |
| Full recording is slow, observer-only is fast | Output retention dominates | Prioritize observers and compact study results |
| Solver-only is slow on GPU | Solver structure/layout dominates | Implement explicit batched Thomas |
| Batched Thomas is much faster than current | Current layout/vmap path was limiting | Integrate batched Thomas into production |
| Batched Thomas is not faster | XLA may already optimize current path | Profile HLO/layout before custom kernels |
| Factorized drive improves total time | Input materialization dominates | Prioritize Phase 8 drive reuse/factorized API |
| Amplitude batching improves throughput | GPU needs larger effective batch | Design threshold/recruitment around `B_effective` |
| PCR wins only at Nx>=128 | Not useful for current target | Defer PCR |
| CPU still wins at B=512 after optimizations | GPU threshold is higher than desired | Use backend selection and increase `B_effective` |
| GPU wins strongly at B>=1024 | Batch axis is sufficient | Avoid high-maintenance custom kernels |

---

## 17. Expected likely outcome

The most likely winning path for `Nx = 30–100` and `B > 500` is:

```text
1. No dense zero Iinj
2. Compact observer-only output
3. Explicit batched block-Thomas solver
4. Layout choice based on [B,Nx] vs [Nx,B] benchmarks
5. Factorized extracellular drive
6. Amplitude/condition batching to increase B_effective
```

PCR or cyclic reduction may still be useful, but it is not the first bet for this workload. It is more likely to matter for longer cables or if profiling proves that the sequential `Nx` scan remains the dominant issue after the batch-oriented changes.

---

## 18. Practical target performance milestones

These are not promises; they are useful milestone targets.

### Milestone 1

For:

```text
Nx = 51
B = 512
Nt = 1000
recording = none or activation observer
Iinj = None
Vext = dense
```

Target:

```text
GPU should be clearly competitive with CPU, not just near crossover.
```

### Milestone 2

For:

```text
Nx = 51
B = 1024–2048
Nt = 1000
recording = none or activation observer
Iinj = None
Vext = dense or factorized
```

Target:

```text
GPU should clearly win end-to-end.
```

### Milestone 3

For threshold/recruitment:

```text
B = 512
A = 8 amplitude candidates
B_effective = 4096
Nx = 51
```

Target:

```text
One batched multi-amplitude call should be much cheaper than A separate calls.
```

---

## 19. Minimal implementation checklist

If time is limited, do only this subset first:

- [ ] Add the canonical benchmark matrix.
- [ ] Re-run current baseline for `Nx=32/51/96`, `B=512/1024/2048`.
- [ ] Confirm no dense zero `Iinj` is passed for double-cable no-Iinj runs.
- [ ] Add double-cable activation observer in batched mode for homogeneous unpadded groups.
- [ ] Implement `solve_block_tridiagonal_2x2_batched` solver-only.
- [ ] Compare `[B,Nx]` and `[Nx,B]` layouts.
- [ ] Integrate the winner behind a private feature flag.
- [ ] Benchmark again.

Only after that:

- [ ] Prototype factorized drive.
- [ ] Batch amplitudes.
- [ ] Profile HLO/kernel layout.
- [ ] Consider Pallas/Triton.
- [ ] Consider PCR.

---

## 20. Final recommendation

For the stated workload, do not start with a broad solver rewrite.

Start by making the current double-cable path behave like a specialized small-`Nx`, large-`B` GPU workload:

```text
B is the main parallel axis.
Nx is small and possibly padded/specialized.
No dense zero inputs are allowed.
No full traces are retained unless requested.
Extracellular drive should eventually be factorized.
Study dimensions should be batched to increase B_effective.
```

Only move to PCR/cyclic reduction or custom GPU kernels if the benchmark matrix proves that the simpler batched approach is still insufficient.
