# AxonScope Double-Cable GPU Scaling Test Plan

**Audience:** AxonScope maintainers working on GPU performance for single-cable and double-cable axon pool simulations.  
**Branch reviewed:** `louisreg/AxonScope`, branch `bench-colab`.  
**Date:** 2026-06-15.  
**Goal:** define a detailed, reproducible, evidence-driven test plan for improving double-cable GPU scaling, especially the solver path.

---

## 0. Executive Summary

The current evidence strongly suggests that the double-cable GPU scaling issue is not primarily caused by generic extracellular preprocessing, public result packaging, or dense stimulation lowering. Those areas still matter, but the main double-cable bottleneck appears to be the time-step kernel itself, specifically the nested structure:

```text
batch vmap over B
  time scan over Nt
    double-cable implicit step
      2x2 block-tridiagonal solve
        forward scan over Nx
        reverse scan over Nx
```

For the single-cable path, the implicit cable solve can use a standard tridiagonal formulation. JAX exposes `jax.lax.linalg.tridiagonal_solve`, whose diagonal inputs can be batched over leading dimensions and which has CPU/GPU support. For the double-cable path, AxonScope currently has a custom 2x2 block-tridiagonal scalar solver. That solver is compact and avoids materializing `(Nx, 2, 2)` block arrays inside the time loop, but it still performs two spatial dependency chains per time step: a forward elimination scan and a reverse substitution scan.

The top-level hypothesis to test is therefore:

> Double-cable GPU under-utilization is dominated by the sequential spatial dependency inside the per-time-step 2x2 block solver. The GPU only sees enough independent work through the batch axis, so it performs poorly for small and medium `B`, and only becomes attractive when `B` is large enough to amortize launch and dependency overhead.

The test plan below is organized into phases:

1. **Reproduce and lock the baseline** so all future measurements are comparable.
2. **Separate preprocessing, input materialization, result packaging, and solver throughput** using existing and new solver-only workloads.
3. **Measure the zero-Iinj kernel specialization** that has already been started in the branch.
4. **Stress the current block solver** with controlled `B`, `Nt`, `Nx`, dtype, and layout sweeps.
5. **Prototype alternative double-cable solvers**: block cyclic reduction / PCR, pentadiagonal Schur complement, small dense oracle, and custom-kernel routes.
6. **Test backend-selection policy** so tiny and medium double-cable workloads do not automatically use GPU when CPU is faster.
7. **Test factorized extracellular forcing** as a Phase 8-style prototype without committing the public API too early.
8. **Add double-cable solver-side observers** so threshold/recruitment studies can avoid retaining full Vm traces.
9. **Define acceptance criteria** for merging, deferring, or dropping each candidate.

The most promising improvement path is not one large rewrite immediately. It is a staged sequence:

```text
A. Keep the current solver as correctness baseline.
B. Finish and benchmark the no-Iinj specialization.
C. Add a solver-only microbenchmark matrix around the 2x2 block solve.
D. Prototype block-PCR or cyclic reduction for Nx-parallelism.
E. Compare against CPU/SciPy/JAX-CPU backend policy for small workloads.
F. Only then consider Pallas/FFI/custom CUDA if JAX-level PCR is promising but not fast enough.
```

---

## 1. Current Evidence From the Branch

The current branch already contains useful benchmark evidence and implementation hints. This section should be treated as the starting point for the next experiments, not as speculation.

### 1.1 Benchmark evidence already recorded

The roadmap in `todo.md` records these relevant points:

- A local warm path matrix at `n=100`, `duration=2 ms`, `dt=0.02 ms`, target `51` compartments measured:
  - single-cable intracellular: `89.9 ms`
  - single-cable point-source extracellular: `76.0 ms`
  - MRG double-cable extracellular: `176.0 ms`
- A Colab double-cable extracellular long run, labelled `colab_cpu_gpu_kernel_double_cable_extracellular_long_20260615_175837`, reported:
  - `n=100`: GPU loses, `824.9 ms` GPU vs `465.1 ms` CPU
  - `n=300`: GPU crosses over, `954.8 ms` GPU vs `1165.8 ms` CPU
  - `n=600`: GPU wins, `1070.1 ms` GPU vs `2485.8 ms` CPU
- The same evidence ledger states that dense zero `Iinj` was skipped in memory estimates at `n=600`, saving a reported `108 MB` of skipped dense input.
- A matching single-cable extracellular long run scaled much better on GPU, with CPU/GPU speedups of `5.43x`, `6.81x`, and `9.78x` at `n=100/300/600`.
- The TODO explicitly identifies the first double-cable GPU kernel issue as the `Nt` scan plus per-step forward/reverse `Nx` scans inside `solve_block_tridiagonal_2x2_scalar`, giving the GPU mostly batch-axis parallelism.
- The TODO records a local smoke test after keeping absent double-cable intracellular input as `None` through the kernel path instead of passing a dense device zero `Iinj[B,Nt,Nx]`, but says a Colab double-cable rerun is still needed to measure GPU impact.

### 1.2 Code-level evidence already visible

The relevant functions and structures are:

- `src/axonscope/solvers/common.py`
  - `solve_block_tridiagonal_2x2_scalar(...)`
  - uses a forward `jax.lax.scan(...)` to compute modified coefficients
  - uses a reverse `jax.lax.scan(...)` for back-substitution
- `src/axonscope/solvers/batch_kernels.py`
  - `_run_double_cable_batch_vm_scan(...)`
  - wraps one double-cable run in a `jax.vmap(...)` over batch rows
  - in one visible path, if `Iinj_mid is None`, it fills `Iinj_mid = jnp.zeros_like(vext_mid)` before calling `_run_double_cable_vm_scan(...)`
  - later stateful/chunked paths have already started preserving `None` for absent `Iinj`
- `src/axonscope/dispatcher/execution.py`
  - `_can_run_batch_group(...)` currently rejects double-cable batch execution when solver-side observers are present: `if observers and group.mode == "double": return False`
- `src/axonscope/performance.py`
  - `Runtime`, `Device`, and `PrecisionPolicy` exist in the estimate/planning layer
  - `estimate_simulation(...)` tracks skipped dense `Iinj` metadata when no intracellular contexts are present

### 1.3 Interpretation

The current evidence points to the following likely root causes, ranked by expected impact:

| Rank | Hypothesis | Why it matters | Expected impact if fixed |
|---:|---|---|---|
| 1 | Per-time-step block-tridiagonal solve has sequential spatial scans | GPU parallelism is mostly `B`, not `Nx`; poor utilization until large `B` | High |
| 2 | Backend choice is not calibrated for double-cable small/medium workloads | CPU beats GPU at `n=100`; GPU only crosses over later | High for real workflows with threshold loops |
| 3 | Absent `Iinj` still creates device-side zero arrays in some full-scan paths | Extra memory bandwidth and shape pressure | Low to medium, easy win |
| 4 | Dense extracellular forcing still materializes `B x Nt x Nx` | Memory pressure for long runs and sweeps | Medium, especially Phase 8 studies |
| 5 | Double-cable observer path is not batched | Trace-free threshold/recruitment workflows fall back to scalar or retain too much | Medium to high for study workloads |
| 6 | Layout and dtype may be suboptimal | Scan memory access and register pressure can matter | Medium, must measure |
| 7 | Current benchmark output may not expose enough kernel internals | We need better attribution before rewriting solver | Indirect but important |

---

## 2. Ground Rules for All Performance Tests

Before optimizing, make the measurements boring and reproducible. Every experiment below should follow these rules.

### 2.1 Always separate cold and warm measurements

JAX compilation can dominate the first run. Every benchmark should record at least:

- `cold_total_ms`
- `compile_or_first_call_ms`, if separable
- `warmup_count`
- `warm_total_ms_median`
- `warm_total_ms_p50`, `p90`, `p95`, ideally `min/max`
- `kernel_enqueue_ms`
- `kernel_wait_ms`
- `input_materialization_ms`
- `runtime_prepare_ms`
- `result_packaging_ms`

The branch already has cold/warm annotations in benchmark outputs. Preserve and expand that rather than starting a separate benchmark style.

### 2.2 Always force synchronization for GPU timing

JAX dispatch is asynchronous. Timing just the Python call can measure enqueue latency rather than actual device execution. For GPU timing, every timed kernel output must call `.block_until_ready()` on a representative returned array, or the benchmark harness must explicitly wait on all returned device arrays.

Recommended pattern:

```python
start = time.perf_counter()
out = compiled_fn(*args)
# Choose a representative output that depends on all work.
jax.tree_util.tree_map(
    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
    out,
)
elapsed_ms = (time.perf_counter() - start) * 1000.0
```

For AxonScope’s existing hotpath harness, keep separate spans for enqueue and wait if possible:

```text
kernel.enqueue
kernel.wait
```

### 2.3 Record exact shapes and policies

Every benchmark row must include:

- `B`: axon count / batch size
- `Nt`: number of time steps
- `Nx`: compartments after padding / target compartments
- `formulation`: `single` or `double`
- `axon_template`: e.g. HH, MRG-like, exact constructor if possible
- `stimulus_kind`: intracellular, point-source extracellular, typed drive, zero-field
- `recording_policy`: full, center, probes, none
- `observers`: none, activation, peak, etc.
- `dtype`: `float32` or `float64`
- `backend`: JAX CPU, JAX GPU, SciPy/NumPy prototype, etc.
- `device_name`: from JAX device metadata or environment capture
- `jax/jaxlib` versions
- CUDA/cuDNN versions if available
- whether persistent compilation cache is enabled
- whether X64 is enabled
- whether `Iinj` is dense, sparse, or absent/None
- whether `Vstim` is dense or factorized
- whether the run is solver-only, public simulation, or full pool dispatch

### 2.4 Correctness gates come before speed gates

For every candidate solver or path, compare against the current implementation.

Minimum correctness checks:

```text
max_abs_delta_Vm_mV
max_rel_delta_Vm
activation_status_delta_count
first_spike_time_delta_ms, if activation is available
peak_voltage_delta_mV, if PeakVoltage observer is available
```

Recommended tolerances:

| Mode | Suggested tolerance |
|---|---:|
| Same algorithm refactor, same dtype | `max_abs_delta <= 1e-5 mV` for float32, tighter for float64 |
| Different linear solver, float32 | `max_abs_delta <= 1e-4 mV` initially, then investigate |
| Observer status | exact match required for non-borderline stimuli |
| Threshold/recruitment | compare away from threshold boundary first |

For threshold-like tests, avoid using stimuli exactly on the activation boundary because tiny numerical differences can flip activation status.

### 2.5 Do not benchmark only one size

The current evidence already shows that double-cable has a CPU/GPU crossover. A single benchmark at one `B` can be misleading. Every meaningful run should include at least:

```text
B = 1, 5, 20, 50, 100, 300, 600, 1000
Nt = 100, 500, 1000, 2000
Nx = 21, 51, 101, 201
```

The full Cartesian product may be too large. Use a staged matrix:

- smoke: `B={5, 100}`, `Nt={100}`, `Nx={51}`
- local warm: `B={5, 20, 100}`, `Nt={100, 500}`, `Nx={51}`
- Colab/GPU: `B={100, 300, 600, 1000}`, `Nt={1000}`, `Nx={51}`
- solver-stress: `B={100, 600}`, `Nt={1000}`, `Nx={21, 51, 101, 201}`

---

## 3. Baseline Reproduction Tests

### 3.1 Test A0 — Environment capture

**Purpose:** Make all performance numbers interpretable.

**Run before any benchmark session.**

Capture:

```bash
python benchmark/runtime/environment_info_demo.py
python - <<'PY'
import jax, platform, sys
print('python', sys.version)
print('platform', platform.platform())
print('jax', jax.__version__)
try:
    import jaxlib
    print('jaxlib', jaxlib.__version__)
except Exception as exc:
    print('jaxlib unavailable', exc)
print('devices', jax.devices())
print('default backend', jax.default_backend())
print('x64 enabled', jax.config.jax_enable_x64)
PY
nvidia-smi || true
```

Store this in the benchmark manifest. Do not compare GPU runs across machines unless the device and JAX stack are recorded.

**Acceptance:** Every result folder has a manifest with versions and device metadata.

---

### 3.2 Test A1 — Re-run the current public hotpath matrix

**Purpose:** Confirm the branch still reproduces the evidence before optimizing.

Run:

```bash
python benchmark/hotpaths/run.py --list
python benchmark/hotpaths/run.py \
  --workload path_comparison_matrix \
  --preset smoke \
  --warmup-count 1 \
  --repeat-count 3
```

Then run a local warm version that includes:

- single-cable intracellular
- single-cable point-source extracellular
- double-cable/MRG-like point-source extracellular
- recording: center, probes, full, none
- observer off/on where supported

**Record:**

```text
root total
runtime.prepare
inputs.intracellular
inputs.extracellular
kernel.enqueue
kernel.wait
results.split_batch / result packaging
```

**Acceptance:** Results should be directionally consistent with the current evidence ledger. If not, stop and explain the regression before adding new optimizations.

---

### 3.3 Test A2 — Re-run matching single-cable and double-cable long GPU cases

**Purpose:** Keep the central comparison visible: single-cable scales well, double-cable scales weakly.

Run the existing long cases if present:

```bash
python benchmark/hotpaths/run.py \
  --workload kernel_single_cable_extracellular_long \
  --preset colab \
  --warmup-count 1 \
  --repeat-count 3

python benchmark/hotpaths/run.py \
  --workload kernel_double_cable_extracellular_long \
  --preset colab \
  --warmup-count 1 \
  --repeat-count 3
```

If the runner uses a different naming convention, use the workload names listed by:

```bash
python benchmark/hotpaths/run.py --list
```

**Matrix:**

```text
B = 100, 300, 600
Nt = 1000
Nx = 51
recording = same policy for both paths
stimulus = point-source extracellular / typed equivalent
Iinj = absent
```

**Acceptance:** The qualitative relationship should remain:

- single-cable GPU speedup is strong already at `B=100`
- double-cable GPU loses or is weak at `B=100`
- double-cable GPU crosses over later

If the zero-Iinj specialization already changed this, keep both old and new rows in the report.

---

## 4. Solver-Only Isolation Tests

The most important next step is to measure the solver directly, not the whole public simulation stack.

### 4.1 Test B0 — Solver-only precomputed-input baseline

**Purpose:** Isolate kernel throughput from dispatch, runtime preparation, input lowering, and result packaging.

Use or extend the existing `solver_only_precomputed` workload.

Recommended matrix:

```text
formulation = single-cable, double-cable
B = 1, 5, 20, 50, 100, 300, 600, 1000
Nt = 100, 500, 1000
Nx = 51
recording = full and none/observer if available
stimulus = precomputed dense extracellular drive
Iinj = absent/None
backend = JAX CPU, JAX GPU
```

**Record:**

```text
compile time
warm kernel enqueue
warm kernel wait
output bytes
input bytes
device memory peak if available
```

**Expected result:** If preprocessing is not the issue, solver-only double-cable should still show the weak GPU scaling pattern.

**Decision rule:**

- If solver-only reproduces the same double-cable GPU weakness, prioritize solver rewrite/prototype.
- If solver-only is fast but full simulation is slow, reprioritize input lowering, dispatch, or packaging.

---

### 4.2 Test B1 — Direct block solver microbenchmark

**Purpose:** Measure `solve_block_tridiagonal_2x2_scalar(...)` in isolation.

Create a benchmark that constructs realistic coefficient arrays, then applies the solver in a batched and time-looped way without membrane state updates.

Pseudo-code:

```python
import jax
import jax.numpy as jnp
from axonscope.solvers.common import solve_block_tridiagonal_2x2_scalar

@jax.jit
def block_solver_only(a00, a01, a10, a11, off0, off1, rhs0, rhs1):
    def one_row(inputs):
        a00_r, a01_r, a10_r, a11_r, off0_r, off1_r, rhs0_r, rhs1_r = inputs
        return solve_block_tridiagonal_2x2_scalar(
            a00_r, a01_r, a10_r, a11_r, off0_r, off1_r, rhs0_r, rhs1_r
        )
    return jax.vmap(one_row)((a00, a01, a10, a11, off0, off1, rhs0, rhs1))
```

Then wrap it in a time scan to mimic `Nt` repeated solves:

```python
@jax.jit
def repeated_block_solver(coeffs, rhs0_init, rhs1_init):
    def step(carry, _):
        rhs0, rhs1 = carry
        x0, x1 = block_solver_only(*coeffs, rhs0, rhs1)
        return (x0, x1), None
    final, _ = jax.lax.scan(step, (rhs0_init, rhs1_init), xs=None, length=Nt)
    return final
```

Use coefficient ranges from actual double-cable runs when possible. Add one synthetic diagonally dominant case for numerical safety.

**Matrix:**

```text
B = 1, 5, 20, 50, 100, 300, 600, 1000, 2000
Nx = 21, 51, 101, 201
Nt = 1, 10, 100, 1000
```

**Record:**

```text
ms per solve = total_ms / Nt
ms per row solve = total_ms / (Nt * B)
effective solves/s
kernel wait fraction
```

**Acceptance:** This should reveal whether the block solver alone has the same crossover behavior as the full double-cable kernel.

---

### 4.3 Test B2 — Compare current scalar block representation vs materialized block representation

**Purpose:** Check whether the current component-wise solver is faster than a clearer materialized `(Nx, 2, 2)` block solver.

The current function avoids materializing block arrays, which is good for memory. But a materialized version may allow XLA to produce a different program. This test is not meant as a final implementation; it is a compiler behavior probe.

Implement a second solver:

```text
A_diag:  (Nx, 2, 2)
A_lower: (Nx, 2, 2)
A_upper: (Nx, 2, 2)
rhs:     (Nx, 2)
```

Then run a block Thomas algorithm using 2x2 matrices, similar to the older or generic code path visible near the current solver.

**Compare:**

```text
current component solver
materialized block solver
```

**Matrix:**

```text
B = 100, 600
Nt = 100, 1000
Nx = 51, 101
backend = CPU, GPU
```

**Expected:** The current component solver should probably win on memory and scalar arithmetic. If the materialized version wins on GPU, inspect the lowered HLO because the compiler may be vectorizing the block operations more effectively.

**Decision rule:**

- If materialized block is slower, keep current representation as baseline.
- If materialized block is faster on GPU by >10%, investigate why before writing PCR.

---

### 4.4 Test B3 — Layout sweep: batch-major vs space-major

**Purpose:** Check whether data layout affects memory coalescing and scan behavior.

Current natural shapes are likely batch-major:

```text
B x Nt x Nx
B x Nx
```

Try solver-only layouts:

```text
layout A: B x Nx
layout B: Nx x B
layout C: B grouped into tiles, tile_B x Nx
```

For the block solver, a scan over `Nx` with each scan element carrying vectors of shape `B` may expose more parallel work inside each scan step than `vmap` over rows where each row has a scalar scan. In other words, test these two equivalent structures:

```text
Option 1: vmap over B, scan over Nx inside each row
Option 2: scan over Nx, vectorized arithmetic over B inside each scan step
```

Pseudo-structure for option 2:

```python
def solve_block_tridiagonal_2x2_batched_space_scan(...):
    # inputs shaped (B, Nx) or transposed to (Nx, B)
    # scan over i in Nx, carry arrays shaped (B,)
    # arithmetic at each step is vectorized over B
```

This does not remove the sequential `Nx` dependency, but it may improve GPU occupancy because each scan step operates over all `B` rows at once.

**Matrix:**

```text
B = 20, 100, 300, 600, 1000
Nx = 51, 101
Nt = 100, 1000
```

**Acceptance:**

- If space-scan/vectorized-B is faster by >15% without correctness changes, it is a near-term candidate.
- If it is neutral or slower, document and move on to PCR.

---

## 5. Zero-Iinj Specialization Tests

The branch already appears to have partially implemented this optimization. The remaining step is to measure it carefully.

### 5.1 Background

For extracellular-only stimulation, there is no intracellular current input. The ideal kernel should carry that fact as `None` or a static zero policy and should not materialize or read `Iinj[B,Nt,Nx]`.

The current full double-cable visible path contains this pattern:

```python
if Iinj_mid is None:
    Iinj_mid = jnp.zeros_like(vext_mid)
```

That avoids host-side dense input in some paths but still creates a device-side dense zero array inside the jitted computation. The branch later shows a stateful/chunked path that keeps `intracellular_current_abs_mid = None` when input is absent. Test whether the fully retained-output path has also been specialized.

### 5.2 Test C0 — Colab rerun after no-Iinj specialization

**Purpose:** Measure the real GPU impact of preserving absent `Iinj` through the double-cable kernel path.

Run exactly the same Colab workload as the evidence ledger:

```text
workload = kernel_double_cable_extracellular_long
B = 100, 300, 600
Nt = 1000
Nx = 51
Iinj = absent
recording = same as previous evidence
```

**Compare:**

```text
before specialization, if historical result available
current branch after specialization
```

**Record:**

```text
kernel enqueue/wait
input materialization
static memory estimate skipped_dense_iinj_nbytes
peak device memory if available
compile time
```

**Expected:**

- Memory pressure should improve.
- Runtime may improve modestly if the previous kernel read/constructed zeros repeatedly.
- Do not expect this alone to turn double-cable into single-cable-like scaling, because the solver dependency remains.

**Decision rule:**

| Result | Action |
|---|---|
| >10% GPU speedup | Keep and add regression benchmark |
| 2–10% speedup | Keep as cleanup/easy win |
| <2% speedup but memory lower | Keep if code remains simpler or memory metadata improves |
| Slower | Inspect HLO and shape specialization; consider separate static functions for `has_iinj=False` |

---

### 5.3 Test C1 — Static branch vs dynamic optional input

**Purpose:** Check whether passing `None` through a jitted function produces the best compiled program.

Test three implementations:

```text
Variant A: current optional None path
Variant B: separate jitted function for no-Iinj
Variant C: dense zero input path, old behavior
```

The no-Iinj function should have no `intracellular_current_density_mid` argument at all:

```python
@jax.jit
def _run_double_cable_batch_vm_scan_no_iinj(...):
    # No Iinj argument.
    # Inside each step, use scalar zero or omit term entirely.
```

**Why this matters:** A separate function can reduce HLO size and eliminate shape-dependent optional logic. It also makes benchmark interpretation easier.

**Acceptance:**

- If Variant B is faster or compiles to smaller HLO, keep separate no-Iinj kernels.
- If Variant A is equal, keep the simpler optional path.

---

## 6. Current Solver Stress Tests

Before implementing a new solver, characterize the current one thoroughly.

### 6.1 Test D0 — Complexity scaling with Nx

**Purpose:** Confirm the spatial scan is the problem.

Run solver-only double-cable with:

```text
B = 600
Nt = 1000
Nx = 11, 21, 51, 101, 201, 401
backend = JAX CPU, JAX GPU
```

Plot:

```text
runtime_ms vs Nx
runtime_ms / Nx vs Nx
GPU speedup vs Nx
```

**Expected:** Current block Thomas should scale approximately linearly with `Nx` per time step. GPU speedup may worsen for small `B` and larger `Nx` if spatial dependency dominates.

**Decision rule:**

- If runtime scales linearly with `Nx`, PCR/cyclic reduction becomes more attractive.
- If runtime is flat or dominated elsewhere, investigate memory or launch overhead first.

---

### 6.2 Test D1 — Complexity scaling with B

**Purpose:** Find GPU saturation threshold for the current solver.

Run:

```text
B = 1, 2, 5, 10, 20, 50, 100, 200, 300, 600, 1000, 2000, 5000
Nt = 1000
Nx = 51
backend = CPU, GPU
```

Plot:

```text
runtime_ms vs B
throughput = B * Nt / runtime_ms
speedup = CPU_ms / GPU_ms
```

**Expected:** GPU throughput should increase with `B` until saturation. The current evidence suggests the double-cable crossover is somewhere between `B=100` and `B=300` on the tested Colab GPU.

**Acceptance:** This curve becomes the default backend-selection calibration curve.

---

### 6.3 Test D2 — Complexity scaling with Nt

**Purpose:** Distinguish launch overhead from per-step cost.

Run:

```text
B = 100, 600
Nx = 51
Nt = 10, 50, 100, 500, 1000, 2000, 5000
backend = CPU, GPU
```

Plot:

```text
runtime_ms vs Nt
runtime_ms / Nt
```

**Expected:** If launch overhead is significant, short `Nt` will be especially bad on GPU. If per-step solver dominates, runtime should scale linearly with `Nt` after warmup.

---

### 6.4 Test D3 — Dtype sweep

**Purpose:** Check whether float64 or accidental X64 settings are hurting GPU performance.

Run the same solver-only matrix with:

```text
dtype = float32, float64
jax_enable_x64 = false, true where appropriate
```

For scientific correctness, float64 may be useful, but for GPU throughput float32 is usually much faster on many consumer/Colab GPUs. Do not assume the answer: measure both.

**Acceptance:**

- If float32 passes correctness and is much faster, make float32 the default for GPU benchmarks.
- If float64 is required for stability in double-cable, isolate why and document the numerical condition.

---

### 6.5 Test D4 — HLO / compiler inspection

**Purpose:** See whether XLA is generating the expected structure.

For a representative solver-only function, dump HLO/MLIR:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/axonscope_xla --xla_dump_hlo_as_text" \
python benchmark/hotpaths/run.py \
  --workload solver_only_precomputed \
  --preset smoke
```

Inspect:

```text
number of while loops / scans
fusion boundaries
large constant materialization
whether zeros are actually eliminated
shape of carried arrays
unrolled vs while-loop scan lowering
```

**Questions to answer:**

- Does the no-Iinj path remove zero buffers from HLO?
- Does `vmap(scan(...))` lower differently from `scan(vmap(...))`?
- Are there excessive transposes or broadcasts?
- Are time and space scans nested in the expected order?

**Acceptance:** Summarize HLO findings in a benchmark note. Do not optimize blindly based only on wall time.

---

## 7. Alternative Solver Prototypes

This is the core optimization section. The point is not to immediately replace the production solver. The point is to create small, isolated candidates and compare them fairly.

### 7.1 Candidate E0 — Keep current block Thomas as baseline

**Description:** Current `solve_block_tridiagonal_2x2_scalar(...)` with forward and reverse scans.

**Role:** Correctness baseline and fallback.

**Strengths:**

- Simple and compact.
- Numerically familiar.
- Avoids materializing full block matrices.
- Already integrated.

**Weaknesses:**

- Sequential dependency in `Nx`.
- Nested under `Nt` scan.
- GPU parallelism mostly comes from `B`.

**Action:** Do not delete or rewrite this first. Keep it as a reference solver while testing all alternatives.

---

### 7.2 Candidate E1 — Batched-space Thomas: scan over Nx, vectorize over B

**Description:** Rewrite the current solver so that one solver call handles the full batch at once. Instead of:

```text
vmap over B:
  scan over Nx with scalar carry
```

use:

```text
scan over Nx:
  carry arrays of shape (B,)
  vectorized arithmetic over B at each spatial index
```

**Why test it:** This does not change algorithmic depth, but it may give the GPU larger vector operations inside each scan step. It is simpler than PCR and can be compared quickly.

**Implementation notes:**

- Inputs can remain `(B, Nx)` but transpose internally to `(Nx, B)` if that improves scan input layout.
- Carry:

```text
c00_prev: (B,)
c01_prev: (B,)
c10_prev: (B,)
c11_prev: (B,)
d0_prev:  (B,)
d1_prev:  (B,)
```

- Forward outputs:

```text
c00, c01, c10, c11, d0, d1: (Nx, B)
```

- Reverse scan similarly carries `(B,)` arrays.
- Transpose output back to `(B, Nx)`.

**Benchmark matrix:**

```text
B = 20, 100, 300, 600, 1000
Nx = 51, 101
Nt = 100, 1000
backend = CPU, GPU
```

**Expected:** Potential moderate speedup, especially at `B=100/300`. It probably will not solve the fundamental sequential-depth problem.

**Acceptance:**

- Keep if GPU speedup >15% and CPU is not much worse.
- If CPU worsens but GPU improves, keep as GPU-specific path selected by backend policy.

---

### 7.3 Candidate E2 — Parallel cyclic reduction / PCR for 2x2 block tridiagonal systems

**Description:** Replace Thomas forward/back substitution with a parallel reduction algorithm over the spatial dimension. Cyclic reduction and parallel cyclic reduction reduce the dependency depth from roughly `O(Nx)` sequential steps to roughly `O(log Nx)` reduction stages, at the cost of more arithmetic and more temporary storage.

**Why test it:** `Nx≈51` is small but not tiny. Current double-cable repeats the spatial solve for every `Nt` and every batch row. Reducing spatial dependency depth may improve GPU utilization even if arithmetic increases.

**High-level idea:**

For a block-tridiagonal system:

```text
L_i x_{i-1} + D_i x_i + U_i x_{i+1} = r_i
```

eliminate odd or even nodes in parallel at each stage. For each remaining node, update:

```text
D'_i = D_i - L_i inv(D_{i-1}) U_{i-1} - U_i inv(D_{i+1}) L_{i+1}
r'_i = r_i - L_i inv(D_{i-1}) r_{i-1} - U_i inv(D_{i+1}) r_{i+1}
```

For the AxonScope double-cable structure, each `D_i` is 2x2 and off-diagonal blocks are diagonal, so the per-stage 2x2 inversions can be explicit.

**Prototype scope:**

- Implement only solver-only first.
- Support fixed `Nx` known at compile time.
- Support power-of-two padded sizes internally if that simplifies the algorithm.
- Mask padded spatial nodes carefully.
- Compare to current solver on random diagonally dominant systems and real double-cable coefficients.

**Pseudo-interface:**

```python
def solve_block_tridiagonal_2x2_pcr(
    a00, a01, a10, a11,
    off0, off1,
    rhs0, rhs1,
    *,
    mask=None,
):
    """Return x0, x1 for one row or batched rows."""
```

Prefer designing it batched from the start:

```text
a00:  (B, Nx)
a01:  (B, Nx)
a10:  (B, Nx)
a11:  (B, Nx)
off0: (B, Nx-1) or (Nx-1,)
off1: (B, Nx-1) or (Nx-1,)
rhs0: (B, Nx)
rhs1: (B, Nx)
```

**Correctness tests:**

1. Synthetic diagonally dominant systems.
2. Real coefficient snapshots captured from double-cable runs.
3. Full one-step comparison.
4. Full `Nt` simulation comparison.

**Benchmark matrix:**

```text
B = 20, 100, 300, 600, 1000
Nx = 21, 51, 101, 201
Nt = 100, 1000
backend = CPU, GPU
```

**Expected:**

- For small `Nx=21`, PCR may lose due to extra arithmetic.
- For `Nx=51`, it might become competitive on GPU if implemented cleanly.
- For larger `Nx=101/201`, PCR should become more attractive.

**Acceptance:**

| Outcome | Decision |
|---|---|
| PCR wins >20% on GPU at `Nx=51`, `B<=300` | Strong candidate for production GPU path |
| PCR only wins at `Nx>=101` | Keep as optional path for larger compartment counts |
| PCR loses everywhere | Drop or revisit with Pallas/custom kernel only if profiling says JAX implementation is poor |
| PCR has correctness drift | Keep as research branch until numerical issue is understood |

---

### 7.4 Candidate E3 — Hybrid Thomas/PCR threshold policy

**Description:** Even if PCR wins only for certain sizes, a hybrid solver can choose:

```text
Thomas for small Nx or CPU
PCR for larger Nx or GPU
```

**Policy dimensions:**

```text
backend: CPU/GPU
B: batch size
Nx: spatial compartments
Nt: time steps
recording/observer mode
```

**Initial policy candidates:**

```text
CPU: current Thomas
GPU, Nx <= 32: current Thomas or batched-space Thomas
GPU, Nx >= 51 and B <= 300: PCR if benchmark confirms
GPU, B >= 600: whichever is faster in measured matrix
```

**Acceptance:** Use measured crossover tables, not intuition.

---

### 7.5 Candidate E4 — Schur complement / pentadiagonal scalar solve

**Description:** Eliminate one of the two unknown fields, e.g. `Ve`, to obtain a scalar system for `Vi` or `Vm`. Depending on algebra and discretization, this can lead to a wider-band scalar system, likely pentadiagonal.

**Why test it:** A scalar pentadiagonal solver might be easier to parallelize or map to existing banded solvers than a 2x2 block-tridiagonal solver. It may also expose more structure.

**Risks:**

- Algebra may be more complex.
- Numerical conditioning may worsen.
- A pentadiagonal Thomas-like solver still has sequential dependency unless using a parallel banded method.
- JAX does not expose a direct built-in pentadiagonal solve analogous to `tridiagonal_solve`.

**Prototype plan:**

1. Derive the exact discrete block equations for one time step.
2. Symbolically eliminate `Ve_i` or `Vi_i` for a small `Nx` case.
3. Build the equivalent full dense matrix for small `Nx` and verify equality.
4. Implement scalar pentadiagonal coefficients.
5. Solve using:
   - dense `jnp.linalg.solve` for oracle only
   - custom pentadiagonal Thomas for baseline
   - future PCR-like banded reduction if promising

**Benchmark matrix:**

```text
Nx = 21, 51
B = 100, 600
Nt = 100
```

**Acceptance:**

- Continue only if algebra is clean and correctness is excellent.
- Do not prioritize over block-PCR unless it shows an obvious speed or simplicity advantage.

---

### 7.6 Candidate E5 — Dense small-matrix oracle

**Description:** For `Nx=51`, the double-cable linear system has `2*Nx=102` unknowns per batch row per time step. A dense solve is not a production solution, but it is a useful oracle and sometimes a surprisingly good GPU baseline for very small systems.

**Implementation:**

- Materialize `A: (B, 2*Nx, 2*Nx)` for one step.
- Use `jax.vmap(jnp.linalg.solve)`.
- Only test small `Nt` first because dense materialization is expensive.

**Purpose:**

- Validate block solver correctness.
- Bound the performance of “throw it at GPU dense linear algebra.”
- Help catch algebra errors in PCR or Schur complement prototypes.

**Expected:** Usually too slow and memory-heavy for production.

**Acceptance:** Keep only as a correctness oracle and debugging tool unless it unexpectedly wins.

---

### 7.7 Candidate E6 — Pallas or FFI custom kernel

**Description:** If JAX-level PCR or batched-space Thomas reveals a promising algorithm but has poor generated code, implement the hot solver as a custom kernel.

**Do not start here.** First prove the algorithmic direction with JAX prototypes.

Possible routes:

```text
Pallas kernel for batched 2x2 block Thomas/PCR
JAX FFI to custom CUDA/C++ block solver
Triton-style kernel if integrated through JAX path
```

**Use custom kernel only if:**

- Solver-only microbenchmarks show the block solve dominates.
- A JAX prototype shows expected algorithmic improvement but generated HLO is poor.
- The target workloads are important enough to justify maintenance cost.

**Acceptance:**

- At least 1.5x speedup on representative GPU double-cable workloads.
- Correctness matches current solver within tolerance.
- CPU fallback remains current solver.
- Benchmark and CI skip behavior are clear when GPU/custom dependencies are absent.

---

## 8. Backend Selection Tests

The current benchmark evidence already says GPU is not always better for double-cable. Make that an explicit policy.

### 8.1 Test F0 — JAX CPU vs JAX GPU crossover table

**Purpose:** Build a backend decision table for double-cable workloads.

Matrix:

```text
B = 1, 5, 20, 50, 100, 200, 300, 600, 1000
Nt = 100, 500, 1000
Nx = 51, 101
recording = full, center, none if supported
stimulus = extracellular point-source or typed drive
```

For each row compute:

```text
speedup = CPU_ms / GPU_ms
winner = CPU or GPU
```

**Output:** A table like:

```text
formulation,double; Nx=51; Nt=1000; dtype=float32
B      CPU_ms    GPU_ms    speedup    recommended_device
100    ...       ...       ...        cpu
300    ...       ...       ...        gpu
600    ...       ...       ...        gpu
```

**Acceptance:** Add this table to benchmark reports and use it to calibrate `Runtime.AUTO` / `Device.auto()` decisions.

---

### 8.2 Test F1 — SciPy backend prototype for tiny workloads

**Purpose:** Avoid paying JAX compile/enqueue overhead for tiny scalar-ish threshold loops.

The TODO already lists a SciPy backend prototype as Phase 7.6.3. That should be tested against:

```text
JAX CPU cold
JAX CPU warm
JAX GPU cold
JAX GPU warm
SciPy CPU
```

Workloads:

```text
B = 1, 2, 5, 10, 20
Nt = 100, 500, 1000
Nx = 51
formulation = single, double
use case = threshold search repeated calls
```

For double-cable, SciPy options include:

- sparse block matrix with `scipy.sparse.linalg.spsolve`
- banded formulation if derived
- custom NumPy Thomas block solver

**Key metric:** end-to-end time for repeated threshold search, not just one solve.

**Acceptance:**

- If SciPy wins for tiny workloads, implement backend policy:

```text
if formulation == double and B < measured_gpu_crossover and no need for GPU-specific path:
    choose CPU/JAX or SciPy
```

- If SciPy loses to JAX CPU warm but wins cold, use it only for first-call or no-cache workflows.

---

### 8.3 Test F2 — Backend policy simulation

**Purpose:** Test policy decisions without changing public behavior first.

Add a dry-run planner that reports:

```text
estimated backend: cpu/gpu/scipy
reason: small double-cable B below crossover
estimated memory
estimated compile risk
```

Then compare policy recommendations to actual timings.

**Acceptance:** Policy should choose the measured winner for at least 80–90% of rows in the calibration matrix. Keep manual override options.

---

## 9. Factorized Extracellular Forcing Tests

The TODO says true in-scan `waveform[Nt] * footprint[B,Nx]` forcing is deferred until Phase 8 reuse/study contracts. That is reasonable for public API design. But a private solver-only prototype is still useful.

### 9.1 Test G0 — Dense vs factorized solver-only forcing

**Purpose:** Measure whether avoiding dense `Vstim[B,Nt,Nx]` improves double-cable runs.

Current dense path concept:

```text
vext_mid:      (B, Nt, Nx)
vext_previous: (B, Nt, Nx) or initial previous path
```

Factorized prototype:

```text
footprint: (B, Nx)
waveform_mid: (Nt,)
waveform_previous: (Nt,) or initial previous handling
```

Inside the time scan:

```python
vext_mid_t = waveform_mid[t] * footprint
vext_prev_t = waveform_previous[t] * footprint
```

For multiple electrodes/drives:

```python
vext_mid_t = sum_k waveform_mid[t, k] * footprint[k, B, Nx]
```

Start with one drive only.

**Matrix:**

```text
B = 100, 300, 600, 1000
Nt = 1000, 5000
Nx = 51
formulation = double
backend = CPU, GPU
```

**Record:**

```text
input materialization time
host-to-device transfer bytes
kernel time
peak memory
compile time
```

**Expected:**

- Memory and transfer should improve.
- Kernel time may improve if memory bandwidth was significant.
- If solver compute dominates, runtime may improve only modestly.

**Decision rule:**

- If factorized forcing gives >10% end-to-end improvement for long runs, prioritize Phase 8 drive-reuse contract.
- If it only reduces memory, still valuable for large studies, but not the core solver fix.

---

### 9.2 Test G1 — In-scan diffusion forcing vs precomputed forcing

For single-cable, extracellular forcing often uses a diffusion operator applied to `Vstim`. For double-cable, ensure the equivalent terms are treated consistently.

Compare:

```text
A. precompute forcing arrays outside scan
B. compute forcing from factorized drive inside scan
C. compute raw vext inside scan and apply spatial operator inside scan
```

**Acceptance:** Keep the variant with best end-to-end time and lowest memory for study-sized workloads.

---

## 10. Double-Cable Observer Tests

The dispatcher currently avoids batch mode for double-cable groups when observers are present. That blocks trace-free double-cable studies from getting the same benefit as single-cable observer-only runs.

### 10.1 Test H0 — Add double-cable batch observer smoke path

**Purpose:** Allow `Recording.none()` plus solver-side observers for homogeneous double-cable batches.

Start with the simplest observer:

```text
PeakVoltage at center or distal index
```

Then activation:

```text
Activation threshold with blanking
```

**Implementation idea:** Mirror the single-cable observer state update pattern, but feed `Vm = Vi - Ve` or the canonical membrane voltage used by the double-cable solver.

**Initial constraints:**

- No padded groups at first.
- Homogeneous shared geometry only.
- Same observer positions across rows.
- No row-specific masks until later.

**Benchmark matrix:**

```text
B = 100, 300, 600, 1000
Nt = 1000
Nx = 51
recording = none
observer = peak, activation
formulation = double
backend = CPU, GPU
```

**Compare:**

```text
full Vm retention
center/probe recording
observer-only recording.none
scalar fallback with observers
```

**Expected:** This may not fix the linear solver bottleneck, but it should reduce output memory and result packaging overhead for threshold/recruitment workflows.

**Acceptance:**

- Observer-only double-cable batch path must match post-hoc analysis on retained Vm for non-borderline cases.
- Dispatcher should no longer reject `observers and group.mode == "double"` when the group meets supported constraints.

---

### 10.2 Test H1 — Observer-only threshold/recruitment benchmark

**Purpose:** Measure real workflow impact.

Build a benchmark that runs repeated stimulation amplitudes:

```text
amplitudes = 16 or 32
B = 100, 300, 600
formulation = double
recording = none
observer = activation
```

Compare:

```text
current scalar or retained-output behavior
new double-cable batch observer behavior
```

**Key metric:** total workflow time and peak retained memory, not single kernel time.

**Acceptance:** If observer-only double-cable reduces workflow time or memory by a large factor, it is worth prioritizing even before PCR.

---

## 11. Benchmark Reporting Improvements

The TODO already mentions improving benchmark summaries with percentages of root time, median/p95 columns, parent names, and enough dimensions to compare runs without reopening every `events.jsonl`. This is important for the double-cable work.

### 11.1 Required report columns

Every report row should include:

```text
session_id
workload
scenario_label
formulation
axon_template
B
Nt
Nx
dtype
backend
device
recording
observers
Iinj_format
Vstim_format
cold_or_warm
warmup_count
repeat_count
root_total_ms_median
kernel_enqueue_ms_median
kernel_wait_ms_median
runtime_prepare_ms_median
inputs_intracellular_ms_median
inputs_extracellular_ms_median
result_packaging_ms_median
retained_output_mib
temporary_input_mib_estimated
skipped_dense_iinj_mib
compile_count or compile_logged
```

### 11.2 Derived columns

Add:

```text
kernel_total_ms = kernel_enqueue_ms + kernel_wait_ms
kernel_pct_of_root = kernel_total_ms / root_total_ms
preprocessing_pct_of_root
packaging_pct_of_root
speedup_vs_cpu
speedup_vs_baseline_solver
```

### 11.3 Minimum plots

For each major experiment, generate:

1. `runtime_ms vs B`
2. `speedup_vs_cpu vs B`
3. `runtime_ms vs Nx`
4. `kernel_pct_of_root vs B`
5. `memory_mib vs B`

Do not overfocus on total runtime if the kernel is only 30% of the root time. Conversely, if kernel is 90% of root time, prioritize solver changes.

---

## 12. Suggested Work Order

### Stage 1 — Lock baseline and no-Iinj measurement

1. Re-run baseline single/double long GPU matrix.
2. Re-run double-cable after no-Iinj kernel specialization.
3. Confirm solver-only reproduces double-cable weakness.
4. Add benchmark report columns if missing.

**Deliverable:** A short benchmark note with before/after tables.

---

### Stage 2 — Current solver characterization

1. Direct block solver microbenchmark.
2. Nx scaling.
3. B scaling.
4. Nt scaling.
5. dtype sweep.
6. layout sweep: `vmap(scan)` vs `scan(vectorized-B)`.
7. HLO inspection for representative cases.

**Deliverable:** A solver bottleneck report that answers:

```text
Is the block solver definitely the bottleneck?
Which shape dimension controls GPU crossover?
Does layout improve current Thomas enough to avoid PCR?
```

---

### Stage 3 — Alternative solver prototypes

1. Materialized block solver probe.
2. Batched-space Thomas.
3. PCR / cyclic reduction prototype.
4. Dense oracle for correctness.
5. Optional Schur complement/pentadiagonal prototype.

**Deliverable:** Solver comparison table:

```text
current Thomas
batched-space Thomas
PCR
dense oracle
pentadiagonal prototype, if implemented
```

---

### Stage 4 — Backend policy

1. Build CPU/GPU crossover table.
2. Prototype SciPy/NumPy tiny backend.
3. Implement dry-run policy recommendations.
4. Compare policy to measured winners.

**Deliverable:** Backend selection policy proposal.

---

### Stage 5 — Workflow-level improvements

1. Double-cable observer-only batch path.
2. Threshold/recruitment benchmark.
3. Factorized extracellular forcing private prototype.
4. Decide what belongs in Phase 8 public reuse/study API.

**Deliverable:** Workflow benchmark note and Phase 8 API input.

---

## 13. Concrete Acceptance Criteria

### 13.1 Merge immediately

A candidate can be merged quickly if:

- It improves performance by at least 5–10% on a representative workload, or materially reduces memory.
- It does not complicate public API.
- It passes unit tests and numerical comparison tests.
- It has a benchmark row or smoke test preventing regression.

Examples likely in this category:

```text
no-Iinj specialization
benchmark report improvements
backend dry-run metadata
small layout cleanup if clearly faster
```

### 13.2 Merge behind internal option

A candidate should be hidden behind a solver option if:

- It is faster only for some shapes.
- It is more complex but promising.
- It needs more validation.

Examples:

```text
PCR solver
batched-space Thomas if GPU-only win
factorized forcing internal prototype
```

### 13.3 Defer

Defer if:

- It depends on Phase 8 public API contracts.
- It improves memory but not runtime and current workloads are not memory-bound.
- It adds maintenance burden without clear measured gain.

Examples:

```text
full public factorized-drive API
complex Schur complement if PCR is better
custom kernel before JAX prototype proof
```

### 13.4 Drop

Drop if:

- It is slower in the solver-only matrix.
- It causes correctness drift that cannot be explained.
- It only improves one artificial case while hurting representative workloads.

---

## 14. Risks and Failure Modes

### 14.1 PCR may not win for Nx=51

`Nx=51` is small enough that the extra arithmetic and temporary arrays of PCR may outweigh the reduced dependency depth. That would not invalidate the diagnosis; it would mean the current problem is better handled by backend policy, layout, and maybe custom kernels rather than a high-level PCR rewrite.

### 14.2 Dense oracle may look good for tiny smoke cases

A dense solve can look attractive for very small `Nx` and `Nt`, but it likely will not scale to long runs or large batches. Treat it as an oracle unless it wins in full realistic matrices.

### 14.3 Float32 correctness may be workload-dependent

If float32 is much faster but produces borderline activation differences, benchmark non-borderline stimuli first. Then test threshold workflows separately with tolerances appropriate to the algorithm.

### 14.4 Benchmark noise can hide small wins

Small speedups under 5% are hard to trust on Colab. Use repeated warm runs, p50/p95, and compare against local CPU runs. Keep only wins that survive repetition.

### 14.5 Public API pressure

Do not expose factorized-drive or solver-selection knobs prematurely. Keep prototypes internal until Phase 8 reuse policies are designed.

---

## 15. Recommended Immediate PRs / Commits

### PR 1 — Benchmark and evidence cleanup

Contents:

- Ensure `solver_only_precomputed` covers double-cable with absent `Iinj`.
- Add summary columns listed above.
- Add CPU/GPU crossover table output.
- Add environment capture if missing.

**Why first:** It reduces uncertainty before solver work.

---

### PR 2 — No-Iinj specialization finalization

Contents:

- Remove device-side `jnp.zeros_like(vext_mid)` from full double-cable no-Iinj path if still present.
- Add separate no-Iinj jitted function if it benchmarks faster.
- Add smoke test verifying no dense `Iinj` input is passed.
- Re-run Colab double-cable long benchmark.

**Why second:** Easy win and already aligned with TODO.

---

### PR 3 — Block solver microbenchmarks

Contents:

- Add direct block solver benchmark.
- Add layout variants under internal benchmark-only functions.
- Add HLO dump instructions to benchmark docs.

**Why third:** This determines whether PCR is worth implementing.

---

### PR 4 — Batched-space Thomas prototype

Contents:

- Implement `solve_block_tridiagonal_2x2_batched_space_scan(...)` internally.
- Compare against current solver.
- Keep disabled by default unless it clearly wins.

**Why fourth:** Lower complexity than PCR and may provide quick GPU occupancy improvements.

---

### PR 5 — PCR prototype branch

Contents:

- Solver-only PCR implementation.
- Correctness tests against current solver and dense oracle.
- Benchmark matrix for `B`, `Nt`, `Nx`.

**Why fifth:** Highest potential solver-side improvement but more risk.

---

### PR 6 — Backend policy prototype

Contents:

- Add dry-run backend recommendations.
- Optional SciPy tiny backend if benchmark confirms.
- Use measured crossover thresholds for double-cable.

**Why sixth:** This improves real user workflows even if solver rewrite takes longer.

---

### PR 7 — Double-cable observer-only batch support

Contents:

- Add minimal double-cable observer state update.
- Support homogeneous non-padded double-cable groups first.
- Remove dispatcher rejection for supported observer cases.
- Benchmark threshold/recruitment workflows.

**Why seventh:** It reduces retained output and unlocks study workloads.

---

## 16. Minimal Experiment Checklist

Use this checklist before deciding on the solver rewrite.

```text
[ ] Baseline single/double long GPU matrix re-run
[ ] Double-cable no-Iinj Colab rerun
[ ] Solver-only double-cable reproduces weak GPU scaling
[ ] Direct block solver microbenchmark added
[ ] Nx scaling curve generated
[ ] B crossover curve generated
[ ] Nt scaling curve generated
[ ] dtype sweep completed
[ ] vmap(scan) vs scan(vectorized-B) tested
[ ] HLO inspected for no-Iinj and current solver
[ ] Dense oracle implemented for small correctness tests
[ ] PCR prototype benchmarked solver-only
[ ] Backend policy table drafted
[ ] Double-cable observer-only feasibility tested
```

Do not start a custom CUDA/Pallas kernel before at least the first ten items are complete.

---

## 17. Final Recommendation

The most valuable next test is not a large solver rewrite. It is a controlled solver-only benchmark that proves the current 2x2 block-tridiagonal solve is the dominant GPU limiter across `B`, `Nt`, and `Nx`. Once that is confirmed, the next best low-risk implementation test is a batched-space rewrite of the current Thomas solver, because it preserves the algorithm while changing the compiler-visible parallelism. If that does not move the needle enough, implement a PCR/cyclic-reduction prototype as a solver-only candidate.

In parallel, finish the no-Iinj specialization and backend crossover policy. Those two are practical improvements even if the solver rewrite takes longer. Finally, add double-cable observer-only batch support because real threshold/recruitment studies care about workflow-level runtime and retained memory, not only raw kernel throughput.

---

## 18. Source References

- AxonScope branch reviewed: `https://github.com/louisreg/AxonScope/tree/bench-colab`
- Current roadmap and benchmark evidence: `https://raw.githubusercontent.com/louisreg/AxonScope/bench-colab/todo.md`
- Current block-tridiagonal solver file: `https://github.com/louisreg/AxonScope/blob/bench-colab/src/axonscope/solvers/common.py`
- Batch kernel file: `https://github.com/louisreg/AxonScope/blob/bench-colab/src/axonscope/solvers/batch_kernels.py`
- Dispatcher file: `https://github.com/louisreg/AxonScope/blob/bench-colab/src/axonscope/dispatcher/execution.py`
- Performance estimate/planning file: `https://github.com/louisreg/AxonScope/blob/bench-colab/src/axonscope/performance.py`
- JAX tridiagonal solve documentation: `https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html`
- JAX asynchronous dispatch note: `https://docs.jax.dev/en/latest/async_dispatch.html`
- JAX GPU performance tips: `https://docs.jax.dev/en/latest/gpu_performance_tips.html`
- JAX Pallas documentation: `https://docs.jax.dev/en/latest/pallas/index.html`
