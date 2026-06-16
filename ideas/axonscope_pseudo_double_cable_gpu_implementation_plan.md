# AxonScope: GPU-Compatible Pseudo-Double-Cable Implementation Plan

Status: standby as of 2026-06-16. Keep this as background research and do not
use it as the active implementation roadmap unless pseudo-double work is
explicitly resumed. Current active solver work is the exact double-cable GPU
optimization in `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`.
Pseudo-double modes remain validation-only under `benchmark/pseudo_double/` and
must not be added to `BatchOptions.double_cable_block_solver` or `auto`.

## 0. Goal and scope

This document describes a step-by-step implementation plan for adding one or more **pseudo-double-cable** modes to AxonScope.

The target workload is:

- `Nx = 30–100` compartments per fiber.
- `B > 500` fibers, ideally `B_effective >= 1024–4096` after batching fibers, amplitudes, electrode configurations, or stimulation conditions.
- GPU execution, especially Google Colab / CUDA-backed JAX.
- Large screening workloads: activation, threshold, recruitment, relative electrode/configuration comparison.

The main goal is not to replace the exact double-cable model everywhere. The goal is to add one or more **GPU-friendly surrogate modes** that keep the most important double-cable behavior while avoiding the exact 2x2 block-tridiagonal solve at every time step.

The exact double-cable solver remains the reference implementation.

The pseudo-double-cable modes should be used as:

1. A fast screening model.
2. A pre-filter before exact double-cable refinement.
3. A way to run large GPU studies where exact double-cable is too slow.
4. A controlled approximation whose error is explicitly measured against the exact double-cable model.

---

## 1. Current situation in the branch

The current branch already has several important pieces:

- An exact 2x2 block-tridiagonal double-cable solver.
- A scalar-specialized implementation for the 2x2 block system.
- A GPU-oriented PCR-style alternative in `solve_block_tridiagonal_2x2_pcr`.
- Batch kernels for single- and double-cable simulation.
- A dispatcher that routes compatible batches through batch kernels.
- Benchmark notes showing that single-cable scales well on GPU while double-cable scales less well and only becomes clearly useful at larger batch sizes.

The current exact double-cable bottleneck is structurally expected: each time step contains a spatial solve. The specialized block-Thomas solver has a forward pass and a backward pass along `Nx`, and therefore exposes limited spatial parallelism. For `Nx=30–100`, this is not catastrophic, but it means the GPU must be fed primarily through a large batch axis.

A pseudo-double-cable mode should therefore try to replace:

```text
exact double-cable step = pointwise HH/gating + exact 2x2 block-tridiagonal solve
```

with one of:

```text
pseudo step A = pointwise HH/gating + one scalar tridiagonal solve
pseudo step B = pointwise auxiliary update + one scalar tridiagonal solve
pseudo step C = two scalar tridiagonal solves + local mixing transform
```

The key idea is to return to GPU-friendly scalar tridiagonal solves and pointwise operations.

---

## 2. Proposed model family

Add three progressively more faithful modes:

```text
single
pseudo_double_effective
pseudo_double_split
pseudo_double_modal
double
```

The modes should be ordered by expected speed and fidelity:

| Mode | Expected speed | Expected fidelity | Solver structure | Intended use |
|---|---:|---:|---|---|
| `single` | best | lowest | one scalar tridiagonal | existing fast baseline |
| `pseudo_double_effective` | excellent | medium | one scalar tridiagonal | large screening |
| `pseudo_double_split` | very good | medium/high | pointwise auxiliary update + one scalar tridiagonal | screening with better double-cable dynamics |
| `pseudo_double_modal` | good/very good | high if assumptions hold | two scalar tridiagonal solves | high-fidelity surrogate |
| `double` | slowest | reference | exact 2x2 block-tridiagonal | validation/refinement |

The first implementation should prioritize `pseudo_double_effective` and `pseudo_double_split`. `pseudo_double_modal` is more ambitious and should be implemented only after the first two give useful speed/fidelity tradeoffs.

---

## 3. Acceptance criteria

The pseudo-double modes should not be judged only by voltage-trace error. They should be judged by the quantities that matter for the intended large-batch use cases.

### 3.1 Speed criteria

For the target range:

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096
Nt = realistic production value
```

Minimum useful target:

```text
pseudo_double_effective speedup vs exact double >= 2x
pseudo_double_split     speedup vs exact double >= 1.5x
pseudo_double_modal     speedup vs exact double >= 1.25x
```

Strong target:

```text
pseudo_double_effective speedup vs exact double >= 5x
pseudo_double_split     speedup vs exact double >= 3x
pseudo_double_modal     speedup vs exact double >= 2x
```

### 3.2 Accuracy criteria

Measure error against exact double-cable, not against single-cable.

For activation/threshold workloads:

```text
threshold relative error <= 1–3%      ideal
threshold relative error <= 5%        acceptable for screening
threshold relative error > 5–10%      use only as rough pre-screen
```

For recruitment curves:

```text
fiber activation classification agreement >= 95% near operating range
rank correlation of electrode/configuration scores >= 0.95
ambiguous-zone recall >= 99% if pseudo mode is used as a pre-filter
```

For full Vm traces:

```text
peak Vm error             tracked, but not the only target
time-to-peak error        tracked
RMS Vm error              tracked
activation time error     tracked
node of activation error  tracked
```

Pseudo modes may be acceptable even when local trace error is not perfect, if threshold and recruitment decisions remain accurate.

---

## 4. Implementation overview

Recommended sequence:

```text
Step 1  Add mode plumbing and API names.
Step 2  Add baseline validation and benchmark harnesses.
Step 3  Implement pseudo_double_effective.
Step 4  Validate pseudo_double_effective against exact double.
Step 5  Implement pseudo_double_split.
Step 6  Validate pseudo_double_split against exact double.
Step 7  Optionally implement pseudo_double_modal.
Step 8  Add GPU traces and regression benchmarks.
Step 9  Add hybrid screening + exact-refinement workflow.
Step 10 Document limitations and defaults.
```

Do not start with the modal implementation. Start with the simplest useful surrogate, get the benchmark/validation infrastructure correct, then add complexity only if the error requires it.

---

# Step 1 — Add public mode plumbing

Standby override: do not implement this step while pseudo-double work is
paused. Keep any pseudo-double plumbing confined to `benchmark/pseudo_double/`
unless the project explicitly resumes this line of work and updates `todo.md`
first.

## Goal

Make pseudo-double modes first-class options in the solver API without changing existing behavior.

## Files likely to touch

Exact paths may differ slightly, but likely targets are:

```text
src/axonscope/solvers/*
src/axonscope/dispatcher/execution.py
src/axonscope/solvers/batch_kernels.py
src/axonscope/performance.py
src/axonscope/models/* or axon/fiber configuration code
```

## Actions

Add explicit mode names:

```python
CableMode = Literal[
    "single",
    "double",
    "pseudo_double_effective",
    "pseudo_double_split",
    "pseudo_double_modal",
]
```

If the project does not currently use a shared enum or literal type, add one. Avoid scattering raw strings.

## Dispatch rules

Initial dispatch should be conservative:

```text
single                    -> existing single batch kernel
pseudo_double_effective   -> new pseudo batch kernel, scalar tridiagonal
pseudo_double_split       -> new pseudo batch kernel, scalar tridiagonal + auxiliary state
pseudo_double_modal       -> new pseudo batch kernel, two scalar tridiagonals
double                    -> existing exact double batch kernel
```

Do not silently reinterpret `double` as pseudo-double.

## Configuration suggestion

Expose pseudo options explicitly:

```python
simulate(..., mode="pseudo_double_effective")
simulate(..., mode="pseudo_double_split")
simulate(..., mode="pseudo_double_modal")
```

Optional future API:

```python
simulate(..., mode="double", approximation="effective")
```

But the first version should use explicit mode names because it is safer and easier to benchmark.

## Done when

- Existing `single` tests pass.
- Existing `double` tests pass.
- New mode strings are accepted by the dispatcher.
- New modes currently raise a clear `NotImplementedError` until their kernels are implemented.

---

# Step 2 — Add validation and benchmark harnesses first

## Goal

Before implementing the approximation, create the tests that will tell whether it is useful.

## Why this matters

Pseudo-double-cable is not automatically correct. It is useful only if it gives the right operational outputs: activation, threshold, recruitment, and ranking.

## New files

Add benchmark and validation scripts:

```text
benchmarks/validate_pseudo_double.py
benchmarks/bench_pseudo_double_gpu.py
benchmarks/profile_pseudo_double_jax.py
```

Add optional test fixtures:

```text
tests/test_pseudo_double_shapes.py
tests/test_pseudo_double_reference.py
tests/test_pseudo_double_threshold.py
```

## Validation matrix

Start small and deterministic:

```text
Nx = 32, 51, 64, 96
B  = 16 for correctness
B  = 512, 1024, 2048, 4096 for GPU speed
Nt = 100, 500, production Nt
```

Use a fixed seed for synthetic fibers/stimuli.

## Outputs to compare

For each mode:

```text
Vm trace, if recording is enabled
peak Vm per fiber
peak Vm per node
activation boolean per fiber
activation time
threshold amplitude
recruitment curve
```

## Benchmark rules

Always separate:

```text
compile time
first run time
steady-state run time
host-to-device transfer
device execution
postprocessing/output time
```

Always synchronize timing with:

```python
jax.block_until_ready(out)
```

Do not report asynchronous dispatch time as GPU runtime.

## Done when

- You can run exact double and pseudo modes on the same generated workload.
- The script prints speed, memory/output size, and error metrics.
- The script can save a JSON or CSV summary.

Suggested output schema:

```json
{
  "mode": "pseudo_double_effective",
  "B": 1024,
  "Nx": 64,
  "Nt": 1000,
  "backend": "gpu",
  "dtype": "float32",
  "runtime_s": 0.0,
  "node_steps_per_s": 0.0,
  "speedup_vs_double": 0.0,
  "threshold_rel_error_mean": 0.0,
  "threshold_rel_error_p95": 0.0,
  "activation_agreement": 0.0
}
```

---

# Step 3 — Implement `pseudo_double_effective`

## Goal

Add the simplest GPU-compatible surrogate: a single scalar cable equation with effective parameters derived from, or calibrated against, the double-cable model.

This should be the first implemented pseudo mode.

## Concept

Replace the exact double-cable system with:

```text
Cm_eff * dVm/dt = axial_eff(Vm) + ionic(Vm, gates) + drive_eff(Vext)
```

Then solve the diffusion/cable part with the existing scalar tridiagonal path.

This keeps the computational structure close to the current fast single-cable solver.

## Design choices

There are two ways to get effective coefficients.

### Option A — physics-derived effective parameters

Use available double-cable parameters to derive:

```text
Cm_eff
Gm_eff
Ra_eff or D_eff
Vext coupling factor
```

This is fast and deterministic but may require careful unit checks.

### Option B — calibration-derived effective parameters

Fit effective coefficients so that the pseudo model matches exact double-cable responses on a small calibration set.

This may be more robust than trying to derive a closed-form approximation for every heterogeneous fiber case.

Recommended first version:

```text
Implement physics-derived defaults.
Allow optional calibration multipliers.
```

Example parameterization:

```python
@dataclass(frozen=True)
class PseudoDoubleEffectiveConfig:
    cm_scale: float = 1.0
    axial_scale: float = 1.0
    vext_scale: float = 1.0
    leak_scale: float = 1.0
    calibration_name: str | None = None
```

## Implementation sketch

Add helper:

```python
def effective_double_cable_coeffs(axon, dtype):
    """Return scalar-cable coefficients for pseudo_double_effective.

    This function must use the same units as the existing solver.
    It should return coefficients compatible with the existing
    single-cable CN/tridiagonal path.
    """
    lower, diag, upper = diffusion_operator_coeffs(axon, dtype)

    # Placeholder multipliers; replace with derived or calibrated values.
    lower_eff = axial_scale * lower
    diag_eff = axial_scale * diag
    upper_eff = axial_scale * upper

    cm_eff_scale = cm_scale
    vext_eff_scale = vext_scale

    return lower_eff, diag_eff, upper_eff, cm_eff_scale, vext_eff_scale
```

The first version can reuse the existing single-cable solver with modified coefficients and forcing.

## Important implementation constraint

Do not copy or duplicate the whole single-cable solver if avoidable.

Preferred structure:

```text
common scalar solver core
    used by single
    used by pseudo_double_effective
    used by pseudo_double_split
```

If the existing single-cable path is not factored that way, add a small internal helper first.

## Expected performance

This mode should be close to single-cable performance.

Expected speed ranking:

```text
single ~= pseudo_double_effective > pseudo_double_split > pseudo_double_modal > exact double
```

## Done when

- `pseudo_double_effective` runs on CPU and GPU.
- It uses a scalar tridiagonal solve, not a 2x2 block solve.
- It supports `B > 500` batch runs.
- It supports no full trace output / observer-only output if the rest of the code supports it.
- It passes shape and dtype tests.

---

# Step 4 — Validate `pseudo_double_effective`

## Goal

Decide whether the simple effective model is already good enough for screening.

## Validation protocol

Run exact double and pseudo-effective on the same workload:

```text
Nx = 32, 51, 64, 96
B  = 64 for validation, then 512+ for performance
stimulus amplitudes = below threshold, near threshold, above threshold
recording = center/probes + compact activation outputs
```

## Metrics

Report:

```text
peak Vm absolute error
peak Vm relative error
activation boolean agreement
threshold amplitude error
recruitment curve error
rank correlation across electrode configs
```

## Go/no-go

Proceed to `pseudo_double_split` if:

```text
threshold p95 relative error > 3–5%
activation agreement near threshold is poor
ranking changes significantly
```

Keep `pseudo_double_effective` as a production screening mode if:

```text
speedup >= 3–5x
threshold error <= 5%
activation agreement >= 95%
```

Even if error is too high for final results, keep it as a rough pre-screen if it has high recall for potentially activated fibers.

---

# Step 5 — Implement `pseudo_double_split`

## Goal

Add a more faithful surrogate with an auxiliary double-cable-like state while preserving a GPU-friendly scalar tridiagonal solve.

## Concept

Maintain two state variables:

```text
Vm      primary membrane voltage state
Vy      auxiliary myelin/periaxonal/outer-cable state
```

But avoid solving the fully coupled 2x2 block-tridiagonal system.

Use an operator-split step:

```text
1. Update Vy locally or semi-locally using old Vm and Vext.
2. Build an effective RHS for Vm.
3. Solve one scalar tridiagonal system for Vm.
4. Optionally correct Vy using new Vm.
```

This preserves an explicit slow auxiliary response without paying for the exact block solve.

## Step structure

Pseudo-code:

```python
def pseudo_double_split_step(carry, inputs):
    Vm, gates, Vy = carry
    stimulus_t = inputs

    # 1. Pointwise auxiliary update.
    Vy_pred = update_auxiliary_state(
        Vy=Vy,
        Vm=Vm,
        Vext=stimulus_t,
        dt=dt,
        params=params,
    )

    # 2. Convert auxiliary state into effective drive/coupling.
    rhs_drive = build_split_rhs_drive(
        Vm=Vm,
        Vy=Vy_pred,
        Vext=stimulus_t,
        params=params,
    )

    # 3. Existing scalar cable step.
    Vm_next = scalar_cable_cn_step(
        Vm=Vm,
        gates=gates,
        rhs_drive=rhs_drive,
        lower=lower_eff,
        diag=diag_eff,
        upper=upper_eff,
    )

    # 4. Optional correction.
    Vy_next = correct_auxiliary_state(
        Vy_pred=Vy_pred,
        Vm_next=Vm_next,
        Vext=stimulus_t,
        dt=dt,
        params=params,
    )

    gates_next = update_gates(Vm_next, gates, dt)
    return (Vm_next, gates_next, Vy_next), outputs
```

## Auxiliary update options

### Option A — explicit Euler auxiliary state

Simplest:

```text
Vy_next = Vy + dt * f(Vy, Vm, Vext)
```

Pros:

```text
easy to implement
fully pointwise
fast
```

Cons:

```text
may require small dt for stability
less accurate for stiff myelin/periaxonal dynamics
```

### Option B — implicit pointwise auxiliary state

Better:

```text
Vy_next = solve_local_1x1_or_2x2(Vy, Vm, Vext)
```

This is still pointwise across compartments and fibers, so it is GPU-friendly.

Pros:

```text
more stable
still avoids spatial block solve
better for stiff auxiliary dynamics
```

Recommended first version:

```text
implicit pointwise auxiliary update
```

## Where to put the code

Add new internal helpers:

```text
src/axonscope/solvers/pseudo_double.py
```

Suggested functions:

```python
def pseudo_double_effective_coeffs(...): ...
def pseudo_double_split_coeffs(...): ...
def update_split_auxiliary_state(...): ...
def build_split_rhs_drive(...): ...
def run_pseudo_double_split_batch(...): ...
```

Then call them from:

```text
src/axonscope/solvers/batch_kernels.py
```

## Important constraints

- Avoid Python conditionals inside the time loop that depend on traced values.
- Keep state arrays shaped consistently, preferably `[B, Nx]` for batch kernels.
- Avoid materializing dense zero intracellular current tensors.
- Keep `Vext` factorization possible in the future: `waveform[Nt] * footprint[B, Nx]`.

## Done when

- `pseudo_double_split` runs on CPU and GPU.
- It uses one scalar tridiagonal solve per time step.
- It has one auxiliary state array, ideally `[B, Nx]`.
- It supports compact outputs.
- It is benchmarked against exact double.

---

# Step 6 — Validate `pseudo_double_split`

## Goal

Determine whether the split model is sufficiently close to exact double-cable for production screening.

## Validation cases

Use the same validation matrix as `pseudo_double_effective`, but add cases where double-cable effects are expected to matter more:

```text
strong extracellular gradients
short pulses
long pulses
fibers with different diameters
heterogeneous cable properties, if supported
near-threshold amplitudes
```

## Metrics

Same as Step 4, plus:

```text
auxiliary state sanity checks
stability across long Nt
sensitivity to dt
```

## Go/no-go

Keep `pseudo_double_split` if:

```text
speedup >= 2–3x vs exact double
threshold error <= 3–5%
classification agreement >= 95–98%
```

If `pseudo_double_effective` and `pseudo_double_split` both fail accuracy targets, proceed to `pseudo_double_modal`.

---

# Step 7 — Optional: implement `pseudo_double_modal`

## Goal

Create a more faithful pseudo-double model by transforming the coupled variables into two approximately decoupled modes, each solved with a scalar tridiagonal solver.

## Concept

The exact double-cable unknowns can be thought of as two coupled variables, for example:

```text
Vi
Ve or periaxonal/myelin-related potential
```

Instead of solving a 2x2 block-tridiagonal system, approximate the system by a local modal transform:

```text
mode_fast = p00 * Vi + p01 * Ve
mode_slow = p10 * Vi + p11 * Ve
```

Then solve:

```text
mode_fast_next = scalar_tridiagonal_solve(...)
mode_slow_next = scalar_tridiagonal_solve(...)
```

Finally transform back:

```text
Vi_next, Ve_next = inverse_transform(mode_fast_next, mode_slow_next)
Vm_next = Vi_next - Ve_next
```

## When this is valid

This is most promising when the double-cable coefficients are:

```text
homogeneous or slowly varying along x
similar across compartments
not strongly discontinuous between node/internode regions
```

If coefficients are highly heterogeneous, use a locally varying transform or fall back to `pseudo_double_split` / exact double.

## Implementation strategy

Start with the simplest constant-transform version:

```python
def compute_modal_transform(axon, dtype):
    """Return P and Pinv for approximate local modal decoupling."""
    # Use representative mean coefficients.
    # Then eigendecompose the local 2x2 coupling matrix.
    return P, Pinv
```

Then construct two scalar tridiagonal systems:

```python
lower_fast, diag_fast, upper_fast = ...
lower_slow, diag_slow, upper_slow = ...
```

The exact derivation depends on how the current double-cable coefficients are assembled. Do not guess silently: implement this behind a test comparing the reconstructed one-step linear solve against the exact block solve for small synthetic systems.

## Validation before time stepping

Before integrating into the full simulator, add a one-step linear-system test:

```text
Given exact double-cable block system A x = rhs
Given pseudo modal approximation A_modal x ~= rhs
Compare x_exact and x_modal for random rhs
```

Use:

```text
Nx = 32, 64, 96
random rhs
representative axon parameters
```

Report:

```text
relative solve error
max absolute solve error
worst node error
```

Only integrate into the full simulator if the one-step approximation error is reasonable.

## Done when

- `pseudo_double_modal` uses two scalar tridiagonal solves.
- It has a one-step solve approximation test.
- It has full simulation validation against exact double.
- It is faster than exact double for `B>500`.

---

# Step 8 — Add factorized stimulation support

## Goal

Avoid materializing dense stimulation tensors when the stimulation can be represented as:

```text
Vext[B, Nt, Nx] = waveform[Nt] * footprint[B, Nx]
```

This matters for pseudo-double and exact double.

## Why this helps

For example:

```text
B  = 2048
Nt = 1000
Nx = 64
float32 Vext = 2048 * 1000 * 64 * 4 bytes ~= 524 MB
```

That is too much memory traffic for something that can often be generated from a waveform and a footprint.

## API suggestion

Support both forms:

```python
simulate(..., vext=vext_dense)
simulate(..., vext_factorized=(waveform, footprint))
```

or:

```python
simulate(..., waveform=waveform, extracellular_footprint=footprint)
```

## Solver loop behavior

Inside the time scan:

```python
vext_t = waveform[t] * footprint_BxNx
```

Do not build `Vext[B, Nt, Nx]` unless explicitly requested for debugging.

## Done when

- Pseudo modes support factorized stimulation.
- Exact double can optionally support it too.
- Benchmarks report dense-vs-factorized memory and runtime.

---

# Step 9 — Add compact observers for pseudo modes

## Goal

Avoid storing full `Vm[B, Nt, Nx]` unless the user asks for it.

## Recommended outputs

For screening:

```text
activated[B]
activation_time[B]
activation_node[B]
peak_vm[B]
peak_node[B]
threshold estimate, if running amplitude search
```

For debugging:

```text
Vm_center[B, Nt]
Vm_probe[B, Nt, K]
Vm_full[B, Nt, Nx]
```

Full traces should be opt-in.

## Implementation rule

Observers should be updated inside the JAX time scan, not as Python postprocessing over a huge full trace.

Pseudo-code:

```python
def update_observer(obs, Vm_t, t):
    peak_vm = jnp.maximum(obs.peak_vm, jnp.max(Vm_t, axis=-1))
    activated_now = jnp.max(Vm_t, axis=-1) >= threshold
    first_activation = activated_now & (~obs.activated)
    activation_time = jnp.where(first_activation, t, obs.activation_time)
    activated = obs.activated | activated_now
    return obs.replace(...)
```

## Done when

- Pseudo modes can run without returning full traces.
- Benchmarks include compact-output mode.
- Memory use is much lower than full trace mode.

---

# Step 10 — Add hybrid pseudo + exact refinement workflow

## Goal

Use pseudo-double for broad screening, then exact double only where needed.

## Workflow

```text
1. Run pseudo-double on all fibers/amplitudes/configs.
2. Classify fibers/configs into:
   - clearly inactive
   - clearly active
   - ambiguous / near threshold
3. Rerun exact double only on ambiguous cases.
4. Merge exact results back into final output.
```

## Ambiguity policy

For threshold work:

```text
ambiguous if abs(pseudo_peak_vm - activation_threshold) < margin
```

For amplitude search:

```text
ambiguous if pseudo threshold lies within tolerance band
```

For recruitment curves:

```text
ambiguous if activation probability or margin is close to decision boundary
```

## Important safety rule

The pseudo pre-filter should be tuned for high recall, not high precision.

It is better to rerun too many fibers with exact double than to incorrectly discard fibers that would activate under the exact model.

## Done when

- A script can run pseudo first and exact second.
- The final merged output clearly marks which fibers were exact-refined.
- The workflow reports how much exact double work was avoided.

---

# Step 11 — Add profiling and tracing

## Goal

Use real JAX traces to confirm that pseudo modes are GPU-friendly.

## Script

Add:

```text
benchmarks/profile_pseudo_double_jax.py
```

It should run:

```bash
python benchmarks/profile_pseudo_double_jax.py --mode pseudo_double_effective --B 1024 --Nx 64 --Nt 1000 --trace-dir /tmp/as-pseudo-effective
python benchmarks/profile_pseudo_double_jax.py --mode pseudo_double_split     --B 1024 --Nx 64 --Nt 1000 --trace-dir /tmp/as-pseudo-split
python benchmarks/profile_pseudo_double_jax.py --mode double                  --B 1024 --Nx 64 --Nt 1000 --trace-dir /tmp/as-double
```

## JAX tracing

Use:

```python
with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
    with jax.profiler.TraceAnnotation("pseudo_double_effective_run"):
        out = run_case()
        jax.block_until_ready(out)
```

## What to inspect

In TensorBoard / Perfetto:

```text
GPU occupancy
number of kernels
large host/device transfers
compile vs run time
single-cable vs pseudo vs exact double timeline shape
whether full trace output dominates runtime
```

## Done when

- Trace artifacts can be generated consistently.
- Pseudo modes show fewer/simpler kernels than exact double.
- No unexpected host-device transfer dominates runtime.

---

# Step 12 — Calibration support

## Goal

Allow pseudo-double modes to be fitted to exact double-cable behavior for a given class of fibers/stimuli.

## Why

A purely physics-derived pseudo model may not match exact double well enough across all use cases. A small calibration layer can make it much more useful.

## Calibration parameters

Start with a small number of stable scalars:

```text
cm_scale
axial_scale
vext_scale
leak_scale
aux_time_constant_scale
aux_coupling_scale
```

Do not start with too many free parameters.

## Calibration target

Fit against exact double outputs:

```text
threshold amplitude
peak Vm
activation time
recruitment curve samples
```

Prefer fitting operational metrics over full trace MSE.

## Calibration script

Add:

```text
benchmarks/calibrate_pseudo_double.py
```

Suggested usage:

```bash
python benchmarks/calibrate_pseudo_double.py \
  --mode pseudo_double_split \
  --Nx 64 \
  --B-calibration 64 \
  --stimulus-set default \
  --output configs/pseudo_double_calibration/default_split.json
```

## Runtime config

Allow:

```python
simulate(
    ..., 
    mode="pseudo_double_split",
    pseudo_calibration="configs/pseudo_double_calibration/default_split.json",
)
```

## Done when

- Calibration improves threshold/recruitment error on held-out fibers.
- Calibration does not destroy speed.
- Calibration files are versioned and include metadata about the training set.

---

# Step 13 — Testing plan

## Unit tests

Add tests for:

```text
mode parsing
shape correctness
CPU/GPU dtype consistency
no dense zero Iinj materialization
factorized stimulation shape handling
compact observer outputs
```

## Numerical tests

For pseudo modes, avoid brittle exact-value tests. Use tolerance-based tests against exact double for small deterministic cases.

Example:

```python
def test_pseudo_effective_threshold_reasonable():
    exact = run(mode="double", ...)
    pseudo = run(mode="pseudo_double_effective", ...)
    assert relative_error(pseudo.threshold, exact.threshold) < 0.10
```

Use loose tolerances at first, then tighten after calibration.

## Performance regression tests

Add optional marked tests:

```text
pytest -m gpu_perf
```

Track:

```text
node_steps_per_s
runtime for B=1024, Nx=64, Nt=1000
memory for full vs compact output
```

Do not make GPU performance tests mandatory in normal CI unless the CI has a stable GPU.

---

# Step 14 — Documentation

## User documentation

Add a section:

```text
Pseudo-double-cable modes
```

Explain:

```text
what each mode does
when to use it
when not to use it
how to validate it against exact double
how to run hybrid pseudo+exact refinement
```

## Warnings to include

```text
pseudo_double_effective is not an exact double-cable model
pseudo_double_split is an approximation
pseudo_double_modal depends on decoupling assumptions
exact double remains the reference model
pseudo modes should be validated for each new fiber/stimulation regime
```

## Suggested user-facing language

```text
Use pseudo-double modes for large GPU screening and recruitment studies.
Use exact double-cable for final validation, ambiguous near-threshold cases, and biophysical analyses where periaxonal/myelin dynamics are the quantity of interest.
```

---

# Step 15 — Recommended implementation order

Do this in order:

```text
1. Add mode names and NotImplemented dispatch.
2. Add validation/benchmark harnesses.
3. Refactor scalar single-cable step into a reusable internal helper if needed.
4. Implement pseudo_double_effective.
5. Add compact observer support for pseudo_double_effective.
6. Benchmark pseudo_double_effective vs exact double.
7. Validate threshold/recruitment error.
8. Implement pseudo_double_split.
9. Benchmark and validate pseudo_double_split.
10. Add factorized stimulation support.
11. Add hybrid pseudo+exact refinement.
12. Add calibration script.
13. Decide whether pseudo_double_modal is still needed.
14. If needed, implement pseudo_double_modal with one-step linear-system validation first.
15. Document modes and recommended defaults.
```

The strongest first milestone is:

```text
pseudo_double_effective runs at near single-cable speed and reports threshold/recruitment error vs exact double.
```

The strongest production milestone is:

```text
pseudo_double_split + compact observers + factorized stimulation gives >=2–3x speedup vs exact double with <=3–5% threshold error on target workloads.
```

---

# Step 16 — Practical defaults for the target workload

For the target workload:

```text
Nx = 30–100
B > 500
```

Recommended defaults:

```text
mode = pseudo_double_split for large screening
mode = pseudo_double_effective for very large rough pre-screening
mode = double for exact refinement
recording = compact observer outputs by default
Vext representation = factorized waveform * footprint when possible
B_effective target = at least 1024, preferably 2048–4096
```

Avoid by default:

```text
full Vm[B, Nt, Nx] output
Vext[B, Nt, Nx] materialization
zero Iinj[B, Nt, Nx] materialization
small GPU batches below crossover
```

---

# Step 17 — Final decision tree

Use this decision tree once the modes exist:

```text
Need exact biophysics or final validation?
    -> use exact double

Need large recruitment or threshold screening?
    -> use pseudo_double_split

Need extremely fast rough screening?
    -> use pseudo_double_effective

Pseudo result near threshold or ambiguous?
    -> rerun exact double on that subset

Pseudo error too high for target regime?
    -> calibrate pseudo parameters
    -> if still too high, test pseudo_double_modal
    -> otherwise use exact double with optimized batched/PCR solver
```

---

## Summary

The recommended path is not to replace exact double-cable immediately. Instead, add a controlled approximation layer:

```text
pseudo_double_effective -> fastest baseline
pseudo_double_split     -> likely best practical compromise
pseudo_double_modal     -> optional higher-fidelity surrogate
exact double            -> reference and refinement
```

For `Nx=30–100` and `B>500`, this is likely a better return-on-effort than trying to make every exact double-cable solve perfectly GPU-efficient. The pseudo modes restore the solver structure that GPUs handle well: scalar tridiagonal solves, pointwise auxiliary updates, compact observer outputs, and large batch axes.
