# AxonScope: Electrical Reduction from Double Cable to GPU-Compatible Single Cable

Status: standby as of 2026-06-16. Keep this as background research for a
possible future pseudo-double/pseudo-MRG validation pass. The active roadmap is
now exact double-cable GPU solver optimization, and reduced modes must remain
outside `BatchOptions.double_cable_block_solver` and `auto`.

## 0. Purpose

This document adds a concrete implementation plan for reducing AxonScope's exact double-cable model into one or more **GPU-compatible single-cable-like surrogate models**.

The target workload is:

- `Nx = 30–100` compartments per fiber.
- `B > 500` fibers, ideally `B_effective >= 1024–4096` after batching fibers, amplitudes, electrode configurations, or stimulation conditions.
- GPU execution with JAX/XLA.
- Large screening workloads: threshold, activation, recruitment, ranking of electrode configurations, and sensitivity studies.

The central idea is:

> Instead of making the exact double-cable 2x2 block-tridiagonal solver perfectly GPU-friendly, reduce the double-cable circuit to a single-cable or single-cable-plus-local-state surrogate that keeps the main electrical effect of myelin/periaxonal dynamics while using scalar tridiagonal solves and pointwise operations.

This is not intended to replace the exact double-cable model as the reference. The exact model should remain the validation target. The new reduced models should be treated as controlled approximations with measured error.

---

## 1. Why this may be better than optimizing the exact double-cable solver

The current exact double-cable solve has a difficult GPU profile:

- Each time step requires solving a 2x2 block-tridiagonal system.
- The current block-Thomas style solve has a sequential forward pass and sequential backward pass over `Nx`.
- For `Nx = 30–100`, the spatial dimension is too small to fully occupy a GPU by itself.
- The GPU only becomes efficient when the batch axis is large enough.
- The exact solver is mathematically elegant but not naturally shaped like the highly optimized scalar tridiagonal path used by single-cable.

A reduction to a single-cable-like system can restore the fast path:

```text
exact double cable:
    per time step: solve 2x2 block-tridiagonal system

reduced pseudo-single cable:
    per time step: pointwise effective-circuit updates
                   + scalar tridiagonal solve
```

This is attractive because:

- Scalar tridiagonal solve is already a good GPU primitive.
- Pointwise updates over `[B, Nx]` are very GPU-friendly.
- Memory traffic is lower.
- It allows compact observer-only workflows for threshold/recruitment.
- It allows a hybrid workflow: run reduced model on all fibers, then exact double-cable only on uncertain cases.

---

## 2. Model family to implement

Implement the reduction as a family of modes rather than a single approximation.

Recommended mode names:

```text
single                                  existing single-cable model
double                                  existing exact double-cable reference
pseudo_double_single_myelinated_chain   one-voltage NODE/MYSA/FLUT/STIN chain
pseudo_double_series                    simplest RC-series reduction
pseudo_double_schur_local               local Schur-complement reduction
pseudo_double_dynamic                   single-cable + local auxiliary myelin state
pseudo_double_calibrated                fitted/calibrated version of one of the above
```

Optional later mode:

```text
pseudo_double_modal             two scalar tridiagonal modes, if modal decoupling works well
```

Initial implementation priority:

1. `pseudo_double_single_myelinated_chain`
2. `pseudo_double_series`
3. `pseudo_double_schur_local`
4. `pseudo_double_dynamic`
5. `pseudo_double_calibrated`
6. optional `pseudo_double_modal`

---

## 3. Electrical intuition

A double-cable model represents at least two voltage domains, usually something like:

```text
Vi = intracellular / axonal potential
Vp = periaxonal or myelin-related potential
Ve = extracellular potential / imposed extracellular field
```

The exact double-cable solve keeps `Vi` and `Vp` as coupled unknowns.

A reduced single-cable model tries to eliminate the second electrical state and solve only one primary voltage, typically:

```text
Vm = Vi - Ve
```

or an equivalent axonal membrane voltage variable.

The eliminated periaxonal/myelin domain appears as an **effective admittance** seen by the axonal cable.

This effective admittance is generally not a simple constant. It may depend on:

- time step `dt`,
- membrane capacitance and conductance,
- myelin capacitance and conductance,
- periaxonal conductance/resistance,
- longitudinal periaxonal coupling,
- node/internode segment type,
- frequency content of the stimulation waveform,
- spatial mode.

Therefore, there are several levels of approximation.

---

## 4. Reduction level A: series membrane/myelin equivalent

### 4.1 Concept

If the axolemma and myelin can be approximated as two local admittances in series, then the effective admittance between axonal interior and extracellular space is:

```text
Y_ax(s) = G_ax + s C_ax
Y_my(s) = G_my + s C_my

Y_eff(s) = (Y_ax(s) * Y_my(s)) / (Y_ax(s) + Y_my(s))
```

For a purely capacitive approximation:

```text
C_eff = (C_ax * C_my) / (C_ax + C_my)
```

For a low-frequency conductance approximation:

```text
G_eff = (G_ax * G_my) / (G_ax + G_my)
```

This creates a single-cable model with effective segment parameters.

### 4.2 Why this is useful

This is the lowest-effort GPU-compatible reduction:

- one voltage variable,
- one scalar tridiagonal solve per time step,
- no 2x2 block solve,
- no auxiliary state,
- easy to compare against exact double-cable.

It is unlikely to reproduce all double-cable dynamics exactly, but it may be sufficient for:

- activation threshold,
- recruitment curves,
- electrode ranking,
- fast screening,
- approximate sensitivity maps.

### 4.3 Implementation formula

For each compartment or segment type, compute:

```python
C_eff = C_ax * C_my / (C_ax + C_my + eps)
G_eff = G_ax * G_my / (G_ax + G_my + eps)
```

Use small `eps` only for numerical safety. Do not let `eps` become a hidden fitting parameter unless explicitly placed under calibration.

Then build a single-cable parameter set:

```text
cm = C_eff
gm = G_eff
ra = original axonal axial resistance, or calibrated effective axial resistance
vext coupling = adjusted effective extracellular coupling
```

### 4.4 Segment-type handling

Do not use one global value for all compartments if the exact model has different segment classes.

Compute effective parameters per segment type:

```text
NODE
MYSA
FLUT
STIN
other supported segment types
```

Suggested initial behavior:

- At nodes: keep close to the original single-cable nodal axolemma behavior.
- At myelinated internodes: apply series axolemma/myelin equivalent.
- At paranodal regions: use either a segment-specific series equivalent or interpolate between node and internode behavior.

### 4.5 Files to touch

Likely files/directories:

```text
src/axonscope/models/               model parameter definitions
src/axonscope/solvers/              single-cable solver integration
src/axonscope/solvers/batch_kernels.py
src/axonscope/dispatcher/           mode dispatch
src/axonscope/performance.py        backend/runtime policy if needed
```

Exact filenames should be confirmed from the branch before implementation.

### 4.6 Acceptance criteria

Minimum acceptance:

```text
- Mode can run end-to-end on GPU.
- Uses scalar single-cable solve path, not 2x2 block solver.
- Supports Nx=30, 51, 64, 96 or realistic AxonScope geometries.
- Supports B >= 512.
- Produces stable voltage traces.
- Does not allocate double-cable state arrays unless requested for diagnostics.
```

Validation target:

```text
threshold error vs exact double-cable < 5–10% before calibration
speedup vs exact double-cable > 3x for B >= 1024
```

A stricter target after calibration:

```text
threshold error < 2–5%
recruitment curve error < 2–5%
speedup > 3x
```

---

## 5. Reduction level B: local Schur complement

### 5.1 Concept

At each implicit time step, the exact double-cable linear system can be written abstractly as:

```text
[Aii  Aip] [Vi] = [bi]
[Api  App] [Vp]   [bp]
```

Eliminating `Vp` gives:

```text
Vp = App^-1 (bp - Api Vi)
```

and therefore:

```text
(Aii - Aip App^-1 Api) Vi = bi - Aip App^-1 bp
```

This is an exact electrical reduction of the second variable.

The problem is that `App^-1` is generally dense if `App` contains longitudinal coupling. That means the exact reduced operator may be nonlocal and no longer a simple tridiagonal single-cable solve.

The GPU-compatible approximation is:

```text
App^-1 ≈ diag(App)^-1
```

or a low-order local approximation.

This gives a **local Schur complement**:

```text
Aeff ≈ Aii - Aip diag(App)^-1 Api
beff ≈ bi  - Aip diag(App)^-1 bp
```

The result remains scalar and local enough to keep the single-cable tridiagonal structure.

### 5.2 Why this may be better than series RC

The series RC reduction uses hand-derived equivalent circuit parameters. The local Schur reduction uses the actual discrete system coefficients from the exact double-cable model.

That means it can naturally include:

- actual `dt`,
- implicit integration coefficients,
- segment-specific conductance/capacitance,
- extracellular forcing terms,
- periaxonal/myelin coupling terms,
- geometry-dependent coefficients.

It is still approximate, but it is tied directly to the exact solver discretization.

### 5.3 Initial formula

Assume the exact block system uses local coupling between `Vi` and `Vp`, and tridiagonal axial coupling within each domain.

Let:

```text
Aii_diag[x]
Aii_lower[x], Aii_upper[x]
App_diag[x]
Aip_diag[x]
Api_diag[x]
bi[x]
bp[x]
```

Initial local Schur approximation:

```text
inv_App_local[x] = 1 / App_diag[x]

Aeff_diag[x]  = Aii_diag[x] - Aip_diag[x] * inv_App_local[x] * Api_diag[x]
Aeff_lower[x] = Aii_lower[x]
Aeff_upper[x] = Aii_upper[x]

beff[x] = bi[x] - Aip_diag[x] * inv_App_local[x] * bp[x]
```

If the signs in the actual block assembly differ, adapt the formula to the code convention. The implementation must be verified by comparing the reconstructed local Schur coefficients against the exact system for small synthetic cases.

### 5.4 Optional local axial correction

If ignoring `App` off-diagonal terms causes too much error, add a local correction.

First-order approximation:

```text
App = D + E
App^-1 ≈ D^-1 - D^-1 E D^-1
```

Then:

```text
Aeff ≈ Aii - Aip D^-1 Api + Aip D^-1 E D^-1 Api
```

This may introduce nearest-neighbor corrections into `Aeff_lower` and `Aeff_upper`, which can still preserve scalar tridiagonal form.

This is a good second version:

```text
pseudo_double_schur_local_v1 = diagonal App inverse only
pseudo_double_schur_local_v2 = first-order local axial correction
```

Do not implement v2 until v1 has been benchmarked and validated.

### 5.5 Implementation shape

The core builder should be pure and JAX-compatible:

```python
def build_pseudo_double_schur_local_coefficients(
    double_coeffs,
    *,
    correction_order: int = 0,
    eps: float = 1e-12,
):
    """Return scalar tridiagonal coefficients and RHS mapping.

    Inputs should come from the same coefficient assembly path used by the
    exact double-cable solver.
    """
```

Return:

```text
lower_eff: [..., Nx-1]
diag_eff:  [..., Nx]
upper_eff: [..., Nx-1]
rhs_eff:   [..., Nx]
```

or a coefficient object that can be passed directly into the existing single-cable batch solver.

### 5.6 Batch and time handling

Some coefficients may be static across time, while RHS terms depend on stimulation at time `t`.

Preferred design:

```text
precompute static effective coefficients once per geometry / dt / model
inside time scan: build rhs_eff[t] from Vext[t] and optional local state
solve scalar tridiagonal
```

Do not rebuild static Schur coefficients at every time step unless required.

### 5.7 Acceptance criteria

Minimum acceptance:

```text
- Same JIT/run path shape as single-cable.
- No 2x2 block-tridiagonal solve in pseudo mode.
- Coefficients are derived from exact double-cable coefficient assembly.
- Runs on B >= 512, Nx=30–100.
- Produces stable voltage traces.
```

Validation target:

```text
threshold error < 5% before calibration for common stimulation cases
speedup > 2–5x vs exact double-cable
```

If threshold error is high but ranking is preserved, the mode may still be useful for screening.

---

## 6. Reduction level C: dynamic pseudo-single cable

### 6.1 Concept

The exact reduction of double-cable to one variable may produce an effective admittance with memory:

```text
Y_eff(s) is not always just G_eff + s C_eff
```

A pure single-cable approximation forces the eliminated myelin/periaxonal dynamics into instantaneous effective parameters. That may lose important transient behavior.

A better compromise is:

```text
single-cable voltage solve + local auxiliary state
```

At each time step:

```text
1. update local auxiliary myelin/periaxonal state pointwise over [B, Nx]
2. build effective RHS for scalar single-cable solve
3. solve scalar tridiagonal system for Vm
4. update observers / compact outputs
```

This avoids the 2x2 block-tridiagonal solve but keeps a local memory effect.

### 6.2 Generic update form

Use an auxiliary state `u` per fiber and compartment:

```text
u[t+1] = alpha * u[t] + beta * Vm[t] + gamma * Vext[t] + delta * input[t]
```

Then use:

```text
rhs_eff[t] = rhs_single[t] + kappa * u[t+1]
```

or:

```text
G_eff_dynamic[t] = G0 + kappa_g * u[t]
C_eff_dynamic[t] = C0 + kappa_c * u[t]
```

The first version is easier and more stable: keep coefficients static and modify only the RHS.

### 6.3 Recommended first dynamic model

Start with a linear local relaxation model:

```text
tau_my * du/dt = Vm - u
```

Discretized implicitly or semi-implicitly:

```text
alpha = tau_my / (tau_my + dt)
beta  = dt / (tau_my + dt)

u_next = alpha * u_prev + beta * Vm_prev
```

Then:

```text
rhs_eff = rhs_base + kappa * (u_next - Vext_current)
```

The exact sign and coupling should be calibrated against the exact double-cable model.

### 6.4 Why pointwise auxiliary state is GPU-friendly

The dynamic model adds arrays of shape:

```text
u: [B, Nx]
```

and pointwise operations:

```text
u_next = alpha * u + beta * Vm
```

This is cheap compared with a block-tridiagonal solve. The only spatial solve remains scalar tridiagonal.

### 6.5 Where to use it

Use this mode if:

- `pseudo_double_series` is too inaccurate.
- `pseudo_double_schur_local` misses transient effects.
- threshold/recruitment error is mostly due to myelin/periaxonal delay or filtering.

Avoid using it as the first implementation because it introduces a new state and calibration parameters.

### 6.6 Acceptance criteria

```text
- Speed remains close to single-cable plus modest overhead.
- No 2x2 block solve.
- Threshold error improves over pure series/local-Schur mode.
- Calibration parameters are stable across B and Nx.
- Does not require full Vm trace output.
```

---

## 7. Optional reduction level D: modal two-tridiagonal model

### 7.1 Concept

In some homogeneous or quasi-homogeneous double-cable systems, a local transformation can approximately decouple the two electrical variables into two modes:

```text
mode_1 = a * Vi + b * Vp
mode_2 = c * Vi + d * Vp
```

If the transformed system becomes close to block-diagonal, then the double-cable solve can be approximated by two scalar tridiagonal solves:

```text
mode_1_next = tridiagonal_solve(...)
mode_2_next = tridiagonal_solve(...)
Vi, Vp = inverse_transform(mode_1_next, mode_2_next)
```

### 7.2 Pros and cons

Pros:

```text
+ more faithful than pure single-cable reduction
+ still uses scalar tridiagonal solves
+ potentially much more GPU-friendly than block-tridiagonal exact solve
```

Cons:

```text
- more complex than local Schur
- may not decouple well for strongly heterogeneous segment types
- still has two solves per time step
- more memory than pure pseudo-single modes
```

### 7.3 Recommendation

Do not implement this first. Use it only if:

- pure pseudo-single modes are too inaccurate,
- exact double-cable remains too slow,
- validation shows clear modal structure,
- two scalar solves are still faster than one block solve.

---

## 8. API design

### 8.1 User-facing mode selection

Add explicit modes rather than hidden flags:

```python
mode="single"
mode="double"
mode="pseudo_double_series"
mode="pseudo_double_schur_local"
mode="pseudo_double_dynamic"
```

Optional later:

```python
mode="pseudo_double_modal"
```

### 8.2 Configuration object

Add a configuration object for pseudo-double modes:

```python
@dataclass(frozen=True)
class PseudoDoubleCableConfig:
    method: Literal[
        "series",
        "schur_local",
        "dynamic",
        "modal",
    ]
    schur_correction_order: int = 0
    calibrate: bool = False
    calibration_profile: str | None = None
    dynamic_tau_scale: float = 1.0
    dynamic_coupling_scale: float = 1.0
    effective_axial_scale: float = 1.0
    effective_capacitance_scale: float = 1.0
    effective_conductance_scale: float = 1.0
```

Keep this config separate from exact double-cable parameters.

### 8.3 Dispatch behavior

Pseudo-double modes should dispatch like single-cable whenever possible:

```text
pseudo_double_* -> scalar single-cable batch kernel
```

Do not route them through the exact double-cable block solver.

### 8.4 Result metadata

Every result should include metadata that clearly identifies the approximation:

```text
model_mode: pseudo_double_schur_local
reference_model: double
calibrated: true/false
calibration_profile: name or hash
error_baseline_available: true/false
```

This is important so reduced-model outputs are never confused with exact double-cable outputs.

---

## 9. Implementation steps

## Step 1 — Add model mode enum / dispatch plumbing

Standby override: do not implement public mode or dispatch plumbing while this
plan is paused. Keep reduced-mode experiments in `benchmark/pseudo_double/`
unless `todo.md` explicitly resumes pseudo-double work.

### Goal

Make AxonScope recognize pseudo-double modes without changing solver behavior yet.

### Actions

1. Add pseudo-double mode strings to the model/mode validation layer.
2. Update dispatcher logic so these modes are accepted in batch mode.
3. Route pseudo-double modes to a placeholder implementation that raises a clear `NotImplementedError`.
4. Add tests verifying mode parsing and error message.

### Done when

```text
mode="pseudo_double_series" is accepted by high-level API
mode="pseudo_double_schur_local" is accepted by high-level API
both fail with intentional NotImplementedError until implemented
```

---

## Step 2 — Implement effective parameter builder for `pseudo_double_series`

### Goal

Construct a single-cable parameter set from double-cable parameters using local series admittance formulas.

### Actions

1. Create a new builder function:

```python
def build_series_reduced_single_cable_params(double_params, *, eps=1e-12):
    ...
```

2. Compute per-segment effective capacitance:

```python
C_eff = C_ax * C_my / (C_ax + C_my + eps)
```

3. Compute per-segment effective conductance:

```python
G_eff = G_ax * G_my / (G_ax + G_my + eps)
```

4. Preserve or optionally scale axial resistance:

```python
Rax_eff = Rax * effective_axial_scale
```

5. Return a parameter object compatible with the scalar single-cable solver.

### Tests

```text
- all effective values are finite
- C_eff <= min(C_ax, C_my) for positive capacitances
- G_eff <= min(G_ax, G_my) for positive conductances
- segment-type shapes match Nx
- JIT-compiles on CPU and GPU
```

### Done when

A pseudo-double series model can run using the scalar single-cable solver.

---

## Step 3 — Run first correctness checks for `pseudo_double_series`

### Goal

Ensure the reduced model is numerically stable and physically plausible.

### Actions

Run small cases:

```text
B = 1, 4
Nx = 30, 51, 64
Nt = short and medium
stimulation = zero, weak, strong
```

Check:

```text
- zero stimulation produces stable resting behavior
- weak stimulation produces bounded response
- strong stimulation does not create NaNs or infs
- no unexpected host/device transfers
```

### Done when

No NaNs, stable traces, and expected qualitative behavior.

---

## Step 4 — Validate `pseudo_double_series` against exact double-cable

### Goal

Measure error before spending time on more complex reductions.

### Actions

For each test case, run:

```text
exact double-cable
pseudo_double_series
single-cable baseline
```

Compare:

```text
peak Vm error
RMS Vm error at selected probes
activation time error
threshold error
recruitment curve error
ranking preservation across electrode configs
```

Use target workloads:

```text
Nx = 32, 51, 64, 96
B = 512, 1024, 2048
Nt = realistic stimulation duration
```

For detailed trace comparison, use smaller batches:

```text
B = 8 or 32
full Vm recording enabled
```

For performance comparison, use compact outputs:

```text
B >= 512
observer-only / threshold mode
```

### Done when

There is a table like:

```text
mode                     speedup   threshold_error   recruitment_error
single                   ...       ...               ...
pseudo_double_series      ...       ...               ...
exact_double              1.0x      0                 0
```

---

## Step 5 — Implement local Schur coefficient extraction

### Goal

Expose the exact double-cable block coefficients in a reusable form so that reduced models can be derived from the same linear system.

### Actions

1. Locate exact double-cable coefficient assembly.
2. Refactor it if needed so it can return a structured object:

```python
@dataclass
class DoubleCableBlockCoefficients:
    Aii_lower: Array
    Aii_diag: Array
    Aii_upper: Array
    App_lower: Array
    App_diag: Array
    App_upper: Array
    Aip_diag: Array
    Api_diag: Array
    # Optional additional coupling terms if present
```

3. Ensure this object can be created without launching the full solver.
4. Add tests comparing the structured coefficients to the original exact solver path.

### Done when

The exact solver and coefficient extraction produce equivalent linear systems for small debug cases.

---

## Step 6 — Implement `pseudo_double_schur_local_v1`

### Goal

Build a scalar tridiagonal reduced system using the diagonal local Schur approximation.

### Actions

Use:

```text
inv_App = 1 / App_diag
Aeff_diag = Aii_diag - Aip_diag * inv_App * Api_diag
Aeff_lower = Aii_lower
Aeff_upper = Aii_upper
```

For RHS:

```text
beff = bi - Aip_diag * inv_App * bp
```

Important: verify signs against actual coefficient conventions.

### Implementation function

```python
def build_schur_local_reduced_coefficients(
    coeffs: DoubleCableBlockCoefficients,
    rhs_i: Array,
    rhs_p: Array,
    *,
    eps: float = 1e-12,
):
    ...
```

### Tests

Use synthetic small systems:

```text
Nx = 3, 4, 5
random positive diagonally dominant coefficients
```

Compare:

```text
exact block solve result for Vi
local Schur approximation result for Vi
```

When `App` has no off-diagonal terms, local Schur should match exact Schur reduction closely or exactly, depending on signs and construction.

### Done when

The mode runs end-to-end and uses only scalar tridiagonal solve in the time loop.

---

## Step 7 — Validate `pseudo_double_schur_local_v1`

### Goal

Determine whether local Schur is materially better than series RC.

### Actions

Run the same validation matrix as Step 4.

Compare:

```text
single
pseudo_double_series
pseudo_double_schur_local
exact_double
```

Focus on:

```text
threshold error
recruitment error
activation timing
ranking preservation
runtime
memory allocation
```

### Go/no-go

Continue with Schur if:

```text
pseudo_double_schur_local is more accurate than pseudo_double_series
and speedup vs exact_double remains > 2x
```

If Schur is not more accurate, keep `series` as the simple fast mode and move to calibration or dynamic model.

---

## Step 8 — Add optional first-order Schur axial correction

### Goal

Account for ignored periaxonal/myelin axial coupling without losing scalar tridiagonal form.

### Actions

Implement correction order 1:

```text
App = D + E
App^-1 ≈ D^-1 - D^-1 E D^-1
```

This produces additional local corrections to diagonal and nearest-neighbor terms.

### Caution

This step is easy to get wrong. Implement only after v1 is validated.

### Tests

Use synthetic systems where `App` off-diagonal strength can be controlled:

```text
periaxial_strength = 0.0, 0.1, 0.3, 0.5
```

Check whether correction order 1 reduces error monotonically.

### Done when

Correction improves accuracy without instability and without large runtime overhead.

---

## Step 9 — Implement `pseudo_double_dynamic`

### Goal

Recover transient myelin/periaxonal effects using a local auxiliary state while keeping scalar tridiagonal solves.

### Actions

1. Add local state array:

```text
u: [B, Nx]
```

2. Add update inside time scan:

```python
u_next = alpha * u_prev + beta * vm_prev + gamma * vext_t
```

3. Build scalar RHS using `u_next`:

```python
rhs_eff = rhs_base + kappa * u_next
```

4. Solve scalar tridiagonal system.
5. Carry `u_next` through time scan.

### Parameters

Start with:

```text
tau_my
kappa
gamma
```

Derived or initialized from double-cable electrical parameters where possible.

### Tests

```text
- zero input stability
- JIT compatibility
- observer-only compatibility
- no dense full trace required
- runtime overhead vs pseudo_double_schur_local
```

### Done when

Dynamic mode improves threshold/recruitment accuracy enough to justify its extra state.

---

## Step 10 — Calibration layer

### Goal

Fit small correction factors so reduced modes match exact double-cable on the quantities that matter.

### Calibration targets

Primary:

```text
threshold error
recruitment curve error
```

Secondary:

```text
peak Vm error
activation timing error
RMS Vm error at selected probes
```

### Parameters to calibrate

For `pseudo_double_series`:

```text
effective_capacitance_scale
effective_conductance_scale
effective_axial_scale
effective_vext_coupling_scale
```

For `pseudo_double_schur_local`:

```text
schur_coupling_scale
app_inverse_scale
effective_axial_scale
vext_rhs_scale
```

For `pseudo_double_dynamic`:

```text
tau_scale
kappa_scale
gamma_scale
```

### Calibration dataset

Use a small representative dataset:

```text
fiber diameters: small, medium, large
Nx: 32, 51, 64, 96
stimulation waveforms: monophasic, biphasic, realistic pulse train if relevant
field configurations: near, far, asymmetric
amplitude range: subthreshold, threshold, suprathreshold
```

### Avoid overfitting

Do not fit a separate parameter per fiber unless absolutely necessary.

Prefer:

```text
one profile per fiber family / geometry class
or one global profile for a use case
```

### Done when

Calibration improves accuracy on held-out cases, not just the calibration set.

---

## Step 11 — Profiling and performance validation

### Goal

Confirm that pseudo modes actually use the GPU efficiently.

### Benchmark matrix

```text
Nx = 32, 51, 64, 96
B  = 512, 1024, 2048, 4096
Nt = 500, 1000, realistic long run
output = full trace, center probe, observer-only
```

Modes:

```text
single
exact_double
pseudo_double_series
pseudo_double_schur_local
pseudo_double_dynamic
```

Metrics:

```text
wall time with block_until_ready
compile time separately
kernel wait time
GPU utilization from trace
memory allocation
host-device transfer count
node-steps/s = B * Nt * Nx / runtime
speedup vs exact_double
accuracy vs exact_double
```

### JAX trace

Run real traces with:

```python
with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
    with jax.profiler.TraceAnnotation("pseudo_double_schur_local_B1024_Nx64"):
        out = run_case(...)
        jax.block_until_ready(out)
```

Compare pseudo modes to exact double-cable.

Expected trace pattern:

```text
pseudo modes:
    scalar tridiagonal solve + pointwise ops
    fewer heavy block-solver kernels
    lower memory pressure

exact double:
    block 2x2 solver / scan-heavy pattern
```

---

## Step 12 — Hybrid workflow

### Goal

Use reduced models for large-scale screening while retaining exact double-cable accuracy where it matters.

### Proposed workflow

```text
1. Run pseudo-double mode for all fibers, amplitudes, and electrode configs.
2. Identify uncertain fibers:
       - near threshold
       - close classification margin
       - large disagreement between pseudo modes
       - physiologically important subset
3. Re-run exact double-cable only on uncertain cases.
4. Merge results:
       - pseudo result for confident cases
       - exact result for uncertain cases
```

### Uncertainty heuristics

Mark a fiber/config as uncertain if:

```text
abs(predicted_margin_to_threshold) < margin
or pseudo_double_series and pseudo_double_schur_local disagree
or activation time is near boundary
or calibration model reports out-of-domain parameters
```

### Why this matters

Even if pseudo-double is not perfect, it can still reduce the number of exact double-cable runs by 5–20x.

This may be more valuable than making every exact double-cable solve 2x faster.

---

## 13. Validation report format

Each pseudo-double implementation should produce a validation report.

Required sections:

```text
1. Model description
2. Exact double-cable reference settings
3. Dataset / workload
4. Runtime benchmark
5. Accuracy benchmark
6. Failure cases
7. Recommended use cases
8. Cases where exact double-cable is still required
```

Recommended summary table:

```text
Mode                         Speedup   Threshold err   Recruit err   Notes
single                       ...       ...             ...           baseline
pseudo_double_series          ...       ...             ...           fastest surrogate
pseudo_double_schur_local     ...       ...             ...           coefficient-derived
pseudo_double_dynamic         ...       ...             ...           best transient match
exact_double                  1.0x      0               0             reference
```

---

## 14. Failure modes to watch for

### 14.1 Wrong extracellular coupling

A reduced model can match passive membrane behavior but get extracellular drive wrong.

Test with:

```text
- uniform extracellular potential
- linear extracellular gradient
- localized extracellular stimulation
- reversed polarity
```

A uniform extracellular shift should not create artificial activation if the model is gauge-invariant.

### 14.2 Incorrect threshold ranking

Even if average threshold error is small, the model may mis-rank electrode configurations.

Always evaluate:

```text
Spearman rank correlation
Kendall rank correlation
pairwise ranking flips
```

### 14.3 Segment boundary artifacts

Series reductions may create discontinuities at node/paranode/internode boundaries.

Inspect:

```text
Vm profiles over x
peak depolarization location
activation initiation location
```

### 14.4 Calibration overfit

A calibrated pseudo model can look excellent on one waveform and fail on another.

Keep a held-out validation set.

### 14.5 Hidden memory regressions

Dynamic modes may accidentally store `u[B, Nt, Nx]` instead of only carrying `u[B, Nx]`.

Ensure the time scan returns only compact observer outputs unless full trace is explicitly requested.

---

## 15. Recommended order of work

Use this sequence:

```text
1. Add mode plumbing and metadata.
2. Implement pseudo_double_series.
3. Validate series model against exact double-cable.
4. Profile speed and memory.
5. Extract exact double-cable block coefficients.
6. Implement pseudo_double_schur_local_v1.
7. Validate Schur v1 against series and exact.
8. Add calibration layer if needed.
9. Add pseudo_double_dynamic only if transient errors remain important.
10. Implement hybrid pseudo + exact workflow.
11. Optionally test modal/two-tridiagonal model.
```

Do not start with the most complex model. The fastest useful surrogate is more valuable than a sophisticated surrogate that is difficult to validate.

---

## 16. Practical go/no-go thresholds

A pseudo-double mode is useful if it satisfies at least one of these:

### Screening-quality surrogate

```text
speedup vs exact_double >= 5x
threshold error <= 5–10%
ranking correlation high
```

### Production-quality surrogate

```text
speedup vs exact_double >= 3x
threshold error <= 2–5%
recruitment curve error <= 2–5%
works across held-out cases
```

### Hybrid pre-filter

```text
speedup vs exact_double >= 5x
false-negative activation rate near zero after uncertainty margin
exact rerun fraction <= 10–30%
```

If none of the pseudo modes meet these thresholds, continue optimizing exact double-cable instead.

---

## 17. Minimal first PR scope

The first PR should be small and testable.

Recommended first PR:

```text
Title: Add pseudo_double_single_myelinated_chain single-cable reduction mode
```

Contents:

```text
- mode enum / parsing
- config object
- NODE/MYSA/FLUT/STIN single-line segment builder
- effective per-segment parameter builder
- segment-specific extracellular coupling vector
- dispatch to scalar single-cable solver
- metadata tagging
- unit tests for segment layout and effective parameters
- small correctness test vs exact double-cable
- one benchmark script for B=512/1024, Nx=51/64
```

Do not include:

```text
- dynamic auxiliary state
- local Schur correction
- calibration fitting
- hybrid workflow
```

Those should be later PRs.

---

## 18. Minimal benchmark script

Create:

```text
benchmarks/bench_pseudo_double_reduction.py
```

Suggested CLI:

```bash
python benchmarks/bench_pseudo_double_reduction.py \
  --modes single,double,pseudo_double_series,pseudo_double_schur_local \
  --B 512,1024,2048 \
  --Nx 32,51,64,96 \
  --Nt 1000 \
  --output observer \
  --device gpu
```

Output CSV columns:

```text
mode
B
Nx
Nt
device
compile_time_s
run_time_s
peak_memory_mb
node_steps_per_s
speedup_vs_double
threshold_error_pct
recruitment_error_pct
rank_corr
nan_count
```

This benchmark should separate compile and run time.

Always synchronize GPU measurements using:

```python
jax.block_until_ready(out)
```

---


## 19. Added mode: `pseudo_double_single_myelinated_chain`

### 19.1 Motivation

A very practical reduction is to keep the **myelinated geometry and segment taxonomy** of the double-cable model, but collapse the electrical state to a single cable:

```text
NODE -- MYSA -- FLUT -- STIN -- STIN -- ... -- FLUT -- MYSA -- NODE -- ...
```

This directly addresses the question:

> Can we model `NODE -- myelin -- myelin -- NODE -- myelin -- ...` as a single cable and connect `Vext` to it like the current single-cable solver?

The answer is yes, as an approximation. It should be implemented as a new surrogate mode, not as a replacement for `double`:

```text
mode = "pseudo_double_single_myelinated_chain"
```

This is the simplest GPU-friendly pseudo-double model because it uses:

```text
one voltage variable per compartment
one scalar tridiagonal solve per time step
pointwise active/passive membrane updates
segment-specific effective parameters
segment-specific Vext coupling
```

It avoids the exact double-cable unknowns:

```text
Vi = intracellular / axonal potential
Vp = periaxonal or under-myelin potential
```

and instead solves only a single cable state, usually equivalent to `Vi` or `Vm = Vi - alpha(x) * Vext`.

### 19.2 Keep MYSA and FLUT explicitly

Do **not** collapse all myelinated regions into one generic `MYELIN` segment type.

The implementation should preserve the segment classes used by MRG-like double-cable models:

```text
NODE    node of Ranvier, active channels
MYSA    paranodal myelin attachment segment
FLUT    paranodal fluted segment
STIN    stereotyped internodal segment
```

The chain should therefore look like:

```text
NODE
  MYSA
  FLUT
  STIN
  STIN
  ...
  STIN
  FLUT
  MYSA
NODE
```

Keeping `MYSA` and `FLUT` matters because these regions can have different geometry, capacitance, leakage, periaxonal properties, and extracellular coupling than the central internode. Even if the first implementation uses simple fitted parameters, the API should expose per-segment-type parameters from day one.

### 19.3 Electrical interpretation

The exact double-cable view is approximately:

```text
axoplasm / intracellular line
      |
 axolemma
      |
periaxonal space / under-myelin voltage
      |
 myelin impedance
      |
extracellular potential Vext
```

The single-chain approximation collapses this into:

```text
axoplasm / intracellular line
      |
 effective membrane + myelin impedance
      |
 effective extracellular drive alpha(x) * Vext
```

So every compartment has one voltage state, but its parameters depend on its segment type:

```text
NODE: active membrane parameters
MYSA: passive effective paranodal attachment parameters
FLUT: passive effective paranodal fluted-region parameters
STIN: passive effective internodal parameters
```

This is equivalent to assuming that the under-myelin/periaxonal voltage can be absorbed into effective local parameters rather than solved as an independent spatially coupled state.

### 19.4 Why this is not exactly the double cable

In the exact double-cable model, the axolemma voltage under myelin is closer to:

```text
V_axolemma = Vi - Vperiaxonal
```

while the myelin voltage is closer to:

```text
V_myelin = Vperiaxonal - Vext
```

The single-chain approximation removes `Vperiaxonal` and uses:

```text
V_effective = Vi - alpha(segment_type, x) * Vext
```

This makes the model much faster, but it cannot reproduce:

```text
periaxonal voltage
explicit myelin current
spatial propagation in the periaxonal space
exact transmyelin dynamics
detailed paranodal/periaxonal effects
```

Therefore, it should be validated primarily against use-case metrics:

```text
threshold
activation yes/no
recruitment curve
spike initiation site
conduction velocity
peak nodal Vm
time-to-spike
ranking of electrode/fiber configurations
```

rather than only pointwise internodal Vm error.

### 19.5 Default segment parameter table

Initial model parameters should be stored as arrays of length `Nx`, built from segment classes.

Suggested structure:

```python
@dataclass(frozen=True)
class PseudoDoubleSingleChainConfig:
    cm_scale_node: float = 1.0
    cm_scale_mysa: float = 1.0
    cm_scale_flut: float = 1.0
    cm_scale_stin: float = 1.0

    gleak_scale_node: float = 1.0
    gleak_scale_mysa: float = 1.0
    gleak_scale_flut: float = 1.0
    gleak_scale_stin: float = 1.0

    axial_resistance_scale: float = 1.0

    vext_alpha_node: float = 1.0
    vext_alpha_mysa: float = 1.0
    vext_alpha_flut: float = 1.0
    vext_alpha_stin: float = 1.0

    use_series_capacitance: bool = True
    use_series_leak: bool = True
    active_nodes_only: bool = True
```

Per-compartment arrays:

```text
segment_type[Nx]      int enum: NODE/MYSA/FLUT/STIN
cm_eff[Nx]            effective capacitance
geff[Nx]              effective passive conductance/leak
ra_eff_edges[Nx-1]    effective axial coupling between neighboring compartments
vext_alpha[Nx]        segment-specific extracellular coupling
is_active[Nx]         true only at NODE by default
```

### 19.6 Parameter construction

For nodes, start from the existing single/double active nodal parameters:

```text
cm_eff[NODE] = cm_node
geff[NODE] = leak_node + active-channel linearization terms
is_active[NODE] = True
vext_alpha[NODE] = 1.0
```

For MYSA, FLUT, and STIN, compute an effective passive membrane/myelin impedance.

A simple starting point is the series capacitance approximation:

```text
C_eff = (C_ax * C_my) / (C_ax + C_my + eps)
```

and, if meaningful for the available parameters, a series leak/conductance approximation:

```text
G_eff = (G_ax * G_my) / (G_ax + G_my + eps)
```

In practice, `G_eff` should probably be calibrated more aggressively than `C_eff`, because conductance/leak strongly affects thresholds and afterpotentials.

Recommended initial defaults:

```text
NODE:
    active = true
    C_eff = existing nodal capacitance
    G_eff = existing nodal leak/active terms
    alpha_vext = 1.0

MYSA:
    active = false
    C_eff = series(C_ax_mysa, C_my_mysa)
    G_eff = fitted or series(G_ax_mysa, G_my_mysa)
    alpha_vext = fitted, initial 1.0

FLUT:
    active = false
    C_eff = series(C_ax_flut, C_my_flut)
    G_eff = fitted or series(G_ax_flut, G_my_flut)
    alpha_vext = fitted, initial 1.0

STIN:
    active = false
    C_eff = series(C_ax_stin, C_my_stin)
    G_eff = fitted or series(G_ax_stin, G_my_stin)
    alpha_vext = fitted, initial 1.0
```

### 19.7 Extracellular stimulation coupling

The first implementation should support two modes.

#### Mode A: direct single-cable coupling

```text
alpha_vext[NODE] = 1.0
alpha_vext[MYSA] = 1.0
alpha_vext[FLUT] = 1.0
alpha_vext[STIN] = 1.0
```

This answers the simple question: "what happens if we just connect `Vext` like the current single cable?"

#### Mode B: segment-specific fitted coupling

```text
alpha_vext[NODE] = 1.0
alpha_vext[MYSA] = alpha_mysa
alpha_vext[FLUT] = alpha_flut
alpha_vext[STIN] = alpha_stin
```

This is likely needed because the exact double-cable model does not transmit `Vext` to the axolemma identically in every under-myelin region. The myelin and periaxonal space filter the drive.

The RHS construction should therefore use:

```python
vext_eff = vext_alpha[None, :] * vext_t  # [B, Nx]
```

or, for factorized stimulation:

```python
vext_eff_t = waveform[t] * (vext_alpha[None, :] * footprint_BxNx)
```

### 19.8 Geometry and segmentation builder

Add a builder that creates a one-line single-cable geometry from the exact double-cable geometry.

Suggested function:

```python
def build_pseudo_double_single_chain_geometry(double_geometry) -> SingleCableGeometry:
    """Return a single-cable geometry preserving NODE/MYSA/FLUT/STIN layout."""
```

Required behavior:

```text
- preserve node positions
- preserve MYSA/FLUT/STIN order
- preserve compartment lengths where possible
- preserve diameter/radius where possible
- expose segment_type[Nx]
- produce single-cable axial edge parameters
- produce per-compartment passive/active parameter arrays
```

Do not hard-code only one MRG layout. Support the current AxonScope geometry builder and allow future variations:

```text
n_stin_per_internode
presence/absence of MYSA
presence/absence of FLUT
custom internode discretization
custom segment lengths
```

### 19.9 Solver routing

The mode should route to the existing single-cable fast path:

```text
pseudo_double_single_myelinated_chain
    -> build effective single-cable parameters
    -> build vext_eff using alpha_vext
    -> call existing single-cable batch kernel
    -> use scalar tridiagonal solve
```

It should **not** call:

```text
solve_block_tridiagonal_2x2_scalar
```

It should not allocate:

```text
Vi[B, Nt, Nx] and Vperiaxonal[B, Nt, Nx]
```

unless debug output explicitly requests surrogate-internal diagnostics.

### 19.10 Implementation files

Likely files/modules to touch:

Standby override: do not touch these core modules for pseudo-double while this
plan is paused.

```text
src/axonscope/models/              add pseudo-double config/model type if such module exists
src/axonscope/geometry/            add or extend segment layout builder
src/axonscope/solvers/             route pseudo mode to single-cable solver
src/axonscope/solvers/batch_kernels.py
src/axonscope/dispatcher/execution.py
src/axonscope/performance.py       include backend/estimator rules for pseudo mode
benchmarks/                        add pseudo-chain benchmarks
tests/                             add layout, parameter, numerical and speed tests
```

If the current codebase does not have exactly these modules, keep the same separation of concerns:

```text
geometry construction
parameter reduction
solver dispatch
benchmarking
validation
```

### 19.11 Step-by-step implementation checklist

#### Step 1: Add the mode name

Add:

```text
pseudo_double_single_myelinated_chain
```

to the public mode enum / parser / dispatcher.

Acceptance criterion:

```text
The mode can be selected from the public API but raises NotImplementedError with a useful message until the builder is connected.
```

#### Step 2: Add segment enum support

Create or reuse a segment enum:

```python
class SegmentType(IntEnum):
    NODE = 0
    MYSA = 1
    FLUT = 2
    STIN = 3
```

Acceptance criterion:

```text
A generated fiber exposes segment_type[Nx] and counts for NODE/MYSA/FLUT/STIN.
```

#### Step 3: Build the single-line myelinated geometry

Implement the conversion from double geometry to one-line geometry.

Acceptance criterion:

```text
For an MRG-like fiber, the sequence around each internode is:
NODE, MYSA, FLUT, STIN..., FLUT, MYSA, NODE
```

#### Step 4: Build effective per-segment parameters

Compute:

```text
cm_eff[Nx]
geff[Nx]
ra_eff_edges[Nx-1]
vext_alpha[Nx]
is_active[Nx]
```

Acceptance criterion:

```text
All arrays have stable static shapes, finite values, and no negative capacitance/conductance.
```

#### Step 5: Route to the scalar single-cable solver

Reuse the existing single-cable solver path.

Acceptance criterion:

```text
A pseudo-chain run launches the same scalar tridiagonal solve path as single-cable, not the double-cable block solver.
```

#### Step 6: Add direct Vext coupling test

Run with:

```text
alpha_vext = 1.0 for all segments
```

Acceptance criterion:

```text
The run produces stable voltages and no NaNs for B=512, Nx in {32, 51, 64, 96}, Nt=1000.
```

#### Step 7: Add fitted Vext coupling

Expose:

```text
alpha_mysa
alpha_flut
alpha_stin
```

Acceptance criterion:

```text
Changing alpha values affects threshold and Vm in expected directions and remains JIT-compatible.
```

#### Step 8: Validate against exact double-cable

Compare:

```text
threshold error
activation agreement
spike initiation site
conduction velocity
peak nodal Vm
time-to-spike
recruitment curve
```

Acceptance criterion:

```text
baseline direct coupling produces a measured error report; no claim of accuracy is made until calibration is complete.
```

#### Step 9: Calibrate segment parameters

Fit the smallest parameter set first:

```text
cm_scale_mysa, cm_scale_flut, cm_scale_stin
geff_scale_mysa, geff_scale_flut, geff_scale_stin
alpha_mysa, alpha_flut, alpha_stin
optional axial_resistance_scale
```

Recommended first fit objective:

```text
weighted threshold error + recruitment error + spike timing error
```

Do not overfit full Vm traces before checking threshold/recruitment metrics.

#### Step 10: Add hybrid pseudo + exact workflow

Use pseudo-chain as a screen:

```text
run pseudo_chain on all fibers/amplitudes
flag uncertain fibers near threshold
rerun exact double only on uncertain fibers
merge results
```

Acceptance criterion:

```text
false-negative activation rate near zero with an uncertainty margin
exact double rerun fraction <= 10–30%
overall speedup >= 3–5x vs all-exact double
```

### 19.12 Unit tests

Add tests for:

```text
segment sequence correctness
per-segment parameter array shape
finite positive capacitance/conductance
active nodes only by default
Vext alpha broadcasting
single-cable solver routing
JIT compatibility
batch compatibility for B > 500
```

Example tests:

```python
def test_pseudo_chain_keeps_mysa_flut_stin():
    geom = build_pseudo_double_single_chain_geometry(example_double_geom)
    assert SegmentType.MYSA in geom.segment_type
    assert SegmentType.FLUT in geom.segment_type
    assert SegmentType.STIN in geom.segment_type


def test_pseudo_chain_routes_to_single_solver():
    # Use a spy/counter or metadata flag to verify scalar solver path.
    result = run(mode="pseudo_double_single_myelinated_chain", ...)
    assert result.metadata["solver_family"] == "single_tridiagonal"
```

### 19.13 Benchmark matrix

Use the target workload:

```text
Nx = 32, 51, 64, 96
B = 512, 1024, 2048, 4096
Nt = 1000 or the real stimulation duration
output = observer-only, center trace, full trace
Iinj = None
Vext = dense and factorized variants
```

Compare:

```text
single
double
pseudo_double_single_myelinated_chain with alpha=1
pseudo_double_single_myelinated_chain calibrated
pseudo_double_series
pseudo_double_schur_local
```

Metrics:

```text
runtime_s
speedup_vs_double
node_steps_per_s
peak_memory_mb
threshold_error_pct
activation_agreement
recruitment_curve_error
rank_correlation
exact_rerun_fraction in hybrid workflow
```

### 19.14 Expected outcome

Expected performance:

```text
pseudo_double_single_myelinated_chain should be close to single-cable speed,
because it uses the same scalar tridiagonal solver family.
```

Expected fidelity:

```text
uncalibrated alpha=1 version: useful diagnostic baseline, not guaranteed accurate
calibrated version: likely useful for threshold/recruitment screening
exact double: remains reference for final validation and uncertain cases
```

Success threshold:

```text
speedup vs exact double >= 3x, ideally >= 5x
threshold error <= 2–5% after calibration
recruitment curve error <= 2–5% after calibration
false-negative activation rate near zero in hybrid workflow
```

### 19.15 Sources and rationale for this mode

The rationale for preserving `NODE/MYSA/FLUT/STIN` while collapsing to a single voltage variable comes from three observations in the literature:

1. The MRG model is the reference myelinated axon model family for many stimulation studies and uses detailed myelinated segment structure with finite-impedance myelin. McIntyre, Richardson, and Grill describe geometrically and electrically detailed mammalian motor nerve fiber models and ModelDB hosts the associated PNS myelinated axon implementation.
2. Richardson, McIntyre, and Grill explicitly compared three myelin representations under extracellular stimulation: perfectly insulating single cable, finite-impedance single cable, and finite-impedance double cable. This supports treating a finite-impedance single-cable myelin representation as a meaningful intermediate model, not just a numerical hack.
3. Johnson et al. later replaced the McIntyre et al. double-cable model with a single-cable model using axon and myelin capacitance in series, with compensations to preserve relevant behavior. This is a direct precedent for a double-to-single electrical reduction.

Recommended references to cite in code comments or documentation:

```text
[1] McIntyre CC, Richardson AG, Grill WM. Modeling the excitability of mammalian nerve fibers: influence of afterpotentials on the recovery cycle. Journal of Neurophysiology, 2002. DOI: 10.1152/jn.00353.2001
    https://journals.physiology.org/doi/full/10.1152/jn.00353.2001
    https://pubmed.ncbi.nlm.nih.gov/11826063/

[2] ModelDB implementation of McIntyre et al. PNS myelinated axon model.
    https://modeldb.science/showmodel?model=3810

[3] Richardson AG, McIntyre CC, Grill WM. Modelling the effects of electric fields on nerve fibres: influence of the myelin sheath. Medical & Biological Engineering & Computing, 2000.
    https://link.springer.com/article/10.1007/BF02345014

[4] Grill WM, Richardson AG, McIntyre CC. Influence of the myelin sheath on excitation properties of nerve fibers. Proceedings of the IEEE EMBS, 2000. DOI: 10.1109/IEMBS.2000.900380
    https://scholars.duke.edu/individual/pub774757

[5] Johnson C. et al. Minimizing the caliber of myelinated axons by means of nodal constrictions. 2015. The paper states that the McIntyre et al. double-cable model was replaced by a single-cable model with axon and myelin capacitance in series.
    https://neurofilament.osu.edu/wp-content/uploads/2017/08/Johnson_2015.pdf
```

Implementation note:

```text
The pseudo-chain model should cite [3] as the conceptual justification for a finite-impedance single-cable myelin representation, [1]/[2] for the MRG-style segment taxonomy, and [5] for the series-capacitance double-to-single reduction precedent.
```

---
## 20. Summary recommendation

Yes, it is feasible to simplify or transform the double-cable model into a GPU-compatible single-cable-like model.

The most practical route is not a single magical exact transformation. It is a controlled ladder of approximations:

```text
1. NODE/MYSA/FLUT/STIN single-line myelinated chain
2. series RC effective single-cable
3. local Schur-complement pseudo-single cable
4. dynamic pseudo-single cable with local auxiliary state
5. optional calibrated version
6. optional modal two-tridiagonal version
```

For AxonScope's target workload (`Nx=30–100`, `B>500`), the best near-term bet is:

```text
pseudo_double_single_myelinated_chain first,
then pseudo_double_series / pseudo_double_schur_local,
then calibration,
then pseudo_double_dynamic only if needed.
```

The exact double-cable model should remain the reference. The pseudo models should be judged by their ability to preserve threshold/recruitment decisions and electrode/fiber ranking while delivering large GPU speedups.
