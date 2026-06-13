# Hybrid JAX Framework for Single-Cable, Double-Cable, and Surrogate Nerve Fiber Modeling

## 1. Objective

This document proposes a hybrid nerve-fiber simulation framework built around an existing JAX solver capable of solving both:

- single-cable equations,
- double-cable equations.

The goal is to preserve high-fidelity biophysics in critical regions while reducing computational cost elsewhere through:

- a local surrogate model,
- a single-cable buffer region,
- adaptive spatial partitioning,
- activating-function-based compartment selection,
- and automatic learning from regions where explicit and surrogate models are evaluated simultaneously.

The intended hierarchy is:

```text
Critical region       -> double-cable model
Transition region     -> single-cable model
Far-field region      -> surrogate or reduced model
```

This avoids a direct and potentially unstable transition between a detailed double-cable formulation and a coarse surrogate.

---

## 2. Core Modeling Principle

The spatial cable physics should remain explicit and globally coupled.

A generic cable equation can be written as:

$$
C_m \frac{\partial V_m}{\partial t}
=
\frac{1}{r_a}
\frac{\partial^2 V_m}{\partial x^2}
-
I_{\mathrm{mem}}
+
I_{\mathrm{ext}}
$$

The hybrid framework changes the local membrane or cable representation depending on the spatial region.

The surrogate may approximate:

$$
I_{\mathrm{mem}}
\approx
\mathcal{S}
\left(
V_m,
\text{history},
I_{\mathrm{axial}},
I_{\mathrm{ext}},
\theta
\right)
$$

where $\theta$ contains morphological and physiological parameters.

The explicit cable coupling should be retained whenever possible because it preserves:

- axial-current propagation,
- fiber geometry,
- extracellular stimulation along the axon,
- orthodromic and antidromic propagation,
- multi-site initiation,
- spike collisions,
- and continuity between compartments.

---

## 3. Global Static Grid

A practical JAX implementation should use a fixed global compartment grid:

```python
x.shape == (n_compartments,)
```

Each compartment is assigned a fidelity level:

```python
SURROGATE = 0
SINGLE_CABLE = 1
DOUBLE_CABLE = 2
```

The fidelity map has a fixed shape:

```python
model_type.shape == (n_compartments,)
```

Using static array shapes is important for:

- `jax.jit`,
- `jax.lax.scan`,
- GPU execution,
- predictable compilation,
- and differentiability.

A first implementation may evaluate several candidate updates and select the appropriate one with masks:

```python
next_state = jnp.where(
    double_mask[..., None],
    double_next_state,
    jnp.where(
        single_mask[..., None],
        single_next_state,
        surrogate_next_state,
    ),
)
```

This is not always the most computationally efficient solution, but it is simple, robust, and compatible with JAX compilation.

Once validated, the implementation can be optimized using fixed-size windows, grouped compartments, or specialized kernels.

---

## 4. Activating-Function-Based Compartmentalization

The activating function provides a physics-informed way to initialize the fidelity map.

For an extracellular potential $V_e(x)$, a common approximation is:

$$
f_{\mathrm{act}}(x)
\propto
-
\frac{\partial^2 V_e}{\partial x^2}
$$

In a discrete uniform grid:

```python
dx = x[1] - x[0]

af_inner = -(
    ve[2:] - 2.0 * ve[1:-1] + ve[:-2]
) / dx**2

af = jnp.pad(af_inner, (1, 1))
```

A region can be classified as critical when:

$$
|f_{\mathrm{act}}(x)| > \tau_f
$$

or when its spatial gradient is large:

$$
\left|
\frac{\partial f_{\mathrm{act}}}{\partial x}
\right|
>
\tau_g
$$

In JAX:

```python
af_grad = jnp.gradient(af, dx)

critical = (
    (jnp.abs(af) > af_threshold)
    | (jnp.abs(af_grad) > grad_threshold)
)
```

The critical mask should then be spatially dilated:

```python
def dilate_mask(mask, radius):
    kernel = jnp.ones(2 * radius + 1)
    expanded = jnp.convolve(
        mask.astype(jnp.float32),
        kernel,
        mode="same",
    )
    return expanded > 0
```

Example:

```python
hf_mask = dilate_mask(critical, radius=5)
```

The safety margin is important because the activating function is only an indicator of likely stimulation sensitivity. It does not capture all nonlinear membrane effects, refractory states, or propagation phenomena.

The activating function should therefore be used as an initialization mechanism, not as the only criterion during the simulation.

---

## 5. Three-Level Fidelity Hierarchy

The existing single-cable and double-cable solvers provide a natural intermediate hierarchy.

A useful spatial arrangement is:

```text
surrogate
    |
single-cable buffer
    |
double-cable core
    |
single-cable buffer
    |
surrogate
```

The masks may be constructed as follows:

```python
double_mask = dilate_mask(core_mask, radius=double_radius)

single_region = dilate_mask(
    core_mask,
    radius=single_buffer_radius,
)

single_mask = single_region & ~double_mask
surrogate_mask = ~(double_mask | single_mask)
```

This has several advantages:

- double-cable physics is retained near stimulation and spike initiation;
- the single-cable region provides a smoother physical transition;
- the surrogate is used only where the dynamics are less critical;
- interface artifacts are reduced;
- the single-cable solver can serve as a lower-cost teacher for some regions.

---

## 6. Shared Global Axial Coupling

The safest interface strategy is to avoid solving disconnected subdomains with independent boundary conditions.

Instead:

1. maintain one global voltage representation;
2. compute axial currents globally;
3. pass the same axial-current information to each local model;
4. switch only the local membrane or cable update.

Conceptually:

```python
i_axial = compute_global_axial_currents(
    global_state,
    geometry,
)
```

Then:

```python
double_update = double_cable_step(
    state=state,
    i_axial=i_axial,
    i_ext=i_ext,
    params=double_params,
)

single_update = single_cable_step(
    state=state,
    i_axial=i_axial,
    i_ext=i_ext,
    params=single_params,
)

surrogate_update = surrogate_step(
    state=state,
    i_axial=i_axial,
    i_ext=i_ext,
    params=surrogate_params,
)
```

This enforces axial-current consistency by construction.

At an ideal interface, the following quantities should remain continuous:

$$
V_m^{\mathrm{left}}
=
V_m^{\mathrm{right}}
$$

and

$$
I_{\mathrm{axial}}^{\mathrm{left}}
=
I_{\mathrm{axial}}^{\mathrm{right}}
$$

A common global axial operator is generally preferable to artificial Dirichlet or Neumann coupling between separate solvers.

---

## 7. Common Hybrid State

Single-cable, double-cable, and surrogate models may use different state variables.

A practical solution is to define a global state containing the union of all required variables.

For example:

```python
from typing import NamedTuple
import jax

class HybridState(NamedTuple):
    vm: jax.Array
    vi: jax.Array
    ve: jax.Array
    v_myelin: jax.Array
    gates: jax.Array
    latent: jax.Array
    refractory: jax.Array
```

Possible usage:

- double-cable model:
  - `vm`,
  - `vi`,
  - `v_myelin`,
  - `gates`;

- single-cable model:
  - `vm`,
  - `gates`;

- surrogate:
  - `vm`,
  - `latent`;

- event-based outer model:
  - `refractory`,
  - spike-arrival state.

Unused fields remain present to preserve a stable JAX PyTree structure.

---

## 8. Local Surrogate Model

The surrogate should preferably be local and shared across compartments.

A local surrogate can be written as:

$$
\left(
V_{m,i}^{t+\Delta t},
\mathbf{z}_i^{t+\Delta t}
\right)
=
\mathcal{S}
\left(
V_{m,i}^{t},
\mathbf{z}_i^t,
I_{\mathrm{axial},i}^{t},
I_{\mathrm{ext},i}^{t},
\theta_i
\right)
$$

where $\mathbf{z}_i$ is a latent dynamic state.

A possible implementation is:

```python
def surrogate_cell_step(
    params,
    vm,
    latent,
    i_axial,
    i_ext,
    morphology_features,
    dt,
):
    inputs = jnp.concatenate([
        vm[None],
        latent,
        i_axial[None],
        i_ext[None],
        morphology_features,
    ])

    output = network_apply(params, inputs)

    dvm = output[0]
    next_latent = output[1:]

    next_vm = vm + dt * dvm

    return next_vm, next_latent
```

It can then be vectorized:

```python
batched_surrogate_step = jax.vmap(
    surrogate_cell_step,
    in_axes=(None, 0, 0, 0, 0, 0, None),
)
```

Useful input features include:

- membrane voltage,
- latent state,
- axial current,
- extracellular drive,
- compartment type,
- fiber diameter,
- internodal length,
- membrane parameters,
- temperature,
- local geometry.

The raw compartment index should generally not be used as an input unless there is a strong physical reason, because it may reduce generalization across geometries.

---

## 9. Fixed and Adaptive Fidelity Maps

### 9.1 Fixed map

The first implementation should use a static map derived before the simulation.

Example:

```text
activating function
        |
double-cable core
        |
single-cable buffer
        |
surrogate exterior
```

This is easier to debug and benchmark.

### 9.2 Adaptive map

The fidelity map can later be updated during simulation using dynamic criteria.

A compartment can be marked as critical when:

```python
dynamic_critical = (
    (state.vm > voltage_alert_threshold)
    | (uncertainty > uncertainty_threshold)
    | (disagreement > disagreement_threshold)
)
```

The complete requested high-fidelity region becomes:

```python
requested_hf = static_af_mask | dynamic_critical
requested_hf = dilate_mask(
    requested_hf,
    radius=buffer_radius,
)
```

Useful promotion criteria include:

- strong depolarization,
- approaching spike front,
- high surrogate uncertainty,
- disagreement between explicit and surrogate predictions,
- proximity to activation threshold,
- possible conduction block,
- spike collision,
- branch point,
- abrupt geometry change.

---

## 10. Promotion and Demotion Hysteresis

A fidelity mask should not switch immediately at every threshold crossing.

Abrupt switching may create:

- state discontinuities,
- numerical oscillations,
- artificial reflections,
- duplicated spikes,
- or unstable repeated promotion and demotion.

A persistent fidelity state can be used:

```python
class FidelityState(NamedTuple):
    level: jax.Array
    promotion_counter: jax.Array
    demotion_counter: jax.Array
```

Promotion should be relatively fast:

```python
promote = critical_for_n_steps
```

Demotion should be slower and require recovery:

```python
demote = (
    recovered
    & low_uncertainty
    & stable_for_m_steps
)
```

A typical policy is:

```text
promotion delay << demotion delay
```

A compartment should return to a reduced model only after:

- repolarization,
- refractory recovery,
- low surrogate uncertainty,
- and stable local dynamics.

---

## 11. State Reconstruction During Model Switching

When a surrogate compartment becomes explicit, detailed state variables may be missing.

For example, a biophysical membrane model may require:

$$
m,\;h,\;n,\;\ldots
$$

while the surrogate contains only a latent vector.

Several strategies are possible.

### 11.1 Evolve explicit states in parallel

The explicit state is evolved in the background, at least in a buffer region.

Advantages:

- simple promotion,
- accurate state initialization,
- high robustness.

Disadvantage:

- reduced speedup.

### 11.2 Learn a reconstruction model

A separate model reconstructs explicit states:

```python
gates = reconstruct_gates(
    vm_history,
    latent_state,
    morphology_features,
)
```

Advantages:

- fast promotion.

Disadvantages:

- additional training problem,
- possible inconsistency,
- potential instability.

### 11.3 Temporal overlap

Before promotion, run both models for several time steps:

```text
surrogate only
    ->
surrogate + explicit synchronization
    ->
explicit only
```

This is usually the safest compromise.

The same principle can be used during demotion to synchronize the surrogate latent state with the explicit trajectory.

---

## 12. Dual-Model Overlap Region

The overlap region should not only stabilize interfaces. It can also automatically generate training data.

In the overlap region, evaluate:

- the explicit model,
- the surrogate model,

under identical local conditions.

Example:

```python
teacher_next = double_cable_step(...)
student_next = surrogate_step(...)
```

or, in a lower-fidelity overlap:

```python
teacher_next = single_cable_step(...)
student_next = surrogate_step(...)
```

This creates automatic teacher-student supervision.

A training sample may contain:

$$
\mathcal{D}_t
=
\left(
V_m^t,
\mathbf{z}^t,
I_{\mathrm{axial}}^t,
I_{\mathrm{ext}}^t,
\theta,
I_{\mathrm{mem}}^{\mathrm{teacher},t},
V_m^{\mathrm{teacher},t+\Delta t}
\right)
$$

The explicit solver therefore generates training examples exactly in the regions where the surrogate is most likely to fail.

---

## 13. Training Objectives

A surrogate loss may combine several terms:

$$
\mathcal{L}
=
\lambda_V \mathcal{L}_V
+
\lambda_I \mathcal{L}_I
+
\lambda_{\mathrm{event}} \mathcal{L}_{\mathrm{spike}}
+
\lambda_{\mathrm{rollout}} \mathcal{L}_{\mathrm{rollout}}
$$

### Voltage loss

$$
\mathcal{L}_V
=
\left\|
V_m^{\mathrm{S},t+\Delta t}
-
V_m^{\mathrm{teacher},t+\Delta t}
\right\|^2
$$

### Membrane-current loss

$$
\mathcal{L}_I
=
\left\|
I_{\mathrm{mem}}^{\mathrm{S}}
-
I_{\mathrm{mem}}^{\mathrm{teacher}}
\right\|^2
$$

### Event loss

This may penalize errors in:

- spike initiation,
- spike timing,
- threshold crossing,
- conduction failure,
- propagation direction.

### Rollout loss

One-step accuracy is not sufficient.

The surrogate should also be trained or validated over closed-loop trajectories because small local errors can accumulate and modify:

- activation threshold,
- conduction velocity,
- refractoriness,
- spike shape,
- or numerical stability.

---

## 14. Active Learning

Not all overlap samples are equally useful.

Samples should be retained preferentially when:

- surrogate uncertainty is high,
- surrogate and explicit outputs disagree,
- the state is near threshold,
- a spike is initiated,
- a spike reaches an interface,
- conduction block occurs,
- two spikes collide,
- the waveform differs from the training distribution,
- morphology or temperature is unusual.

A sample priority score may be defined as:

$$
p_t
=
\alpha_u u_t
+
\alpha_d d_t
+
\alpha_e e_t
+
\alpha_o o_t
$$

where:

- $u_t$ is uncertainty,
- $d_t$ is teacher-student disagreement,
- $e_t$ is event importance,
- $o_t$ is an out-of-distribution score.

This turns the explicit solver into an active data-generation system.

---

## 15. Continual Learning Strategy

The surrogate can improve automatically from explicit-solver data, but unrestricted online optimization inside the time-stepping loop is risky.

Potential problems include:

- catastrophic forgetting,
- changing solver behavior during a trajectory,
- loss of reproducibility,
- unstable closed-loop dynamics,
- expensive recompilation,
- difficult debugging.

A safer architecture uses two separate loops.

### 15.1 Simulation loop

```python
trajectory, training_samples = run_hybrid_simulation(
    initial_state,
    stimulation,
    geometry,
    surrogate_params,
)
```

### 15.2 Training loop

```python
new_surrogate_params = train_on_replay_buffer(
    surrogate_params,
    training_samples,
    replay_buffer,
)
```

The updated surrogate should be validated before deployment.

A safe deployment pipeline is:

```text
collect samples
    ->
train candidate surrogate
    ->
run stability and accuracy tests
    ->
compare with current surrogate
    ->
accept or reject candidate
```

Rollback to the previous surrogate should always be possible.

---

## 16. Replay Buffer

A replay buffer is important to preserve previously learned regimes.

The buffer should include examples from:

- resting state,
- subthreshold stimulation,
- threshold stimulation,
- suprathreshold activation,
- refractory states,
- pulse trains,
- conduction block,
- spike collision,
- different diameters,
- different waveforms,
- different temperatures,
- different electrode configurations.

Sampling should combine:

- recent difficult examples,
- representative historical examples,
- rare physiological events.

This reduces catastrophic forgetting and improves global stability.

---

## 17. Uncertainty Estimation

Adaptive promotion requires an uncertainty or reliability estimate.

Possible approaches include:

- ensemble of small surrogates,
- Monte Carlo dropout,
- learned variance output,
- distance in latent or feature space,
- disagreement with a reduced analytical model,
- teacher-student disagreement in overlap zones.

A compartment should fall back to explicit simulation when:

$$
u_i > \tau_u
$$

or when the state lies outside the validated training domain.

The uncertainty threshold should be calibrated against actual rollout errors, not only one-step prediction errors.

---

## 18. Time Integration with `jax.lax.scan`

The temporal solver should ideally use `jax.lax.scan`.

Example:

```python
def simulation_step(carry, inputs):
    state, fidelity_state = carry
    i_ext_t, ve_t = inputs

    next_state, next_fidelity_state, diagnostics = hybrid_step(
        state=state,
        fidelity_state=fidelity_state,
        i_ext_t=i_ext_t,
        ve_t=ve_t,
        surrogate_params=surrogate_params,
        config=config,
    )

    return (
        next_state,
        next_fidelity_state,
    ), diagnostics
```

Then:

```python
(final_state, final_fidelity_state), diagnostics = jax.lax.scan(
    simulation_step,
    (initial_state, initial_fidelity_state),
    time_inputs,
)
```

Benefits include:

- one compiled temporal loop,
- good accelerator utilization,
- controlled memory,
- compatibility with autodiff,
- easier batching across fibers.

For long trajectories, use:

- `jax.checkpoint`,
- rematerialization,
- chunked scans,
- or custom adjoint strategies.

---

## 19. Suggested Hybrid Step

A conceptual hybrid time step is:

```python
def hybrid_step(
    state,
    fidelity_state,
    i_ext_t,
    ve_t,
    surrogate_params,
    single_params,
    double_params,
    geometry,
    config,
):
    # 1. Compute global axial coupling
    i_axial = compute_global_axial_currents(
        state,
        geometry,
    )

    # 2. Evaluate candidate model updates
    double_next = double_cable_step(
        state,
        i_axial,
        i_ext_t,
        double_params,
        config.dt,
    )

    single_next = single_cable_step(
        state,
        i_axial,
        i_ext_t,
        single_params,
        config.dt,
    )

    surrogate_next, uncertainty = surrogate_step(
        state,
        i_axial,
        i_ext_t,
        surrogate_params,
        geometry,
        config.dt,
    )

    # 3. Update fidelity requests
    dynamic_critical = (
        (state.vm > config.voltage_alert_threshold)
        | (uncertainty > config.uncertainty_threshold)
    )

    requested = (
        config.static_af_mask
        | dynamic_critical
    )

    next_fidelity_state = update_fidelity_state(
        fidelity_state,
        requested,
        state,
        uncertainty,
        config,
    )

    # 4. Select model output
    next_state = select_model_state(
        double_next,
        single_next,
        surrogate_next,
        next_fidelity_state.level,
    )

    # 5. Collect overlap supervision
    training_samples = collect_overlap_samples(
        state,
        double_next,
        single_next,
        surrogate_next,
        uncertainty,
        next_fidelity_state,
    )

    diagnostics = {
        "uncertainty": uncertainty,
        "fidelity_level": next_fidelity_state.level,
        "training_samples": training_samples,
    }

    return next_state, next_fidelity_state, diagnostics
```

---

## 20. Recommended API

A top-level API could be:

```python
def hybrid_solve(
    initial_state,
    extracellular_potential,
    stimulation,
    geometry,
    single_params,
    double_params,
    surrogate_params,
    config,
):
    """
    Returns
    -------
    trajectory
        Simulated hybrid state trajectory.

    diagnostics
        Fidelity maps, uncertainty, interface errors,
        spike events, and performance metrics.

    overlap_training_data
        Teacher-student samples collected from
        overlap regions.
    """
```

A configuration object may contain:

```python
class HybridConfig(NamedTuple):
    dt: float

    af_threshold: float
    af_gradient_threshold: float

    double_radius: int
    single_buffer_radius: int
    overlap_width: int

    voltage_alert_threshold: float
    uncertainty_threshold: float
    disagreement_threshold: float

    promotion_delay: int
    demotion_delay: int

    static_af_mask: jax.Array
```

---

## 21. Recommended Development Roadmap

### Phase 1: Fixed partition

Implement:

- activating-function map,
- fixed double-cable core,
- fixed single-cable buffer,
- surrogate exterior,
- shared global axial coupling.

Do not use dynamic switching yet.

Goal:

- validate interface behavior,
- measure speedup,
- identify numerical artifacts.

### Phase 2: Overlap and data collection

Add:

- explicit and surrogate evaluation in an overlap region,
- disagreement diagnostics,
- automatic training-sample collection,
- replay-buffer construction.

Do not update the surrogate during the simulation yet.

Goal:

- build a representative dataset,
- identify where the surrogate fails.

### Phase 3: Offline active learning

Add:

- prioritized sampling,
- candidate surrogate retraining,
- rollout validation,
- deployment gates,
- rollback mechanism.

Goal:

- improve the surrogate iteratively without destabilizing the solver.

### Phase 4: Adaptive fidelity

Add:

- dynamic promotion,
- slow demotion,
- uncertainty thresholds,
- state synchronization,
- moving high-fidelity windows.

Goal:

- concentrate explicit computation around moving spike fronts and difficult events.

### Phase 5: Large-scale vectorization

Add:

- `vmap` across fibers,
- `pmap` or sharding across devices,
- grouped fidelity patterns,
- reduced recompilation,
- batched stimulation conditions.

Goal:

- scale to large nerve populations and optimization loops.

---

## 22. Validation Plan

The hybrid model should be benchmarked against the full double-cable solver.

### Physiological metrics

- activation threshold,
- strength-duration curve,
- spike-initiation site,
- activation latency,
- conduction velocity,
- spike amplitude,
- spike waveform,
- refractory period,
- paired-pulse response,
- pulse-train response,
- conduction block,
- spike collision,
- orthodromic and antidromic propagation.

### Numerical metrics

- interface voltage mismatch,
- interface axial-current mismatch,
- artificial reflection amplitude,
- accumulated rollout error,
- stability at rest,
- spontaneous spike rate,
- sensitivity to overlap width,
- sensitivity to threshold values,
- sensitivity to time step.

### Performance metrics

- JAX compilation time,
- execution time after compilation,
- GPU memory,
- throughput per fiber,
- speedup versus full double-cable,
- speedup versus full single-cable,
- cost of uncertainty estimation,
- cost of overlap evaluation,
- training-data generation rate.

Compilation time and post-compilation execution time should be reported separately.

---

## 23. Main Failure Modes

The main risks are:

- artificial spike reflection at interfaces,
- incorrect conduction velocity,
- attenuation or amplification,
- duplicated or missing spikes,
- spontaneous activation,
- incorrect refractory-state transfer,
- unstable surrogate rollout,
- inaccurate state reconstruction,
- excessive fidelity-mask switching,
- catastrophic forgetting,
- deployment of an unstable surrogate,
- poor extrapolation to unseen waveforms or morphologies.

These failures should be tested explicitly.

---

## 24. Practical Recommendation

Given an existing JAX implementation of both single-cable and double-cable equations, the most practical architecture is:

```text
extracellular field
        |
activating function
        |
initial fidelity map
        |
double-cable critical core
        |
single-cable physical buffer
        |
surrogate outer region
        |
dual-model overlap data
        |
active-learning replay buffer
        |
validated surrogate updates
```

The single-cable solver is especially valuable because it provides:

- a physical transition layer,
- a cheaper explicit fallback,
- an intermediate teacher,
- and a way to reduce the discontinuity between double-cable dynamics and a learned surrogate.

The explicit double-cable solver remains the highest-fidelity reference and should always be available as a fallback when:

- uncertainty is high,
- the input is outside the validated domain,
- a critical physiological event occurs,
- or the surrogate fails a deployment test.

---

## 25. Final Design Principle

The recommended system is not a surrogate replacing the cable solver.

It is a self-improving multilevel solver in which:

- cable physics remains globally coupled,
- model fidelity varies in space and time,
- the activating function initializes compartmentalization,
- explicit overlap regions supervise the surrogate,
- uncertainty controls fallback,
- and JAX provides compilation, vectorization, and differentiation.

The central design objective should be:

> Maximize computational speed while preserving activation thresholds, propagation dynamics, interface stability, and reliable fallback to explicit biophysics.

---

## 26. Positioning Relative to the S-MF Nature Communications Study

A relevant reference is the 2024 Nature Communications study introducing a simplified myelinated-fiber model, referred to here as S-MF.

That work demonstrates that a physically structured, learned single-cable model can reproduce many behaviors of a more detailed MRG double-cable reference while providing very large computational speedups.

The key lesson is that a surrogate does not have to be a purely black-box mapping from stimulation parameters to activation.

The S-MF strategy preserves:

- cable dynamics,
- nodal membrane behavior,
- spatiotemporal voltage propagation,
- nonlinear conductances,
- spike initiation,
- conduction block,
- collision phenomena,
- and compatibility with gradient-based optimization.

This explains why it generalizes better than a simple classifier of the form:

```text
stimulation parameters -> activated / not activated
```

A physically constrained surrogate is less dependent on the exact training set than a global black-box predictor, although it remains dependent on the range of morphologies, diameters, waveforms, and stimulation conditions used during calibration and validation.

### 26.1 Conceptual comparison

The S-MF approach can be summarized as:

```text
double-cable MRG teacher
        |
offline calibration
        |
single-cable surrogate used globally
```

The hybrid framework proposed here can be summarized as:

```text
double-cable in critical regions
        |
single-cable in transition regions
        |
surrogate in low-risk regions
        |
online disagreement and uncertainty monitoring
        |
fallback to explicit simulation
```

The two methods therefore target different objectives.

S-MF primarily targets maximum throughput through global model replacement.

The proposed hybrid method targets reliability-aware acceleration through local and adaptive model selection.

### 26.2 Main distinction

The central distinction is the location of the explicit double-cable model during inference.

In the S-MF approach, the double-cable model is mainly used as a teacher during model construction and as a reference for validation.

In the proposed hybrid approach, the double-cable solver remains available during inference and can be activated locally when:

- the activating function identifies a sensitive region,
- surrogate uncertainty becomes high,
- explicit and surrogate predictions disagree,
- a spike front reaches an interface,
- conduction block is suspected,
- a morphology change occurs,
- or the local state is outside the validated training distribution.

This makes the hybrid framework a multi-fidelity solver rather than a global surrogate replacement.

---

## 27. Scientific Contribution of the Hybrid Approach

The proposed method should not be presented simply as a better surrogate than S-MF.

A stronger and more defensible positioning is:

> An adaptive multi-fidelity extension of physics-informed surrogate fiber modeling.

The main scientific question becomes:

> Can a solver retain most of the computational benefit of a reduced cable model while dynamically restoring double-cable fidelity whenever local error, stimulation sensitivity, or uncertainty indicates that the reduced model may be unreliable?

Potential contributions include the following.

### 27.1 Spatially varying fidelity

The fidelity level changes across the fiber:

```text
high activating-function magnitude
        -> double-cable

intermediate transition region
        -> single-cable

weak and slowly varying region
        -> surrogate
```

This differs from applying one reduced model uniformly over the entire fiber.

### 27.2 Local error control

In overlap regions, the explicit and surrogate models can be evaluated under identical conditions.

A local discrepancy metric may be defined as:

$$
e_i(t)
=
\left|
V_{m,i}^{\mathrm{surrogate}}(t)
-
V_{m,i}^{\mathrm{explicit}}(t)
\right|
$$

Additional discrepancy metrics may compare:

- membrane current,
- next-step voltage,
- spike timing,
- conduction velocity,
- latent state,
- or event classification.

If the discrepancy exceeds a threshold, the region is promoted to a higher-fidelity model.

### 27.3 Active data generation

The explicit model generates new training data exactly where the surrogate is weak.

This is particularly useful near:

- activation thresholds,
- conduction-block transitions,
- spike collisions,
- unusual stimulation waveforms,
- small fiber diameters,
- intraneural field configurations,
- or geometries absent from the original training set.

### 27.4 Out-of-distribution protection

A global learned model may perform well inside its validated domain but degrade outside it.

The hybrid solver can provide explicit fallback when:

$$
u_i > \tau_u
$$

or when feature-space distance, model disagreement, or validation rules indicate an out-of-distribution state.

This does not eliminate extrapolation risk, but it makes the risk observable and actionable.

---

## 28. Important Difference in Computational Baselines

The computational conclusions of the S-MF study should not be transferred directly to an existing JAX implementation.

A major comparison in that study is between:

- a reduced differentiable model executed efficiently on modern accelerators,
- and a detailed MRG reference implemented in NEURON on CPU.

The present context is different because the detailed single-cable and double-cable solvers are already implemented in JAX.

Therefore:

- the explicit reference may already be JIT-compiled;
- the double-cable model may already run on GPU;
- fibers may already be vectorized with `vmap`;
- the number of compartments may be small;
- the cost of Python and NEURON execution may already have been removed.

Consequently, a large speedup relative to NEURON does not imply an equally large speedup relative to a compiled JAX double-cable solver.

This is a central experimental question.

---

## 29. When Full Double-Cable JAX May Be Preferable

A full double-cable JAX solver may remain the best practical solution when:

- the number of compartments is small;
- the number of fibers is limited;
- the simulation duration is short;
- the number of optimization evaluations is moderate;
- the solver is already efficiently vectorized;
- model fidelity is more important than latency;
- or the hybrid bookkeeping cost is comparable to the saved computation.

The relevant total cost scales approximately with:

$$
C_{\mathrm{total}}
\propto
N_{\mathrm{comp}}
N_{\mathrm{time}}
N_{\mathrm{fibers}}
N_{\mathrm{simulations}}
$$

Even if one double-cable simulation is inexpensive, the cumulative cost can become large during:

- waveform optimization,
- population studies,
- uncertainty quantification,
- Monte Carlo simulation,
- inverse problems,
- or real-time control.

The hybrid method becomes more attractive as this cumulative workload increases.

---

## 30. Expected Break-Even Behavior

The hybrid method is unlikely to dominate in every regime.

A useful hypothesis is that there exists a break-even point:

$$
N_{\mathrm{workload}}^\star
$$

below which full double-cable JAX is simpler and sufficiently fast, and above which the hybrid solver provides a meaningful advantage.

The break-even point may depend on:

- number of compartments;
- number of fibers;
- stimulation duration;
- overlap width;
- fraction of the fiber assigned to double-cable;
- surrogate architecture;
- uncertainty-estimation cost;
- mask fragmentation;
- device type;
- and batch size.

The hybrid approach should therefore be evaluated as a Pareto problem rather than as a universally superior replacement.

---

## 31. Required Experimental Baselines

At minimum, the following three systems should be compared.

### Baseline A: Full double-cable JAX

This is the accuracy reference and the most relevant performance baseline.

### Baseline B: Global S-MF-like model

A learned or calibrated single-cable model is applied over the entire fiber.

This is the closest conceptual baseline to the Nature Communications study.

### Baseline C: Adaptive hybrid model

The fiber uses:

- double-cable in critical regions,
- single-cable in transition regions,
- surrogate in low-risk regions,
- overlap-based monitoring,
- and uncertainty-controlled fallback.

The key comparison should be expressed as an error-runtime Pareto front:

$$
\text{physiological error}
\quad \text{versus} \quad
\text{execution time}
$$

---

## 32. Recommended Benchmark Matrix

The three baselines should be evaluated over a matrix of conditions.

### Spatial complexity

- small number of compartments;
- medium number of compartments;
- long fibers;
- fragmented critical regions;
- smooth critical regions.

### Population size

- one fiber;
- tens of fibers;
- thousands of fibers;
- very large batched populations.

### Stimulation regime

- single rectangular pulse;
- biphasic pulse;
- pulse train;
- high-frequency block;
- arbitrary waveform;
- multi-electrode stimulation.

### Generalization regime

- diameters inside the training distribution;
- diameters outside the training distribution;
- unseen temperatures;
- unseen electrode geometries;
- unseen morphologies;
- intraneural versus extraneural fields.

### Accuracy outputs

- activation threshold;
- spike-initiation site;
- latency;
- conduction velocity;
- waveform error;
- refractory response;
- collision behavior;
- conduction block.

### Performance outputs

- compilation time;
- post-compilation runtime;
- memory usage;
- throughput per fiber;
- speedup relative to full double-cable JAX;
- fraction of compartments simulated explicitly;
- overlap overhead;
- uncertainty-estimation overhead.

---

## 33. Risks Specific to the Hybrid Method

The hybrid method may be more reliable than a global surrogate, but it is also more complex.

Potential disadvantages include:

- reduced GPU efficiency due to irregular masks;
- branch divergence;
- evaluation of several models at the same location;
- state-conversion overhead;
- fragmented high-fidelity regions;
- synchronization cost;
- larger JAX compilation graphs;
- and more difficult debugging.

If all candidate models are evaluated everywhere before masking, the expected speedup may disappear.

The implementation should therefore distinguish between:

### Validation implementation

Evaluate all models and select by masks.

This is simple and useful for correctness testing.

### Optimized implementation

Use:

- compact fixed-size windows,
- grouped fibers with similar masks,
- sparse explicit regions,
- static partition templates,
- or specialized kernels.

This is required to realize the full performance benefit.

---

## 34. Proposed Positioning Statement

A concise research positioning is:

> Existing physics-informed surrogate models maximize throughput by globally replacing a detailed double-cable model with a calibrated reduced cable model. The proposed framework instead targets reliability-aware acceleration by retaining explicit double-cable dynamics locally and adaptively whenever stimulation sensitivity, model disagreement, or uncertainty indicates that the reduced model may be unreliable.

A stronger version for a technical manuscript is:

> We introduce an adaptive multi-fidelity nerve-fiber solver that combines a JAX double-cable model, a single-cable transition model, and a learned local surrogate. Fidelity is initialized using the activating function and updated using uncertainty and teacher-student disagreement. Unlike global surrogate replacement, the method preserves explicit biophysics in locally critical or out-of-distribution regions and uses overlap zones for online error estimation and active data generation.

---

## 35. Updated Practical Recommendation

The recommended development order is:

```text
full double-cable JAX baseline
        |
profiling and scaling analysis
        |
global S-MF-like single-cable baseline
        |
fixed hybrid partition
        |
overlap-based error monitoring
        |
active data collection
        |
adaptive fidelity and fallback
```

The full double-cable solver should remain the primary baseline.

The hybrid approach should only be considered successful if it provides one or more of the following:

- lower runtime at matched physiological accuracy;
- better out-of-distribution reliability than a global surrogate;
- controlled fallback to explicit biophysics;
- lower cumulative cost for large optimization workloads;
- or a better error-speed Pareto front.

The expected outcome is not necessarily that the hybrid method is always faster.

A scientifically useful result may instead be the identification of the regimes in which:

```text
full double-cable JAX is optimal
```

versus the regimes in which:

```text
global reduced model is optimal
```

or:

```text
adaptive multi-fidelity modeling is optimal
```

That break-even map is itself a meaningful contribution.

