# Hybrid and Reduced Single-Cable Hodgkin–Huxley Modeling in JAX

## 1. Purpose

This document proposes a practical modeling strategy for an axon represented by a **single cable** with Hodgkin–Huxley-type membrane dynamics.

The objective is to balance:

- biophysical fidelity,
- GPU throughput,
- robustness outside the training distribution,
- differentiability,
- and implementation complexity.

The central recommendation is:

```text
Start with a fully explicit Hodgkin–Huxley cable in JAX.

Only introduce model reduction if profiling shows a real bottleneck.

If reduction is required, preserve the cable and ion-channel structure,
learn only a small residual correction, and retain full Hodgkin–Huxley
as a local or global fallback.
```

The preferred hierarchy is:

```text
Full HH cable
    ↓
Reduced HH
    ↓
Reduced HH + learned residual correction
    ↓
Spatially adaptive reduced / corrected / full HH
```

This approach is intended to be safer and more interpretable than a pure black-box surrogate.

---

## 2. Full Single-Cable Hodgkin–Huxley Model

The classical Hodgkin–Huxley model describes membrane voltage using sodium, potassium, and leakage currents [1].

For compartment $i$:

$$
C_m \frac{dV_{m,i}}{dt}
=
I_{\mathrm{axial},i}
-
I_{\mathrm{Na},i}
-
I_{\mathrm{K},i}
-
I_{\mathrm{L},i}
+
I_{\mathrm{ext},i}
$$

with:

$$
I_{\mathrm{Na},i}
=
\bar{g}_{\mathrm{Na}}
m_i^3 h_i
\left(
V_{m,i}-E_{\mathrm{Na}}
\right)
$$

$$
I_{\mathrm{K},i}
=
\bar{g}_{\mathrm{K}}
n_i^4
\left(
V_{m,i}-E_{\mathrm{K}}
\right)
$$

$$
I_{\mathrm{L},i}
=
g_{\mathrm{L}}
\left(
V_{m,i}-E_{\mathrm{L}}
\right)
$$

The gating variables obey:

$$
\frac{dx_i}{dt}
=
\alpha_x(V_{m,i})(1-x_i)
-
\beta_x(V_{m,i})x_i
$$

or equivalently:

$$
\frac{dx_i}{dt}
=
\frac{x_\infty(V_{m,i})-x_i}
{\tau_x(V_{m,i})}
$$

for:

$$
x\in\{m,h,n\}
$$

The axial term for a uniform cable may be discretized as:

$$
I_{\mathrm{axial},i}
=
G_{\mathrm{ax}}
\left(
V_{m,i-1}
-
2V_{m,i}
+
V_{m,i+1}
\right)
$$

with appropriate modifications for nonuniform geometry.

---

## 3. Why Full HH Should Be the First Baseline

Classical HH has only four dynamic states per compartment:

```text
Vm
m
h
n
```

This is already a relatively compact model.

If the number of compartments is small, a full HH implementation may be sufficiently fast, especially when:

- compiled with `jax.jit`,
- integrated with `jax.lax.scan`,
- vectorized across fibers with `jax.vmap`,
- and executed on a GPU or TPU [8–10].

Therefore, a surrogate should not be introduced before measuring:

```text
compilation time
post-compilation runtime
memory usage
throughput per fiber
cost per optimization iteration
```

A reduced or learned model is justified primarily when the cumulative workload is large:

$$
C_{\mathrm{total}}
\propto
N_{\mathrm{comp}}
N_{\mathrm{time}}
N_{\mathrm{fibers}}
N_{\mathrm{simulations}}
$$

Typical high-cost use cases include:

- large fiber populations,
- stimulation waveform optimization,
- uncertainty quantification,
- parameter sweeps,
- inverse problems,
- and near-real-time control.

---

## 4. Optimizing the Full HH Solver Before Reducing It

## 4.1 Exponential Gate Updates

When voltage is treated as constant over one time step, each gate equation has an analytical update:

$$
x^{t+\Delta t}
=
x_\infty(V^t)
+
\left(
x^t-x_\infty(V^t)
\right)
\exp
\left(
-\frac{\Delta t}{\tau_x(V^t)}
\right)
$$

This is closely related to the Rush–Larsen integration strategy for dynamic membrane equations [2].

Advantages:

```text
Stable gate updates
No explicit Euler drift for fast gates
Simple vectorized implementation
Good compatibility with JAX
```

Example:

```python
def update_gate(x, x_inf, tau_x, dt):
    return x_inf + (x - x_inf) * jnp.exp(-dt / tau_x)
```

---

## 4.2 Split or Semi-Implicit Cable Integration

The ionic dynamics and axial cable coupling may be separated.

A practical time step can use:

```text
1. explicit or exponential update of gating variables
2. implicit or semi-implicit solve for axial voltage coupling
3. explicit evaluation of nonlinear ionic currents
```

If the axial operator is tridiagonal, the voltage solve can remain inexpensive.

This is often preferable to replacing the entire HH model with a neural network.

---

## 4.3 JAX Execution Pattern

Use:

- `jax.jit` for compilation,
- `jax.lax.scan` for the time loop,
- `jax.vmap` for batching fibers [8–10].

Example:

```python
def hh_time_step(state, inputs):
    vm, m, h, n = state
    i_ext_t, geometry = inputs

    m_next, h_next, n_next = update_gates(
        vm, m, h, n, dt
    )

    vm_next = cable_voltage_step(
        vm=vm,
        m=m_next,
        h=h_next,
        n=n_next,
        i_ext=i_ext_t,
        geometry=geometry,
        dt=dt,
    )

    next_state = (
        vm_next,
        m_next,
        h_next,
        n_next,
    )

    return next_state, vm_next
```

```python
final_state, vm_history = jax.lax.scan(
    hh_time_step,
    initial_state,
    time_inputs,
)
```

Across fibers:

```python
batched_solver = jax.vmap(
    solve_one_fiber,
    in_axes=(0, 0, None),
)
```

---

## 5. Model Hierarchy

A useful fidelity hierarchy is:

```text
Level 0: Full HH
Level 1: Reduced HH
Level 2: Reduced HH + learned correction
Level 3: Spatially adaptive combination
```

This ordering reflects implementation priority, not computational cost.

The full model should be developed and validated first because it serves as:

- the reference solution,
- the teacher model,
- the fallback model,
- and the source of training data.

---

## 6. Reduced HH Variant A: Instantaneous Sodium Activation

The sodium activation gate $m$ is often faster than $h$ and $n$.

A common reduction is:

$$
m(t)
\approx
m_\infty(V_m(t))
$$

The dynamic state becomes:

```text
Vm
h
n
```

instead of:

```text
Vm
m
h
n
```

The sodium current becomes:

$$
I_{\mathrm{Na}}
=
\bar{g}_{\mathrm{Na}}
m_\infty(V_m)^3
h
\left(
V_m-E_{\mathrm{Na}}
\right)
$$

Advantages:

```text
Simple
Interpretable
No state reconstruction for h and n
Preserves explicit Na, K, and leak currents
```

Limitations:

```text
The m gate is not always infinitely fast
Errors may increase at high injected currents
Spike shape and firing-range predictions may shift
```

Meunier showed that assuming instantaneous sodium activation can preserve the qualitative bifurcation structure while introducing quantitative errors in the periodic-firing range [4].

Therefore, this reduction must be validated for the intended stimulation regime.

---

## 7. Reduced HH Variant B: Two-State Approximation

A stronger reduction combines slow variables.

For example:

$$
h
\approx
H(n)
$$

or:

$$
(h,n)
\rightarrow
w
$$

The model becomes:

```text
Vm
w
```

with:

$$
C_m\frac{dV_m}{dt}
=
I_{\mathrm{axial}}
-
I_{\mathrm{ion}}(V_m,w)
+
I_{\mathrm{ext}}
$$

$$
\frac{dw}{dt}
=
F_w(V_m,w)
$$

Systematic reductions of conductance-based models have been developed using time-scale separation and variable transformations [3,4].

Advantages:

```text
Very small state
High throughput
Useful for qualitative excitability
```

Limitations:

```text
Potential loss of refractory accuracy
Potential loss of spike-shape accuracy
Potential errors in pulse trains
Potential errors near conduction block
Harder reconstruction of m, h, and n
```

This model is appropriate only after demonstrating that the reduced state preserves the required observables.

---

## 8. Reduced HH Variant C: Effective Kinetics

Instead of eliminating gates, retain the HH structure but replace the original kinetics by calibrated effective functions:

$$
\tilde{x}_\infty(V)
$$

and:

$$
\tilde{\tau}_x(V)
$$

for:

$$
x\in\{m,h,n\}
$$

The gate equations remain:

$$
\frac{dx}{dt}
=
\frac{
\tilde{x}_\infty(V)-x
}{
\tilde{\tau}_x(V)
}
$$

This creates a physically structured pseudo-HH model.

Advantages:

```text
Preserves gate interpretation
Preserves state dimension
Easy promotion to full HH
No latent-state reconstruction
Can approximate a more complex membrane model
```

Limitations:

```text
Requires calibration
Effective kinetics may depend on waveform and temperature
Can still fail outside the calibration domain
```

This is a strong option when HH is used as a reduced approximation of a more complex conductance model.

---

## 9. Reduced HH with Learned Residual Correction

The preferred learned approach is to retain the explicit reduced HH model and learn only its discrepancy.

The model becomes:

$$
\text{corrected model}
=
\text{reduced HH}
+
\text{small learned residual}
$$

Hybrid neural differential equations have been used to model missing ion-channel dynamics while preserving mechanistic ODE structure [5].

---

## 9.1 Current Residual

The ionic current is written as:

$$
I_{\mathrm{ion}}
=
I_{\mathrm{ion}}^{\mathrm{reduced}}
+
\mathcal{R}_\theta
$$

where:

$$
\mathcal{R}_\theta
=
\mathcal{R}_\theta
\left(
V_m,
m,
h,
n,
I_{\mathrm{axial}},
I_{\mathrm{ext}},
\theta_{\mathrm{morph}},
\text{history}
\right)
$$

The voltage equation becomes:

$$
C_m\frac{dV_m}{dt}
=
I_{\mathrm{axial}}
-
I_{\mathrm{ion}}^{\mathrm{reduced}}
-
\mathcal{R}_\theta
+
I_{\mathrm{ext}}
$$

Advantages:

```text
Simple correction target
Direct physical unit: current density
Residual magnitude is interpretable
Easy to disable the correction
```

---

## 9.2 Kinetic Residual

A more structured approach corrects gate steady states and time constants.

For the steady-state gate value:

$$
x_{\infty,\theta}
=
\sigma
\left(
\operatorname{logit}
\left(
x_\infty^{\mathrm{base}}
\right)
+
r_{\infty,\theta}
\right)
$$

This guarantees:

$$
0<x_{\infty,\theta}<1
$$

For the time constant:

$$
\tau_{x,\theta}
=
\tau_x^{\mathrm{base}}
\exp
\left(
r_{\tau,\theta}
\right)
$$

This guarantees:

$$
\tau_{x,\theta}>0
$$

Advantages:

```text
Preserves channel-state bounds
Preserves positive time constants
More interpretable than direct voltage prediction
Suitable for differentiable calibration
```

This is the preferred correction strategy when interpretability matters.

---

## 9.3 Direct State Residual

A less structured alternative is:

$$
V_m^{t+\Delta t}
=
V_{m,\mathrm{reduced}}^{t+\Delta t}
+
\Delta V_{\theta}^{t+\Delta t}
$$

This may be fast, but it is more dangerous because the correction acts directly on the integrated state.

Risks:

```text
Closed-loop drift
Spontaneous spikes
Voltage instability
Dependence on the training time step
Poor long-rollout behavior
```

Direct voltage correction should be used only with strong rollout validation.

---

## 10. Residual Regularization

The residual should remain small relative to the explicit HH contribution.

A possible loss is:

$$
\mathcal{L}
=
\lambda_V\mathcal{L}_V
+
\lambda_I\mathcal{L}_I
+
\lambda_g\mathcal{L}_{\mathrm{gates}}
+
\lambda_s\mathcal{L}_{\mathrm{spike}}
+
\lambda_r\mathcal{L}_{\mathrm{rollout}}
+
\lambda_R
\left\|
\mathcal{R}_\theta
\right\|^2
$$

Useful constraints include:

```text
bounded residual output
near-zero residual at rest
temporal smoothness
spatial smoothness
positive time constants
gate values constrained to [0, 1]
```

The analytical HH component should explain most of the dynamics.

---

## 11. Proposed Spatial Multi-Fidelity Architecture

If profiling shows that a global reduced model is insufficient, use spatially varying fidelity.

A practical hierarchy is:

```text
Reduced HH
        |
Reduced HH + learned correction
        |
Full HH
```

The spatial arrangement may be:

```text
reduced HH
    |
corrected reduced HH
    |
full HH critical core
    |
corrected reduced HH
    |
reduced HH
```

---

## 11.1 Reduced HH Region

Use reduced HH where:

- extracellular forcing is weak,
- the field is spatially smooth,
- the membrane is far from threshold,
- surrogate uncertainty is low,
- and no critical event is occurring.

---

## 11.2 Corrected Reduced HH Region

Use corrected reduced HH where:

- the analytical reduction has a known bias,
- the state remains within the learned domain,
- the residual is moderate,
- and higher accuracy is required without full HH cost.

---

## 11.3 Full HH Region

Use full HH for:

- spike-initiation sites,
- strong extracellular polarization,
- approaching spike fronts,
- collision regions,
- possible conduction block,
- abrupt geometry changes,
- high residual magnitude,
- high uncertainty,
- or out-of-distribution states.

---

## 12. Activating-Function-Based Initialization

The initial fidelity map can be informed by the activating function.

For extracellular potential $V_e(x)$:

$$
f_{\mathrm{act}}(x)
\propto
-
\frac{\partial^2 V_e}{\partial x^2}
$$

This follows the classical analysis of extracellular axon stimulation [6].

For a uniform grid:

```python
dx = x[1] - x[0]

af_inner = -(
    ve[2:]
    - 2.0 * ve[1:-1]
    + ve[:-2]
) / dx**2

af = jnp.pad(
    af_inner,
    (1, 1),
)
```

Define two thresholds:

```python
full_hh_core = jnp.abs(af) > tau_full
corrected_region = jnp.abs(af) > tau_corrected
```

Then dilate the masks:

```python
full_hh_mask = dilate_mask(
    full_hh_core,
    full_radius,
)

corrected_hh_mask = (
    dilate_mask(
        corrected_region,
        corrected_radius,
    )
    & ~full_hh_mask
)

reduced_hh_mask = ~(
    full_hh_mask
    | corrected_hh_mask
)
```

The activating function should be used only as an initial indicator.

It does not fully capture:

- membrane nonlinearities,
- refractory state,
- waveform history,
- conduction block,
- or spike collisions.

---

## 13. Dynamic Promotion and Demotion

The fidelity map can evolve during simulation.

Promote to full HH when:

```python
promote_to_full = (
    (vm > voltage_alert)
    | (uncertainty > uncertainty_threshold)
    | (residual_magnitude > residual_threshold)
    | spike_front_detected
    | possible_block
    | collision_detected
)
```

Promote from reduced to corrected HH when:

```python
promote_to_corrected = (
    (jnp.abs(af) > af_corrected_threshold)
    | (reduced_model_error > error_threshold)
    | moderate_uncertainty
)
```

Demotion should be slower than promotion.

A compartment should return to a cheaper model only after:

- repolarization,
- refractory recovery,
- low residual magnitude,
- low uncertainty,
- and stable behavior for several time steps.

---

## 14. State Reconstruction During Switching

State switching is easy if all model levels retain $m$, $h$, and $n$.

This is a major advantage of effective-kinetics HH.

If a stronger reduction removes gates, promotion to full HH requires state reconstruction.

Possible strategies are:

### Strategy A: Background Full Gates

Continue evolving $m$, $h$, and $n$ in a small buffer region even when reduced dynamics are used for voltage.

```text
Most robust
Higher cost
Easy promotion
```

### Strategy B: Learned Reconstruction

Estimate:

$$
(m,h,n)
=
\mathcal{G}_\phi
\left(
V_m,
w,
V_m^{\mathrm{history}}
\right)
$$

```text
Fast
Requires additional training
Can create inconsistent states
```

### Strategy C: Temporal Synchronization

Run both models for several steps:

```text
reduced HH
    ->
reduced HH + full HH
    ->
full HH
```

This is generally the safest compromise.

---

## 15. Uncertainty and Out-of-Distribution Detection

The corrected model should produce or support a reliability estimate.

Possible methods include:

- an ensemble of small residual networks,
- learned predictive variance,
- latent-space distance,
- residual magnitude,
- disagreement between reduced and full HH in overlap zones,
- or feature-range checks.

Fallback to full HH when:

$$
u_i>\tau_u
$$

A useful combined indicator is:

$$
q_i
=
\alpha_u u_i
+
\alpha_r
\left|
\mathcal{R}_{\theta,i}
\right|
+
\alpha_d d_i
$$

where:

- $u_i$ is uncertainty,
- $\mathcal{R}_{\theta,i}$ is residual magnitude,
- $d_i$ is explicit-versus-reduced disagreement.

---

## 16. Teacher–Student Training

The full HH model is the teacher.

The reduced or corrected model is the student.

Training samples may contain:

$$
\mathcal{D}_t
=
\left(
V_m^t,
m^t,
h^t,
n^t,
I_{\mathrm{axial}}^t,
I_{\mathrm{ext}}^t,
\theta,
V_m^{t+\Delta t},
m^{t+\Delta t},
h^{t+\Delta t},
n^{t+\Delta t}
\right)
$$

The overlap region can generate these samples automatically.

Priority should be given to:

- near-threshold states,
- spike initiation,
- refractory transitions,
- pulse trains,
- high-frequency stimulation,
- conduction failure,
- and unfamiliar waveforms.

---

## 17. Training Should Be Separated from Simulation

Do not update neural-network weights at every solver time step in the first implementation.

Use two loops.

### Simulation loop

```python
trajectory, training_samples = run_hybrid_hh(
    initial_state,
    stimulation,
    geometry,
    correction_params,
)
```

### Training loop

```python
candidate_params = train_residual(
    correction_params,
    training_samples,
    replay_buffer,
)
```

Deploy the candidate model only after validation.

Recommended deployment process:

```text
collect data
    ->
train candidate
    ->
test resting stability
    ->
test threshold accuracy
    ->
test long rollouts
    ->
test unseen waveforms
    ->
accept or reject
```

---

## 18. Suggested JAX State

Use a fixed PyTree structure:

```python
from typing import NamedTuple
import jax

class HybridHHState(NamedTuple):
    vm: jax.Array
    m: jax.Array
    h: jax.Array
    n: jax.Array
    reduced_state: jax.Array
    uncertainty_state: jax.Array
    fidelity_level: jax.Array
```

Even if some fields are unused in a region, fixed shapes simplify:

- `jax.jit`,
- `jax.lax.scan`,
- batching,
- and state switching.

JAX requires the `scan` carry to keep a fixed structure, shape, and dtype across iterations [8].

---

## 19. Suggested Hybrid Step

```python
def hybrid_hh_step(
    state,
    stimulation_t,
    geometry,
    full_params,
    reduced_params,
    correction_params,
    config,
):
    full_next = full_hh_step(
        state=state,
        stimulation_t=stimulation_t,
        geometry=geometry,
        params=full_params,
        dt=config.dt,
    )

    reduced_next = reduced_hh_step(
        state=state,
        stimulation_t=stimulation_t,
        geometry=geometry,
        params=reduced_params,
        dt=config.dt,
    )

    corrected_next, uncertainty = corrected_hh_step(
        state=state,
        stimulation_t=stimulation_t,
        geometry=geometry,
        reduced_params=reduced_params,
        correction_params=correction_params,
        dt=config.dt,
    )

    residual_magnitude = compute_residual_magnitude(
        corrected_next,
        reduced_next,
    )

    fidelity_level = update_fidelity_map(
        state=state,
        uncertainty=uncertainty,
        residual_magnitude=residual_magnitude,
        config=config,
    )

    next_state = select_hh_state(
        reduced_next=reduced_next,
        corrected_next=corrected_next,
        full_next=full_next,
        fidelity_level=fidelity_level,
    )

    diagnostics = {
        "uncertainty": uncertainty,
        "residual_magnitude": residual_magnitude,
        "fidelity_level": fidelity_level,
    }

    return next_state, diagnostics
```

For validation, all candidate models can be evaluated.

For production, the implementation should avoid computing every model everywhere.

---

## 20. Pure Surrogate Versus Structured HH Reduction

A pure surrogate may directly learn:

```text
stimulus + geometry -> voltage trajectory
```

or:

```text
stimulus + geometry -> activation
```

This can be extremely fast but is strongly dependent on training coverage.

A structured HH reduction retains:

- cable propagation,
- ionic currents,
- state memory,
- refractoriness,
- and physical parameter interpretation.

A residual correction learns only the missing dynamics.

The recommended preference is:

```text
Full HH if affordable

Otherwise:
Reduced HH

If reduced HH is biased:
Reduced HH + learned residual

If uncertainty remains high:
Full HH fallback
```

The S-MF work by Hussain, Grill, and Pelot demonstrates that a physically structured cable surrogate can reproduce rich spatiotemporal responses and achieve major throughput gains [7]. The lesson is not that all black-box surrogates are safe, but that **retaining cable structure and dynamic state greatly improves usefulness**.

---

## 21. Required Baselines

Compare at least:

### Baseline A

```text
Full HH JAX
```

### Baseline B

```text
HH with instantaneous m
```

### Baseline C

```text
Two-state reduced HH
```

### Baseline D

```text
Effective-kinetics HH
```

### Baseline E

```text
Reduced HH + learned residual
```

### Baseline F

```text
Spatially adaptive reduced / corrected / full HH
```

---

## 22. Validation Metrics

### Electrophysiological Accuracy

- resting potential,
- spike threshold,
- spike amplitude,
- spike width,
- after-hyperpolarization,
- conduction velocity,
- initiation location,
- activation latency,
- refractory period,
- paired-pulse response,
- pulse-train response,
- conduction block,
- collision behavior.

### Numerical Stability

- spontaneous spike rate,
- long-rest drift,
- gate-bound violations,
- sensitivity to time step,
- accumulated rollout error,
- interface reflection,
- promotion/demotion oscillation.

### Generalization

- unseen pulse widths,
- unseen amplitudes,
- arbitrary waveforms,
- unseen temperatures,
- unseen fiber diameters,
- unseen extracellular field profiles,
- unseen compartment counts.

### Performance

- compilation time,
- post-compilation runtime,
- GPU memory,
- throughput per fiber,
- speedup at matched accuracy,
- correction-network overhead,
- uncertainty-estimation overhead,
- fraction of compartments using full HH.

---

## 23. Recommended Development Roadmap

### Phase 1: Full HH JAX Baseline

Implement:

```text
full Vm, m, h, n
Rush–Larsen-style gate updates
lax.scan
vmap
jit
```

Establish accuracy and speed.

### Phase 2: Three-State Reduction

Use:

$$
m=m_\infty(V)
$$

Retain:

```text
Vm
h
n
```

Measure accuracy loss.

### Phase 3: Effective-Kinetics HH

Retain all gates but calibrate:

```text
x_inf(V)
tau_x(V)
conductance parameters
```

This avoids state reconstruction.

### Phase 4: Learned Residual

Add a small correction to:

```text
ionic current
or
gate kinetics
```

Train against full HH.

### Phase 5: Fixed Spatial Partition

Use:

```text
reduced HH exterior
corrected HH buffer
full HH core
```

Initialize with the activating function.

### Phase 6: Dynamic Fidelity

Add:

- uncertainty,
- residual thresholds,
- promotion,
- slow demotion,
- moving full-HH regions.

### Phase 7: Active Learning

Use overlap data to expand the training set.

---

## 24. Practical Recommendation

For a classical HH cable with a small number of compartments, the most likely best solution is:

```text
Full HH everywhere
+ optimized JAX integration
```

If this becomes too expensive, the next best option is:

```text
Effective or reduced HH
+ learned residual correction
+ full HH fallback
```

Only introduce spatially adaptive fidelity if benchmarks show that:

- full HH is the dominant cost,
- critical regions occupy a small fraction of the cable,
- and the cost of switching and uncertainty estimation is lower than the saved computation.

The preferred long-term architecture is:

```text
Activating function
        |
Reduced HH in low-risk regions
        |
Corrected reduced HH in intermediate regions
        |
Full HH in critical regions
        |
Uncertainty-driven fallback
        |
Teacher-student active learning
```

---

## 25. Final Design Principle

The recommended approach is not to replace Hodgkin–Huxley with a neural network.

It is to use Hodgkin–Huxley as the physical backbone and allow learning to correct only what the reduced model fails to represent.

The central design objective is:

> Preserve explicit cable and ion-channel physics, use reduction only where it is validated, learn small residual discrepancies, and retain full Hodgkin–Huxley as the reference and fallback model.

---

# References

[1] A. L. Hodgkin and A. F. Huxley, “A quantitative description of membrane current and its application to conduction and excitation in nerve,” *The Journal of Physiology*, vol. 117, no. 4, pp. 500–544, 1952.  
DOI: https://doi.org/10.1113/jphysiol.1952.sp004764  
Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/

[2] S. Rush and H. Larsen, “A practical algorithm for solving dynamic membrane equations,” *IEEE Transactions on Biomedical Engineering*, vol. 25, no. 4, pp. 389–392, 1978.  
DOI: https://doi.org/10.1109/TBME.1978.326270

[3] T. B. Kepler, L. F. Abbott, and E. Marder, “Reduction of conductance-based neuron models,” *Biological Cybernetics*, vol. 66, no. 5, pp. 381–387, 1992.  
DOI: https://doi.org/10.1007/BF00197717

[4] C. Meunier, “Two and three dimensional reductions of the Hodgkin–Huxley system: separation of time scales and bifurcation schemes,” *Biological Cybernetics*, vol. 67, no. 5, pp. 461–468, 1992.  
DOI: https://doi.org/10.1007/BF00200990

[5] C. L. Lei et al., “Neural Network Differential Equations for Ion Channel Modelling,” *Frontiers in Physiology*, vol. 12, 2021.  
DOI: https://doi.org/10.3389/fphys.2021.708944  
Full text: https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.708944/full

[6] F. Rattay, “Analysis of models for external stimulation of axons,” *IEEE Transactions on Biomedical Engineering*, vol. 33, no. 10, pp. 974–977, 1986.  
DOI: https://doi.org/10.1109/TBME.1986.325670

[7] M. A. Hussain, W. M. Grill, and N. A. Pelot, “Highly efficient modeling and optimization of neural fiber responses to electrical stimulation,” *Nature Communications*, vol. 15, article 7597, 2024.  
Article: https://www.nature.com/articles/s41467-024-51709-8

[8] JAX documentation, `jax.lax.scan`.  
https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

[9] JAX documentation, `jax.vmap`.  
https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html

[10] JAX documentation, just-in-time compilation with `jax.jit`.  
https://docs.jax.dev/en/latest/jit-compilation.html
