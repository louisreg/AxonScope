# AxonScope Solver Review — GPU Batching, Extracellular Modeling, MRG, and Pseudo-MRG

## 10. Pseudo-MRG: Reducing Double Cable to Single Cable

A “pseudo-MRG” model is a strong candidate for large GPU batches.

The objective is to approximate the main effects of the MRG double-cable formulation without solving the periaxonal potential, `Vperi`, as a full dynamic cable over the entire fiber.

The reduced model would retain:

```text
Vm
gating variables
MRG-like compartment structure
effective nodal and internodal parameters
precomputed extracellular stimulation
```

while removing, or only locally retaining:

```text
full dynamic Vperi cable solve
global 2x2 block-tridiagonal solve
```

This produces a single-cable model with effective parameters such as:

```text
Cm_eff(x)
Gleak_eff(x)
D_eff(x)
alpha_ext(x)
tau_ext(x)        optional
beta_feedback(x)  optional
```

The main design objective is not to replace the full MRG formulation unconditionally. It is to create a hierarchy of models that can be selected spatially and, potentially, dynamically.

---

## 11. Possible Pseudo-MRG Variants

### 11.1 Simple Field-Filtered Pseudo-MRG

Approximate the periaxonal potential as:

$$
V_{\mathrm{peri}}(x,t)
\approx
\alpha(x)V_{\mathrm{stim}}(x,t)
$$

or with a local first-order filter:

$$
\tau(x)\frac{dZ}{dt}
=
\alpha(x)V_{\mathrm{stim}}(x,t)-Z
$$

with:

$$
V_{\mathrm{peri}}\approx Z
$$

This variant is highly batchable.

Advantages:

```text
Very fast
No spatial double-cable solve
Easy to implement in JAX
Compatible with vmap and lax.scan
```

Limitations:

```text
No feedback from membrane current to periaxonal voltage
May miss relevant MRG dynamics during spikes
May be inaccurate for conduction block or strong nonlinear regimes
```

---

### 11.2 Effective-Admittance Pseudo-MRG

Absorb the double-cable effect into effective single-cable parameters:

```text
Cm_eff
Gm_eff
D_eff
Istim_eff
```

The reduced equation becomes:

$$
C_{m,\mathrm{eff}}
\frac{dV_m}{dt}
=
I_{\mathrm{axial,eff}}(V_m)
-
I_{\mathrm{ion}}(V_m,\mathbf{g})
+
I_{\mathrm{stim,eff}}(V_{\mathrm{stim}})
$$

This is likely the most practical production model.

Advantages:

```text
Fast
Batchable
Biophysically interpretable
Can preserve MRG-like compartment types
Can be fitted against the full double-cable solver
```

Limitations:

```text
Requires calibration
Effective parameters may depend on stimulus regime
May not reproduce transient periaxonal feedback exactly
```

---

### 11.3 Quasi-Static Elimination of `Vperi`

Assume the periaxonal space equilibrates quickly:

$$
\frac{dV_{\mathrm{peri}}}{dt}
\approx 0
$$

Then approximate:

$$
V_{\mathrm{peri},i}
\approx
\beta_i V_{m,i}
+
\alpha_i V_{\mathrm{stim},i}
$$

or include local spatial information:

$$
V_{\mathrm{peri},i}
\approx
\beta_i V_{m,i}
+
\alpha_i V_{\mathrm{stim},i}
+
\gamma_i
\operatorname{smooth}_{\mathrm{local}}
(V_{\mathrm{stim}})_i
$$

Advantages:

```text
Closer to the double-cable equations
Can preserve membrane-to-periaxonal feedback
Still cheaper than a full dynamic double cable
```

Limitations:

```text
Exact elimination is nonlocal if periaxonal axial conductance is retained
A local approximation requires careful validation
The approximation may degrade near sharp spatial field variations
```

---



---

## 11.4 Pseudo-MRG with Learned Residual Correction

A strong compromise between a purely analytical pseudo-MRG and a global surrogate is to keep the reduced physics explicit and learn only the model discrepancy.

The general form is:

$$
\text{pseudo-MRG prediction}
+
\text{learned residual correction}
$$

For example, the periaxonal potential may be modeled as:

$$
V_{\mathrm{peri},i}
=
\alpha_i V_{\mathrm{stim},i}
+
\beta_i V_{m,i}
+
\gamma_i \Delta_{\mathrm{local}}V_{\mathrm{stim},i}
+
\mathcal{R}_{\theta,i}
$$

where $\mathcal{R}_{\theta,i}$ is a small learned correction model.

The residual model may depend on:

$$
\mathcal{R}_{\theta,i}
=
\mathcal{R}_{\theta}
\left(
V_{m,i},
\mathbf{g}_i,
V_{\mathrm{stim},i},
I_{\mathrm{axial},i},
\text{local field features},
\text{morphology},
\text{recent history}
\right)
$$

An equivalent formulation can correct the membrane update directly:

$$
\frac{dV_m}{dt}
=
F_{\mathrm{pseudo-MRG}}
\left(
V_m,\mathbf{g},V_{\mathrm{stim}}
\right)
+
\mathcal{R}_{\theta}
\left(
V_m,\mathbf{g},V_{\mathrm{stim}},I_{\mathrm{axial}}
\right)
$$

The correction should remain small relative to the analytical pseudo-MRG contribution.

This preserves the main cable structure while allowing the model to capture systematic errors such as:

- transient periaxonal feedback;
- stimulus-dependent effective admittance;
- local nonlinearities near spike initiation;
- small errors in conduction velocity;
- waveform-dependent deviations;
- compartment-specific behavior.

Advantages:

```text
More robust than a pure black-box surrogate
Fewer training data required
Better interpretability
Explicit physics remains active
Easy fallback to the uncorrected pseudo-MRG
Residual magnitude can be monitored
Compatible with local uncertainty estimation
```

Limitations:

```text
Still dependent on the calibration domain
Residual rollout may become unstable
Requires regularization
Needs validation on unseen waveforms and morphologies
```

### Residual regularization

The learned correction should be constrained.

A useful loss is:

$$
\mathcal{L}
=
\lambda_V\mathcal{L}_{V_m}
+
\lambda_P\mathcal{L}_{V_{\mathrm{peri}}}
+
\lambda_R\|\mathcal{R}_{\theta}\|^2
+
\lambda_T\mathcal{L}_{\mathrm{rollout}}
+
\lambda_E\mathcal{L}_{\mathrm{event}}
$$

The residual penalty:

$$
\lambda_R\|\mathcal{R}_{\theta}\|^2
$$

encourages the analytical pseudo-MRG to explain most of the dynamics.

Additional constraints may include:

```text
bounded residual output
smoothness in time
smoothness across neighboring compartments
zero or near-zero correction at rest
monotonicity constraints where physically justified
```

### Recommended use

The learned correction should first be applied to the pseudo-MRG single-cable model.

A practical hierarchy is:

```text
Pseudo-MRG single cable
        |
Pseudo-MRG single cable + learned correction
        |
Pseudo-MRG double cable
        |
Full MRG double cable
```

This gives two independent escalation mechanisms:

```text
Correction is confident and small
    -> corrected pseudo-single model

Correction is large or uncertain
    -> pseudo-double model

Pseudo-double disagreement remains high
    -> full MRG double cable
```

The residual magnitude itself can be used as a fidelity indicator:

$$
r_i(t)
=
\left|
\mathcal{R}_{\theta,i}(t)
\right|
$$

A large residual may indicate that the analytical reduction is outside its reliable regime.


## 12. Proposed Multi-Fidelity Pseudo-MRG Architecture

A useful extension is to combine four model levels along the same fiber:

```text
Low-risk region
    -> pseudo-MRG single cable

Low-to-moderate-risk region
    -> pseudo-MRG single cable with learned residual correction

Intermediate-risk region
    -> pseudo-MRG double cable

Highly critical region
    -> full MRG double cable
```

The proposed spatial arrangement is:

```text
pseudo-MRG single cable
        |
corrected pseudo-MRG single cable
        |
pseudo-MRG double-cable buffer
        |
full MRG double-cable core
        |
pseudo-MRG double-cable buffer
        |
corrected pseudo-MRG single cable
        |
pseudo-MRG single cable
```

The objective is to preserve the most expensive and detailed model only where it materially affects activation or propagation.

This structure is especially relevant when:

- the number of compartments is small enough that full MRG is affordable locally;
- the total number of fibers or stimulation evaluations is large;
- a pure surrogate is considered too dependent on training coverage;
- and a globally applied full double cable is unnecessarily expensive.

---

## 13. Definition of the Three Model Levels

### 13.1 Pseudo-MRG Single Cable

This is the cheapest level.

It retains:

```text
Vm
MRG-like active membrane currents
MRG-like node, MYSA, FLUT, and STIN compartment labels
effective axial conductance
effective membrane capacitance
effective leakage
effective extracellular coupling
```

It removes:

```text
dynamic periaxonal cable state
full periaxonal axial coupling
2x2 block system
```

A generic equation is:

$$
C_{m,\mathrm{eff},i}
\frac{dV_{m,i}}{dt}
=
I_{\mathrm{axial,eff},i}
-
I_{\mathrm{ion},i}
+
I_{\mathrm{ext,eff},i}
$$

This model should be calibrated against the full MRG double cable.

It is the preferred model for:

- resting regions;
- weak extracellular forcing;
- smooth fields;
- propagation far from initiation;
- large homogeneous fiber batches.

---

### 13.2 Corrected Pseudo-MRG Single Cable

This level uses the analytical pseudo-MRG single-cable model plus a small learned residual.

It retains:

```text
explicit cable dynamics
MRG-like active currents
effective physical parameters
small residual correction
uncertainty estimate
```

It is preferred for:

- moderate extracellular forcing;
- regions where the base pseudo-single model has a known systematic bias;
- waveform classes represented in training;
- large GPU batches requiring higher accuracy than the analytical pseudo-single model.

The correction should not replace the base solver. It should only modify its local prediction.

A possible update is:

$$
V_{m,i}^{t+\Delta t}
=
V_{m,i,\mathrm{pseudo}}^{t+\Delta t}
+
\Delta V_{m,i,\theta}^{t+\Delta t}
$$

with:

$$
\Delta V_{m,i,\theta}^{t+\Delta t}
=
\mathcal{R}_{\theta}
\left(
V_{m,i}^{t},
\mathbf{g}_i^t,
I_{\mathrm{axial},i}^{t},
V_{\mathrm{stim},i}^{t},
\theta_i
\right)
$$

The corrected model should be promoted to pseudo-double when:

- correction magnitude is too large;
- uncertainty is too high;
- rollout disagreement exceeds a threshold;
- or the local state is outside the validated feature domain.

---

### 13.3 Pseudo-MRG Double Cable


This intermediate level retains a periaxonal state but simplifies its dynamics.

Possible simplifications include:

```text
local Vperi dynamics only
reduced periaxonal axial stencil
short-range neighbor coupling
quasi-static local solve
lumped myelin/periaxonal admittance
reduced number of periaxonal state variables
```

A possible local reduced formulation is:

$$
C_m\frac{dV_m}{dt}
=
I_{\mathrm{axial},i}(V_m,V_{\mathrm{peri}})
-
I_{\mathrm{ion}}
+
I_{\mathrm{ext}}
$$

$$
\tau_{\mathrm{peri},i}
\frac{dV_{\mathrm{peri},i}}{dt}
=
-
V_{\mathrm{peri},i}
+
\alpha_iV_{\mathrm{stim},i}
+
\beta_iV_{m,i}
+
\gamma_i\Delta_{\mathrm{local}}V_{\mathrm{peri},i}
$$

This model keeps the main periaxonal feedback mechanism without requiring the exact full MRG block system.

It is the preferred model for:

- transition zones;
- regions near, but not exactly at, the stimulation maximum;
- spike fronts approaching a critical zone;
- moderate surrogate uncertainty;
- regions where the single-cable approximation begins to deviate.

---

### 13.4 Full MRG Double Cable

The full model retains:

```text
Vm
Vperi
full MRG compartment structure
full nodal and internodal dynamics
full intracellular axial coupling
full periaxonal axial coupling
exact coupled linear solve
```

It is reserved for:

- likely spike-initiation sites;
- strong activating-function regions;
- conduction block;
- spike collision;
- abrupt geometry changes;
- branch points;
- high surrogate or reduced-model disagreement;
- out-of-distribution stimulation regimes.

This model remains the reference and fallback level.

---

## 14. Critical-Zone Detection

The initial spatial partition can be derived from the activating function.

For extracellular potential $V_e(x)$:

$$
f_{\mathrm{act}}(x)
\propto
-
\frac{\partial^2 V_e}{\partial x^2}
$$

A discrete implementation is:

```python
dx = x[1] - x[0]

af_inner = -(
    ve[2:] - 2.0 * ve[1:-1] + ve[:-2]
) / dx**2

af = jnp.pad(af_inner, (1, 1))
```

Three fidelity levels can be created with two thresholds:

```python
full_mrg_core = jnp.abs(af) > tau_full
pseudo_double_region = jnp.abs(af) > tau_double
```

Then:

```python
full_mrg_mask = dilate_mask(
    full_mrg_core,
    radius=full_radius,
)

pseudo_double_mask = (
    dilate_mask(
        pseudo_double_region,
        radius=double_buffer_radius,
    )
    & ~full_mrg_mask
)

pseudo_single_mask = ~(
    full_mrg_mask | pseudo_double_mask
)
```

The spatial gradient of the activating function should also be considered:

$$
\left|
\frac{\partial f_{\mathrm{act}}}{\partial x}
\right|
$$

Large gradients may indicate sharp transitions where a higher-fidelity model is required even if the absolute activating-function value is moderate.

---

## 15. Dynamic Fidelity Promotion

The activating function gives an initial partition, but the fidelity map should optionally evolve during simulation.

A compartment may be promoted when:

```python
promote_to_full = (
    (jnp.abs(af) > tau_full)
    | (vm > voltage_alert)
    | (full_disagreement > full_error_threshold)
    | (uncertainty > full_uncertainty_threshold)
    | critical_event_detected
)

promote_to_pseudo_double = (
    (jnp.abs(af) > tau_double)
    | (single_disagreement > single_error_threshold)
    | (uncertainty > double_uncertainty_threshold)
)
```

A possible hierarchy is:

```text
pseudo-MRG single cable
        -> corrected pseudo-MRG single cable
        -> pseudo-MRG double cable
        -> full MRG double cable
```

Promotion should be fast, while demotion should be conservative.

Demotion should require:

- membrane repolarization;
- refractory recovery;
- low uncertainty;
- low disagreement;
- stable behavior for several time steps.

This hysteresis prevents rapid oscillation between model levels.

---

## 16. Coupling Strategy

The safest approach is to preserve one global spatial grid and one global axial coupling representation.

The models should not be treated as isolated fibers with separate boundary conditions.

Instead:

1. maintain one global hybrid state;
2. compute shared or compatible axial fluxes;
3. evaluate candidate model updates;
4. select the appropriate state update by fidelity level;
5. use overlap zones to synchronize states.

At an interface, preserve:

$$
V_m^{\mathrm{left}}
=
V_m^{\mathrm{right}}
$$

and:

$$
I_{\mathrm{axial}}^{\mathrm{left}}
=
I_{\mathrm{axial}}^{\mathrm{right}}
$$

For transitions involving `Vperi`, use an overlap zone in which both the pseudo and full double-cable states are evolved.

---

## 17. Suggested Hybrid State

A common JAX PyTree can contain the union of all state variables:

```python
from typing import NamedTuple
import jax

class HybridMRGState(NamedTuple):
    vm: jax.Array
    vperi: jax.Array
    gates: jax.Array
    pseudo_peri: jax.Array
    latent: jax.Array
    refractory: jax.Array
```

Usage by model level:

```text
Pseudo-MRG single cable:
    vm
    gates
    latent optional

Pseudo-MRG double cable:
    vm
    gates
    pseudo_peri

Full MRG double cable:
    vm
    gates
    vperi
```

Keeping all arrays present avoids changing PyTree structure during `jit`-compiled execution.

---

## 18. Candidate JAX Step

A conceptual implementation is:

```python
def hybrid_mrg_step(
    state,
    fidelity_level,
    stimulation_t,
    geometry,
    pseudo_single_params,
    pseudo_double_params,
    full_mrg_params,
    config,
):
    pseudo_single_next = pseudo_mrg_single_step(
        state,
        stimulation_t,
        geometry,
        pseudo_single_params,
        config.dt,
    )

    pseudo_double_next = pseudo_mrg_double_step(
        state,
        stimulation_t,
        geometry,
        pseudo_double_params,
        config.dt,
    )

    full_mrg_next = full_mrg_double_step(
        state,
        stimulation_t,
        geometry,
        full_mrg_params,
        config.dt,
    )

    next_state = select_fidelity_state(
        pseudo_single_next,
        pseudo_double_next,
        full_mrg_next,
        fidelity_level,
    )

    return next_state
```

For the first implementation, all candidate updates may be evaluated before masking.

This is useful for:

- correctness;
- disagreement measurement;
- training-data generation;
- debugging.

For production, only the required model regions should be evaluated.

---

## 19. Pseudo-MRG Calibration

The pseudo-MRG models should be fitted against the full MRG double-cable solver.

### 19.1 Single-cable calibration targets

Fit:

```text
Cm_eff(x)
Gleak_eff(x)
D_eff(x)
alpha_ext(x)
tau_ext(x)
beta_feedback(x)
```

against:

- subthreshold membrane responses;
- activation thresholds;
- spike initiation location;
- conduction velocity;
- action-potential waveform;
- refractory behavior;
- strength-duration curves.

A possible objective is:

$$
\mathcal{L}_{\mathrm{single}}
=
\lambda_V\mathcal{L}_V
+
\lambda_{\mathrm{thr}}\mathcal{L}_{\mathrm{threshold}}
+
\lambda_c\mathcal{L}_{\mathrm{velocity}}
+
\lambda_e\mathcal{L}_{\mathrm{event}}
$$

### 19.2 Pseudo-double calibration targets

The pseudo-double model should additionally reproduce:

- periaxonal voltage;
- membrane-to-periaxonal feedback;
- block-related dynamics;
- double-cable transient behavior;
- local spatial smoothing.

A possible objective is:

$$
\mathcal{L}_{\mathrm{pseudo-double}}
=
\lambda_V\mathcal{L}_{V_m}
+
\lambda_P\mathcal{L}_{V_{\mathrm{peri}}}
+
\lambda_I\mathcal{L}_{I_{\mathrm{peri}}}
+
\lambda_R\mathcal{L}_{\mathrm{rollout}}
$$

---

## 20. Automatic Learning from Overlap Regions

The explicit full-MRG region can act as a teacher.

In overlap zones, run:

```text
pseudo-MRG single cable
pseudo-MRG double cable
full MRG double cable
```

under the same local conditions.

This provides paired training data:

$$
\mathcal{D}_t
=
\left(
V_m^t,
V_{\mathrm{peri}}^t,
\mathbf{g}^t,
V_{\mathrm{stim}}^t,
\theta,
V_m^{t+\Delta t},
V_{\mathrm{peri}}^{t+\Delta t}
\right)
$$

These samples can be used to:

- recalibrate effective coefficients;
- train a local correction model;
- estimate model disagreement;
- update uncertainty thresholds;
- enrich a replay buffer.

The full model therefore generates the most useful examples automatically.

---

## 21. Disagreement-Based Promotion

Define local disagreement between model levels.

For pseudo-single versus full MRG:

$$
e_i^{\mathrm{single}}
=
\left|
V_{m,i}^{\mathrm{pseudo-single}}
-
V_{m,i}^{\mathrm{full}}
\right|
$$

For pseudo-double versus full MRG:

$$
e_i^{\mathrm{double}}
=
\left|
V_{m,i}^{\mathrm{pseudo-double}}
-
V_{m,i}^{\mathrm{full}}
\right|
$$

A region may be promoted according to:

```python
if e_single > tau_single:
    use pseudo_double

if e_double > tau_double:
    use full_mrg
```

This creates an explicit error-control hierarchy.

---

## 22. Why This Is Preferable to a Pure Surrogate

A pure surrogate may be strongly dependent on its training distribution.

The proposed architecture limits this dependency because:

- the pseudo-MRG models preserve cable structure;
- effective parameters remain interpretable;
- the full model remains available as a fallback;
- overlap regions provide online error measurements;
- difficult states automatically generate training data;
- and model fidelity increases when uncertainty rises.

The objective is therefore not to replace biophysics with machine learning.

It is to use learned or calibrated reductions only where they are reliable.

---

## 23. When Full Double Cable May Still Be Best

If the number of compartments is small, a full JAX double-cable solve may already be sufficiently fast.

The total cost scales approximately as:

$$
C_{\mathrm{total}}
\propto
N_{\mathrm{comp}}
N_{\mathrm{time}}
N_{\mathrm{fibers}}
N_{\mathrm{simulations}}
$$

For small workloads, the multi-fidelity architecture may add more complexity than value.

The proposed method becomes attractive primarily when:

- the number of fibers is large;
- the number of stimulation evaluations is large;
- optimization is repeated many times;
- the simulation is part of an inverse problem;
- or near-real-time throughput is required.

---

## 24. Required Baselines

At minimum, compare:

### Baseline A

```text
Full MRG double cable everywhere
```

### Baseline B

```text
Pseudo-MRG single cable everywhere
```

### Baseline C

```text
Corrected pseudo-MRG single cable everywhere
```

### Baseline D

```text
Pseudo-MRG double cable everywhere
```

### Baseline E

```text
Pseudo-MRG single cable
+ corrected pseudo-MRG single-cable region
+ pseudo-MRG double-cable buffer
+ full MRG double-cable critical core
```

The main result should be an error-runtime Pareto front:

$$
\text{physiological error}
\quad\text{versus}\quad
\text{execution time}
$$

---

## 25. Recommended Development Sequence

### Phase 1

Implement the pseudo-MRG single cable.

Validate:

- thresholds;
- conduction velocity;
- waveform;
- subthreshold response.

### Phase 2

Add a learned residual correction to the pseudo-MRG single cable.

Validate:

- residual magnitude;
- rollout stability;
- out-of-distribution behavior;
- correction uncertainty;
- improvement over the analytical pseudo-single model.

### Phase 3

Implement the pseudo-MRG double cable.

Validate:

- periaxonal feedback;
- transient response;
- conduction block;
- difficult stimulation regimes.

### Phase 4

Create a fixed spatial hierarchy using the activating function.

```text
pseudo-single
    |
pseudo-double
    |
full double
```

### Phase 5

Add overlap regions and disagreement measurements.

### Phase 6

Add dynamic promotion and slow demotion.

### Phase 7

Add active learning and periodic recalibration.

---

## 26. Practical Recommendation

The most practical initial production design is:

```text
Activating function
        |
Full MRG double-cable core
        |
Pseudo-MRG double-cable buffer
        |
Corrected pseudo-MRG single-cable region
        |
Pseudo-MRG single-cable exterior
```

The pseudo-MRG single cable should carry the lowest-risk part of the batch.

The corrected pseudo-MRG single cable should provide most of the accuracy improvement at low computational cost.

The pseudo-MRG double cable should absorb transition and moderate-to-high-risk regions.

The full MRG double cable should remain available for:

- initiation;
- block;
- collision;
- high uncertainty;
- out-of-distribution conditions;
- and validation.

The resulting system is not a pure surrogate solver.

It is an adaptive, physics-preserving, multi-fidelity MRG solver designed for GPU batching.
