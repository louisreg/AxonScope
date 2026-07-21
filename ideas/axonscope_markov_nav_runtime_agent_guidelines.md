# Agent Guidelines: GPU/CPU-Optimized Markov Channel Runtime for AxonScope / AxonFleet

## Objective

Implement support for **Nav1.x custom Markov-channel membrane models** in AxonScope/AxonFleet, optimized for simulations of:

```text
thousands of fibers
myelinated and unmyelinated axons
custom Nav1.x isoforms / mutants / parameter sets
single-cable and double-cable formulations
CPU and GPU execution
large batched stimulation / recruitment / threshold workflows
```

The goal is **not** to integrate Markov models as generic ODE systems. The goal is to treat them as **small local kinetic systems** that can be compiled, precomputed, vectorized, and applied across many compartments and fibers.

The default target model class is:

```text
finite-state voltage-dependent Markov kinetic channel
dp/dt = Q(V) p
I_channel = gbar * P_open(p) * (V - E_rev)
```

where:

```text
p      = vector of state occupancies
Q(V)   = voltage-dependent generator matrix
P_open = sum of occupancies of open states
```

The implementation must support both:

```text
myelinated fibers:
    Nav1.x Markov channels mostly at nodes of Ranvier

unmyelinated fibers:
    Nav1.x Markov channels distributed across all compartments
```

---

## Scientific motivation

The motivating reference is the PLOS Computational Biology paper by Balbi, Massobrio, and Hellgren Kotaleski:

```text
Balbi P, Massobrio P, Hellgren Kotaleski J.
A single Markov-type kinetic model accounting for the macroscopic currents of all human voltage-gated sodium channel isoforms.
PLOS Computational Biology, 2017.
DOI: 10.1371/journal.pcbi.1005737
```

The paper proposes a unified Markov-type kinetic framework for human voltage-gated sodium channels. It uses a parsimonious topology with:

```text
6 states:
    two closed
    two open
    two inactivated

12 transitions
```

The authors emphasize that Markov models can represent channel kinetics more explicitly than Hodgkin-Huxley gates, but that large Markov schemes may become computationally heavy for multicompartmental cells and networks. Their simplified topology is therefore a useful target for AxonScope/AxonFleet: detailed enough to represent Nav1.x isoforms, but small enough to optimize aggressively.

Reference URL:

```text
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005737
```

---

# 1. High-level design principle

## 1.1 Do not treat Markov channels as generic ODEs

Do **not** implement Markov channel updates through:

```text
scipy.solve_ivp
generic ODE solver per compartment
generic dense matrix exponential per time step
Python loops over fibers/compartments
object-per-channel runtime
```

These approaches will not scale to thousands of fibers.

Instead, implement Markov channels as:

```text
compiled local kinetic graphs
precomputed voltage-dependent transition operators
state-major arrays
batched updates over active sites and fibers
```

---

## 1.2 Separate topology, parameters, and runtime

The implementation should distinguish:

```text
MarkovTopology:
    states, transitions, open states

RateLaws:
    voltage-dependent transition functions

ParameterSet:
    isoform-specific or mutation-specific parameters

MarkovChannel:
    topology + rate laws + parameters + conductance metadata

MarkovRuntime:
    precomputed tables + backend update kernels
```

This allows multiple Nav1.x isoforms or mutants to share the same generated update kernel when they share topology, while using different precomputed tables.

Example:

```text
Nav1.6 WT
Nav1.6 mutant A
Nav1.7 WT
Nav1.7 mutant B
```

If they share the same topology and state count:

```text
same compiled kernel
different transition/operator tables
different conductance parameters
```

---

# 2. User-facing API

## 2.1 Proposed `MarkovChannel` object

Add a user-facing membrane-channel object:

```python
@dataclass(frozen=True)
class MarkovTransition:
    source: str
    target: str
    rate: RateLaw
    label: str | None = None


@dataclass(frozen=True)
class MarkovChannel(MembraneChannel):
    name: str
    states: tuple[str, ...]
    transitions: tuple[MarkovTransition, ...]
    open_states: tuple[str, ...]

    gbar: Quantity
    erev: Quantity

    parameter_set: Mapping[str, float] | None = None
    temperature_model: Q10Model | None = None

    backend: MarkovBackend = MarkovBackend.TABLE_EXP
```

The user should be able to define a Nav model as:

```python
Nav16_custom = MarkovChannel(
    name="Nav1.6_custom",
    states=("C1", "C2", "O1", "O2", "I1", "I2"),
    transitions=(
        MarkovTransition("C1", "C2", alpha_C1_C2),
        MarkovTransition("C2", "C1", beta_C2_C1),
        ...
    ),
    open_states=("O1", "O2"),
    gbar=...,
    erev=50.0 * mV,
)
```

Then use it inside a membrane model:

```python
node_membrane = Composite(
    Leak(...),
    Nav16_custom,
    Kv(...),
)
```

For a myelinated axon:

```python
SectionLayout(
    node=node_membrane,
    mysa=Passive(...),
    flut=Passive(...),
    stin=Passive(...),
)
```

For an unmyelinated axon:

```python
axon = Unmyelinated(
    membrane=Composite(
        Leak(...),
        Nav17_custom,
        Kv(...),
    )
)
```

---

## 2.2 Supported Nav1.x use cases

The implementation must support:

```text
one Nav isoform across all fibers
multiple Nav isoforms across fiber subpopulations
custom Nav1.x mutants
different conductance densities per fiber
different temperature settings
different myelinated/unmyelinated layouts
same topology with different parameters
different topologies when needed
```

The runtime should group compatible fibers by:

```text
Markov topology
state count
transition count
rate-law code signature
parameter set / table signature
temperature
dt
active compartment pattern
cable formulation
solver backend
dtype
```

---

# 3. Internal mathematical representation

## 3.1 Generator matrix

For each channel and voltage, build the Markov generator matrix:

```text
dp/dt = Q(V) p
```

Conservation convention:

```text
columns sum to zero if p is a column vector
or rows sum to zero if p is a row vector
```

Pick one convention and enforce it everywhere.

Recommended convention:

```text
p: shape [S, ...]
dp/dt = Q(V) @ p
columns of Q sum to zero
Q[to, from] = rate(from -> to)
Q[from, from] -= rate(from -> to)
```

For a transition:

```text
from = i
to   = j
rate = k_ij(V)

Q[j, i] += k_ij(V)
Q[i, i] -= k_ij(V)
```

## 3.2 Open probability

Use an open-state mask:

```text
open_mask[state] = 1 if state is open else 0
```

Then:

```text
P_open = sum_s open_mask[s] * p[s]
```

The channel current is:

```text
I_Na = gbar * P_open * (V - E_Na)
```

The conductance contribution is:

```text
g_Na = gbar * P_open
```

---

# 4. Runtime backends

Implement three Markov backends.

---

## 4.1 Backend A — `MARKOV_TABLE_EXP`

### Role

This should be the default backend for small-to-medium Nav1.x Markov models.

### Idea

Precompute the local evolution operator:

```text
M(V, dt) = exp(dt * Q(V))
```

on a voltage grid.

At runtime:

```text
p_next = M(V) p
```

with linear interpolation between voltage bins.

### Voltage grid

Recommended initial grid:

```text
V_min = -120 mV
V_max = +80 mV
dV    = 0.25 mV or 0.5 mV
```

Use configuration:

```python
@dataclass(frozen=True)
class MarkovTableConfig:
    v_min_mV: float = -120.0
    v_max_mV: float = 80.0
    dV_mV: float = 0.25
    interpolation: Literal["nearest", "linear"] = "linear"
    clamp_out_of_range: bool = True
```

### Precomputed tables

For each compiled channel signature:

```text
V_grid[V]
rates_table[V, K]
Q_table[V, S, S]           optional debug/reference
M_table[V, S, S]           default runtime table
p_inf_table[V, S]          initialization table
open_mask[S]
```

For optimized runtime, also create an SoA table:

```text
M_s_to_t[S, S, V]
```

or flattened:

```text
M_flat[S*S, V]
```

### Advantages

```text
stable
fast
no ODE solve at runtime
no matrix exponential at runtime
excellent for small S such as 6 states
works on CPU and GPU
easy to validate against implicit/local exact update
```

### Disadvantages

```text
requires fixed dt for the table
requires a voltage grid
interpolation error must be validated
table must be rebuilt if dt, temperature, or parameters change
```

---

## 4.2 Backend B — `MARKOV_SPARSE_EDGE`

### Role

Use for large sparse Markov models where dense `M(V)` tables become too expensive.

### Representation

```text
from_state[K]
to_state[K]
rate_table[V, K] or rate_law_e(V)
```

### Explicit flux update

```text
flux_e = rate_e(V) * p[from_e]

p[to_e]   += dt * flux_e
p[from_e] -= dt * flux_e
```

### Advantages

```text
low memory for large sparse models
simple transition-graph representation
good for exploratory custom topologies
```

### Disadvantages

```text
can be unstable for large dt or stiff transitions
may produce negative probabilities
requires smaller dt or stabilized schemes
less robust than TABLE_EXP
```

### Recommended status

Experimental / fallback for large state spaces. Not the default for Nav1.x six-state models.

---

## 4.3 Backend C — `MARKOV_IMPLICIT_LOCAL`

### Role

Reference and validation backend.

### Methods

Backward Euler:

```text
(I - dt Q(V)) p_next = p
```

Crank-Nicolson:

```text
(I - dt/2 Q(V)) p_next = (I + dt/2 Q(V)) p
```

For small `S`, this is a local `S x S` solve per active site.

### Advantages

```text
robust
good validation target
no voltage-grid interpolation error
```

### Disadvantages

```text
more expensive
generic small solves may not be GPU efficient
```

### Recommended status

Use for:

```text
validation
debugging
checking TABLE_EXP interpolation error
stiff custom models
```

---

# 5. State layout and memory organization

## 5.1 State-major layout

Use state-major arrays.

For active sites:

```text
p[state, site, batch]
```

or in tiled form:

```text
p[channel_group, state, site, batch]
p[state, tile, site_pad, B_tile]
```

Avoid:

```text
p[batch, site, state]
```

unless benchmarks prove otherwise.

Reason:

```text
each state is a large contiguous array
component-wise generated updates are simple
good match for SoA codegen
```

---

## 5.2 Myelinated fibers

For myelinated fibers, Nav1.x channels are usually concentrated at nodes of Ranvier.

Do not allocate Markov states for every compartment if the channel is only present at nodes.

Use:

```text
p_nav[state, node_index, batch]
V_node[node_index, batch]
```

not:

```text
p_nav[state, Nx, batch]
```

This is essential for performance.

### Example

If:

```text
Nx = 100
N_nodes = 11
S = 6
B = 4096
```

then:

```text
node-only states:
    6 * 11 * 4096 values

full-compartment states:
    6 * 100 * 4096 values
```

Node-only storage is about 9x smaller.

---

## 5.3 Unmyelinated fibers

For unmyelinated fibers, channels may be distributed across all compartments.

Use:

```text
p_nav[state, Nx, batch]
V[Nx, batch]
```

or tiled:

```text
p_nav[state, n_tiles, Nx_pad, B_tile]
```

This remains GPU-friendly if stored state-major.

---

## 5.4 Multiple Nav1.x isoforms or custom mutants

Group by compiled channel signature:

```text
same topology
same rate-law code
same parameter set
same temperature model
same dt
```

For example:

```text
Group 1:
    Nav1.6 WT
    p[state, active_sites, B1]
    M_table_nav16_wt

Group 2:
    Nav1.6 mutant A
    p[state, active_sites, B2]
    M_table_nav16_mutA

Group 3:
    Nav1.7 WT
    p[state, active_sites, B3]
    M_table_nav17_wt
```

Do not mix many parameter sets inside one kernel unless a parameter-batched table strategy is explicitly implemented.

---

# 6. Component-wise SoA update

## 6.1 Avoid generic small matrix multiplication

Do not rely on generic:

```python
p_next = M @ p
```

for millions of tiny vectors.

For small fixed state counts such as `S=6`, generate explicit component-wise updates:

```text
p0_next = M00*p0 + M01*p1 + M02*p2 + M03*p3 + M04*p4 + M05*p5
p1_next = M10*p0 + M11*p1 + M12*p2 + M13*p3 + M14*p4 + M15*p5
...
p5_next = M50*p0 + M51*p1 + M52*p2 + M53*p3 + M54*p4 + M55*p5
```

This is better for:

```text
JAX/XLA
Pallas
CUDA
CPU vectorization
```

because it avoids tiny matrix abstractions and gives the compiler simple array operations.

---

## 6.2 Interpolated table lookup

Given voltage `V`, compute:

```text
f = (V - V_min) / dV
i0 = floor(f)
w  = f - i0
i1 = i0 + 1
```

Clamp:

```text
i0 = clamp(i0, 0, nV - 2)
i1 = i0 + 1
w  = clamp(w, 0, 1)
```

Interpolate each matrix coefficient:

```text
Mij = (1 - w) * Mij_table[i0] + w * Mij_table[i1]
```

Then update:

```text
p_next = M(V) p
```

---

## 6.3 Probability conservation and numerical cleanup

After update, optionally enforce:

```text
sum(p_next) ≈ 1
p_next >= 0
```

Recommended production behavior:

```text
debug mode:
    check sum and negativity
    report warnings or raise

fast mode:
    no correction unless needed

safe mode:
    clamp small negatives
    renormalize by sum
```

Suggested tolerance:

```text
sum error:
    < 1e-5 for float32
    < 1e-10 for float64

negative states:
    tolerate tiny values such as > -1e-7 in float32
```

Do not silently clamp large negative values; that indicates a bad table, bad dt, or unstable backend.

---

# 7. Initialization

## 7.1 Steady-state initialization

For each voltage grid value, precompute:

```text
p_inf(V)
```

where:

```text
Q(V) p_inf = 0
sum(p_inf) = 1
```

Then initialize:

```text
p_initial = p_inf(V_initial)
```

For myelinated fibers:

```text
p_nav[:, node, batch] = p_inf(V_node_initial)
```

For unmyelinated fibers:

```text
p_nav[:, compartment, batch] = p_inf(V_initial)
```

## 7.2 Protocol-specific initialization

Support optional initialization modes:

```python
class MarkovInitialization(str, Enum):
    STEADY_STATE = "steady_state"
    USER_PROVIDED = "user_provided"
    ALL_CLOSED = "all_closed"
    CUSTOM_PROTOCOL = "custom_protocol"
```

For voltage-clamp validation, user-provided or protocol-conditioned initial states may be necessary.

---

# 8. Coupling Markov channels to the cable solver

## 8.1 Preferred first implementation: operator splitting

At each time step:

```text
1. Given V_n, update Markov states:
       p_{n+1} = M(V_n) p_n

2. Compute open probability:
       P_open = sum_open p_{n+1}

3. Compute channel conductance:
       g_Na = gbar * P_open

4. Build cable linear system:
       diag += g_Na
       rhs  += g_Na * E_Na
       plus other membrane/source terms

5. Solve cable voltage:
       V_{n+1}
```

This is simple, fast, and easy to validate.

---

## 8.2 More accurate mode: predictor-corrector

Add an optional accurate mode:

```text
1. update Markov using V_n
2. build current/conductance
3. solve V_pred
4. update Markov using V_mid = 0.5*(V_n + V_pred) or V_pred
5. rebuild current/conductance
6. solve corrected V_{n+1}
```

Suggested modes:

```python
class MarkovVoltageCoupling(str, Enum):
    EXPLICIT_VN = "explicit_vn"
    MIDPOINT_PREDICTOR = "midpoint_predictor"
    PREDICTOR_CORRECTOR = "predictor_corrector"
```

Start with:

```text
EXPLICIT_VN
```

then validate against:

```text
PREDICTOR_CORRECTOR
MARKOV_IMPLICIT_LOCAL
NEURON/KINETIC reference if available
```

---

## 8.3 Semi-implicit conductance contribution

For the cable solve, treat the channel contribution as a conductance at the current step:

```text
I_Na = g_Na(V_n, p_n) * (V - E_Na)
```

Then place:

```text
diag += g_Na
rhs  += g_Na * E_Na
```

This is analogous to using a conductance-based linearized contribution.

For strong nonlinear coupling or large dt, validate against more accurate coupling.

---

# 9. Runtime grouping and dispatch

## 9.1 Grouping key

Add Markov-channel-specific fields to the execution bucket key:

```python
@dataclass(frozen=True)
class MarkovRuntimeKey:
    topology_hash: str
    rate_law_hash: str
    parameter_hash: str
    state_count: int
    transition_count: int
    dt: float
    temperature: float
    v_grid_signature: str
    backend: MarkovBackend
    active_site_pattern: str
```

Combine with existing execution keys:

```text
cable formulation
Nx_pad
dtype
solver backend
recording mode
stimulation representation
geometry signature
```

---

## 9.2 Myelinated and unmyelinated grouping

Do not force myelinated and unmyelinated fibers into the same Markov update shape.

Use separate runtime groups:

```text
myelinated node-only Markov groups:
    p[state, node, batch]

unmyelinated full-compartment Markov groups:
    p[state, Nx, batch]
```

This avoids wasting work on inactive compartments in myelinated fibers.

---

## 9.3 Handling many custom mutants

### Few parameter sets

If there are a few discrete parameter sets:

```text
precompute one M_table per parameter set
run one group per parameter set
```

### Many parameter sets

If there are many unique parameter sets:

Option A:

```text
group approximate parameter sets if scientifically acceptable
```

Option B:

```text
parameter-batched table generation:
    M_table[param, V, S, S]
```

Option C:

```text
fallback to sparse-edge rate evaluation
```

Start with Option A/B only if needed. The default should be one table per parameter set.

---

# 10. CPU implementation strategy

## 10.1 Use the same table-exp backend

The CPU backend should use the same mathematical approach:

```text
precompute M(V, dt)
state-major arrays
component-wise SoA update
```

Avoid:

```text
Python loops over compartments/fibers
generic ODE solve per site
generic tiny matrix operations per site
```

## 10.2 Backend choices

Recommended order:

```text
1. JAX CPU vectorized implementation
2. NumPy vectorized implementation for reference/debug
3. Numba/C++ only if needed
```

For small fixed `S`, generated component-wise code is likely faster than generic batched matrix multiplication.

---

# 11. GPU implementation strategy

## 11.1 JAX implementation first

Implement:

```text
MARKOV_TABLE_EXP_JAX
```

using:

```text
state-major arrays
SoA M_table
component-wise updates
node-only active sites for myelinated fibers
```

This validates the design without custom kernels.

## 11.2 Pallas implementation second

If JAX traces show too many temporaries or poor fusion, implement:

```text
MARKOV_TABLE_EXP_PALLAS
```

Mapping:

```text
one program handles a tile of active sites × batch
states S fixed at compile time
channel model fixed per compiled kernel
```

Pseudo-kernel:

```text
load V[site, batch]
compute voltage table index
load M coefficients
load p0..pS
compute p_next0..p_nextS
compute P_open
write p_next
write g_channel or current contribution
```

## 11.3 CUDA FFI only if necessary

Use CUDA FFI only if:

```text
Pallas is insufficient
Markov update becomes a proven bottleneck
state count/topology is stable enough to justify low-level code
```

---

# 12. Output and observers for Markov channels

## 12.1 Raw outputs

Allow optional recording of:

```text
state occupancies p_s
open probability P_open
channel conductance g_Na
channel current I_Na
availability / inactivation-state occupancy
```

## 12.2 Compact observers

For large simulations, avoid recording full Markov state histories unless needed.

Useful Markov-specific observers:

```text
peak P_open
time of peak P_open
mean nodal availability
inactivated fraction at stimulus onset
recovery from inactivation metric
open-state integral
Nav current integral
```

## 12.3 Recording policy

Default for large fiber populations:

```text
do not record full p[state, time, site, batch]
record only requested summaries or selected probes
```

Full Markov state traces should be debug/validation mode.

---

# 13. Validation plan

## 13.1 Local Markov validation

For each Markov channel:

```text
1. Check Q(V) conservation:
       columns sum to zero.

2. Check M(V) stochasticity:
       columns sum to one.

3. Check positivity:
       M(V) entries should be non-negative within numerical tolerance.

4. Check p_inf:
       Q(V) p_inf ≈ 0
       sum(p_inf) = 1.
```

## 13.2 Table-exp validation

Compare:

```text
MARKOV_TABLE_EXP
vs
MARKOV_IMPLICIT_LOCAL
vs
direct scipy expm reference
```

Voltage grid:

```text
V = -120 to +80 mV
dt = target dt values
```

Metrics:

```text
max_abs_error in p_next
sum(p) error
P_open error
current error
```

## 13.3 Voltage-clamp protocol validation

Reproduce standard protocols:

```text
activation curve
availability / steady-state inactivation
recovery from inactivation
persistent current if relevant
current-voltage relationship
```

Compare against:

```text
published curves
Balbi et al. reference implementation / ModelDB if available
NEURON KINETIC implementation if available
```

## 13.4 Cable simulation validation

For axon simulations:

```text
single-fiber spike waveform
threshold
conduction velocity
refractory behavior
recruitment curve
myelinated vs unmyelinated behavior
```

Compare:

```text
HH approximation
direct Markov implicit local backend
NEURON reference if available
```

---

# 14. Benchmark plan

## 14.1 Markov update benchmark

Create:

```text
benchmark/membranes/bench_markov_channel_update.py
```

Matrix:

```text
S = 6, 8, 12, 20
active_sites = 8, 16, 32, 64, 128
B = 512, 1024, 2048, 4096, 8192
dtype = float32, float64
backend = table_exp_jax, sparse_edge, implicit_local, pallas
layout = state_site_batch, state_tile_site_batch
```

Metrics:

```text
time per update
updates/s = active_sites × B / time
memory bandwidth proxy
P_open computation time
state normalization error
kernel count
```

## 14.2 End-to-end benchmark

Create:

```text
benchmark/axons/bench_markov_nav_fiber_population.py
```

Cases:

```text
myelinated node-only Nav1.x Markov
unmyelinated full-compartment Nav1.x Markov
single-cable Markov
double-cable Markov
multiple Nav parameter sets
threshold sweep with amplitudes in batch
```

Metrics:

```text
total simulation time
Markov update time
cable solver time
memory usage
output size
threshold/recruitment agreement
```

---

# 15. Implementation phases for the agent

## Phase 1 — Data model and validation utilities

Implement:

```text
MarkovTransition
MarkovChannel
MarkovTopology
RateLaw
MarkovRuntimeKey
```

Add validators:

```text
state names unique
transitions reference valid states
open states valid
rates positive on voltage grid
Q matrix conservation
```

## Phase 2 — Table precomputation

Implement:

```text
build_Q_table
build_M_table
build_p_inf_table
build_M_table_SoA
```

Use SciPy on CPU for precomputation initially:

```python
scipy.linalg.expm
```

because this occurs once per channel signature, not at every time step.

## Phase 3 — JAX runtime update

Implement:

```text
markov_table_exp_update_jax
markov_open_probability
markov_current_contribution
```

Use state-major SoA arrays.

## Phase 4 — Integration with membrane DSL/runtime

Connect `MarkovChannel` to:

```text
Composite membranes
SectionLayout
node-only active site extraction
unmyelinated full-compartment update
runtime state initialization
recording/observers
```

## Phase 5 — Coupling to cable solvers

Implement:

```text
EXPLICIT_VN coupling
PREDICTOR_CORRECTOR coupling
semi-implicit conductance contribution
```

Support both:

```text
single-cable
double-cable
```

## Phase 6 — Grouping and batching

Add Markov signatures to dispatch keys.

Implement separate groups for:

```text
myelinated node-only Markov
unmyelinated full-compartment Markov
different Nav1.x parameter sets
different dt/temperature/table configs
```

## Phase 7 — Benchmarks and validation

Add the benchmark and validation scripts described above.

## Phase 8 — Pallas optimization

Only after JAX implementation is correct and profiled.

Implement:

```text
MARKOV_TABLE_EXP_PALLAS
```

if Markov updates are a bottleneck.

## Phase 9 — Advanced backends

Optional:

```text
MARKOV_SPARSE_EDGE
MARKOV_IMPLICIT_LOCAL optimized
CUDA_FFI_MARKOV_TABLE_EXP
```

---

# 16. Go / no-go criteria

## Keep `MARKOV_TABLE_EXP` as default if:

```text
P_open/current error vs implicit local backend is small
probability conservation is stable
voltage-clamp protocol curves match reference
end-to-end simulations match expected spike behavior
runtime is significantly faster than generic local ODE/implicit solve
```

## Add Pallas if:

```text
JAX implementation is correct
Markov update is >20% of total runtime
JAX trace shows large temporary arrays or poor fusion
Pallas gives >1.3x speedup for Markov update
```

## Use sparse-edge backend if:

```text
state count is large
transition graph is sparse
dense M_table becomes too memory-heavy
dt is small enough for stable explicit/semi-implicit updates
```

## Use implicit local backend if:

```text
model is stiff
table interpolation error is unacceptable
validation/reference mode is needed
```

---

# 17. Practical defaults

Recommended default configuration:

```python
MarkovTableConfig(
    v_min_mV=-120.0,
    v_max_mV=80.0,
    dV_mV=0.25,
    interpolation="linear",
    clamp_out_of_range=True,
)
```

Recommended runtime defaults:

```text
backend:
    MARKOV_TABLE_EXP

state layout:
    state-major SoA

myelinated:
    node-only Markov update

unmyelinated:
    full-compartment Markov update

voltage coupling:
    EXPLICIT_VN initially
    PREDICTOR_CORRECTOR available for accuracy checks

recording:
    P_open and current optional
    full state traces off by default
```

---

# 18. Summary for the agent

The implementation should make Markov Nav1.x channels a first-class membrane mechanism in AxonScope/AxonFleet.

The critical design choices are:

```text
1. Use MarkovChannel as a compiled transition graph.
2. Precompute M(V, dt) = exp(dt Q(V)) on a voltage grid.
3. Store channel states in state-major SoA arrays.
4. Update only active sites:
       nodes for myelinated fibers,
       all compartments for unmyelinated fibers.
5. Group by Markov topology / parameter set / dt / temperature.
6. Apply component-wise generated updates, not generic tiny matmul.
7. Couple to cable solvers through P_open-derived conductance/current.
8. Validate against implicit local solve and voltage-clamp protocols.
9. Use Pallas/CUDA only after JAX table-exp is correct and profiled.
```

This design keeps detailed Nav1.x Markov kinetics while remaining compatible with the main AxonScope/AxonFleet goal:

```text
simulate thousands of heterogeneous nerve fibers efficiently on CPU/GPU
```
