# AxonScope Demo Roadmap: Custom Axon Models from Reusable Membrane Models

**Goal:** demonstrate that AxonScope is not only a simulator for predefined axon templates, but a framework where axonal models can be assembled from reusable membrane components.

The central message of these demos is:

> **A membrane model is local; AxonScope turns it into a spatially extended axon model.**

A membrane model may come from:
- a hand-written Hodgkin-Huxley-style current;
- a Channelpedia / patch-clamp-derived channel model;
- a ModelDB / NeuroML / NEURON-derived membrane mechanism;
- a custom equation written directly by the user.

AxonScope should make it possible to place these membranes into different axonal structures:
- a myelinated MRG-like axon;
- an MRG node of Ranvier only;
- an unmyelinated C-fiber;
- a hybrid or pathological axon.

This document proposes three complementary demos:

1. **MRG focal demyelination**
   Custom structural/pathological modification of a myelinated axon.

2. **MRG with Channelpedia-derived Nav nodes**
   Custom nodal membrane biophysics in an otherwise MRG-like fiber.

3. **Channelpedia-derived sensory C-fiber**
   Assembly of a credible unmyelinated C-fiber from sensory-neuron-like membrane channels.

Together, they show three levels of customization:

| Demo | What changes? | What stays reusable? | Main message |
|---|---|---|---|
| 1. MRG demyelination | Local myelin parameters | MRG nodal membrane and solver | AxonScope can represent local pathology |
| 2. MRG + Nav channels | Nodal membrane channels | MRG geometry and myelin architecture | AxonScope can replace membrane biophysics |
| 3. Channelpedia C-fiber | Full membrane composition and axon class | Same solver interface | AxonScope can turn membrane libraries into axon models |

---

## 0. Common design principle

All examples should be built around a strict separation between:

```text
Membrane model  →  Axon template  →  Stimulation protocol  →  Solver  →  Metrics
```

The membrane model should expose a small, solver-facing interface:

```python
class AxonScopeMembrane:
    state_names: tuple[str, ...]
    current_names: tuple[str, ...]

    def initial_state(self, v, temperature=None):
        """Return initial gating variables and other internal states."""

    def currents(self, v, state, temperature=None):
        """Return ionic currents, usually in A/m² or compatible units."""

    def rhs(self, v, state, temperature=None):
        """Return time derivatives of the state variables."""

    @property
    def capacitance(self):
        """Return membrane capacitance per unit area."""
```

The solver should not need to know whether the membrane comes from MRG, Channelpedia, ModelDB, or a user-defined Python class. It should only need:

```text
I_ion(Vm, state)
dstate/dt = f(Vm, state)
Cm
```

A useful abstraction is a `CompositeMembrane`:

```python
membrane = axs.CompositeMembrane(
    capacitance=1.0 * axs.uF / axs.cm**2,
    channels=[
        Nav17.with_density(80 * axs.S / axs.m**2),
        Nav18.with_density(300 * axs.S / axs.m**2),
        Kv7.with_density(5 * axs.S / axs.m**2),
        axs.Leak(g=0.05 * axs.S / axs.m**2, e=-60 * axs.mV),
    ],
)
```

This should be the conceptual backbone of the three demos.

---

# 1. Demo 1 — MRG focal demyelination

## Suggested file name

```text
examples/advanced/custom_axons/01_mrg_focal_demyelination.py
```

or, for a notebook:

```text
docs/examples/01_mrg_focal_demyelination.ipynb
```

## Scientific pitch

This demo starts from a standard MRG-like myelinated axon and introduces a focal demyelinated region. The membrane model can remain MRG-like, but the local myelin properties are altered in selected internodes.

The user story is:

> We keep a validated myelinated axon architecture, then introduce a local myelin lesion by degrading the electrical insulation of selected internodal regions. Increasing the lesion severity slows conduction and eventually produces conduction block.

This is a strong first example because it is visual, intuitive, and axon-specific.

## Biological and modeling background

The McIntyre-Richardson-Grill model is a double-cable model of a mammalian myelinated axon. It explicitly represents nodes of Ranvier, paranodal regions, internodal regions, and a finite-impedance myelin sheath. This makes it much more suitable for local myelin manipulations than a simple single-cable myelinated approximation.

Demyelination can be modeled as a local degradation of the insulating properties of the myelin sheath. At a minimal electrical level, this means:

```text
demyelination → myelin resistance decreases
demyelination → myelin conductance increases
demyelination → effective myelin capacitance increases
```

This is a deliberately simple representation. It does not attempt to model the full biology of inflammatory demyelination, remyelination, nodal remodeling, sodium-channel redistribution, or ephaptic coupling. Instead, it captures the first-order effect: a damaged myelin sheath leaks more current and presents a larger capacitive load.

## Model construction

Start with a healthy MRG-like fiber:

```python
healthy = axs.MRGFiber(
    diameter=10 * axs.um,
    n_nodes=51,
)
```

Then define a local lesion:

```python
lesion = axs.FocalDemyelination(
    center_node=25,
    n_internodes=3,
    severity=0.0,  # 0 = healthy, 1 = severe
)
```

Apply it:

```python
demyelinated = healthy.with_local_modification(lesion)
```

or:

```python
demyelinated = axs.MRGFiber(
    diameter=10 * axs.um,
    n_nodes=51,
    modifiers=[
        axs.FocalDemyelination(
            center_node=25,
            n_internodes=3,
            severity=0.8,
        )
    ],
)
```

## Suggested implementation of severity

Use a bounded severity parameter:

```text
severity = 0.0 → healthy myelin
severity = 0.3 → mild demyelination
severity = 0.6 → moderate demyelination
severity = 0.9 → severe demyelination
```

Then convert it to an electrical scaling factor:

```python
def demyelination_factor(severity, min_remaining_myelin=0.05):
    remaining = max(1.0 - severity, min_remaining_myelin)
    return 1.0 / remaining
```

For affected `MYSA`, `FLUT`, and `STIN` sections:

```python
factor = demyelination_factor(severity)

section.c_myelin *= factor
section.g_myelin *= factor
section.r_myelin /= factor
```

The exact variables will depend on AxonScope's internal representation, but the principle is:

```text
myelin capacitance ↑
myelin conductance ↑
myelin resistance ↓
```

## Conditions to simulate

Run the same stimulation protocol on several lesion severities:

```python
severities = [0.0, 0.4, 0.7, 0.9, 0.97]
```

For each condition:

```python
axon = axs.MRGFiber(
    diameter=10 * axs.um,
    n_nodes=51,
).with_demyelination(
    center_node=25,
    n_internodes=3,
    severity=severity,
)
```

Use either:
- intracellular current injection at one node, to focus on propagation; or
- extracellular stimulation, to connect directly to neurostimulation applications.

For a first demo, intracellular initiation is cleaner:

```python
stim = axs.IntracellularStimulus.pulse(
    node=5,
    start=0.5 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=2.0 * axs.nA,
)
```

For a stimulation-oriented variant:

```python
stim = axs.ExtracellularStimulus.point_source(
    electrode_position=(0, 100 * axs.um, 0),
    waveform=axs.Stimulus.pulse(
        start=0.5 * axs.ms,
        duration=100 * axs.us,
        amplitude=-100 * axs.uA,
    ),
)
```

## Metrics

Measure:

```text
- propagation success / failure
- spike arrival time after the lesion
- conduction velocity before the lesion
- conduction velocity after the lesion
- peak Vm at each node
- safety factor proxy
- threshold current versus demyelination severity
```

A simple propagation criterion:

```python
success = result.vm[node_after_lesion].max() > 0 * axs.mV
```

Activation time at node `i`:

```python
t_activation[i] = first_time_crossing(result.vm[i], threshold=-20 * axs.mV)
```

Conduction velocity:

```python
cv = distance_between_nodes / (t_activation[j] - t_activation[i])
```

## Figures

Recommended figures:

1. **Axon layout**
   A schematic showing nodes, internodes, and the demyelinated region.

2. **Vm(x, t) heatmap**
   The most important plot. It should show normal propagation for low severity and conduction block for high severity.

3. **Peak Vm versus position**
   Shows attenuation around the lesion.

4. **Activation time versus position**
   Shows conduction slowing around the lesion.

5. **Threshold versus severity**
   Shows that demyelination increases activation threshold or prevents propagation.

## Expected qualitative result

```text
severity = 0.0   → normal saltatory propagation
severity = 0.4   → slightly slower conduction
severity = 0.7   → delayed/attenuated propagation
severity = 0.9   → unreliable propagation
severity = 0.97  → conduction block
```

## What this demo proves

This demo proves that AxonScope can represent local structural modifications without rewriting the solver.

The key message is:

> The user can start from a standard MRG fiber and locally modify the physical properties of the myelin sheath to create a pathological axon model.

## Caveats

This should be explicitly stated:

- This is a first-order electrical demyelination model.
- It does not yet include sodium-channel redistribution.
- It does not model remyelination.
- It does not model immune activity, glial remodeling, or ephaptic coupling.
- The goal is to demonstrate local axon model customization, not to claim a full disease model.

## Possible extension

Add a second mode:

```python
mode="exposed_axolemma"
```

where a severely demyelinated internode is replaced by a bare axolemma membrane:

```text
myelinated internode → exposed passive axolemma
```

or:

```text
myelinated internode → exposed axolemma with low-density Nav/Kv channels
```

This should be left as an advanced extension because it requires assumptions about ion-channel redistribution.

---

# 2. Demo 2 — MRG with Channelpedia-derived Nav nodes

## Suggested file name

```text
examples/advanced/custom_axons/02_mrg_channelpedia_nav_nodes.py
```

## Scientific pitch

This demo keeps the MRG myelinated architecture but replaces the nodal membrane with user-defined voltage-gated sodium channel models, ideally derived from Channelpedia or patch-clamp-style Hodgkin-Huxley fits.

The user story is:

> We keep the MRG geometry and myelin architecture, but replace the nodal sodium current by Channelpedia-derived Nav channels. This allows us to test how Nav subtype choice or disease-like gating shifts affect threshold, spike shape, and propagation.

This is the most direct demonstration of AxonScope's membrane modularity in a myelinated axon.

## Biological motivation

The original MRG model uses nodal ionic currents designed to reproduce mammalian myelinated axon excitability. However, from a biological point of view, it is interesting to replace abstract nodal sodium dynamics with more explicit Nav subtype-based channels.

A biologically defensible starting point is:

```text
Nav1.6-like channel at nodes of Ranvier
```

Nav1.6 is strongly associated with nodes of Ranvier in both peripheral and central axons. This makes it the most natural Channelpedia-derived candidate for a nodal MRG replacement.

You can then compare:

```text
standard MRG node
Nav1.6-like node
Nav1.7-like node
Nav1.6 gain-of-function variant
Nav1.6 loss-of-function variant
```

A Nav1.7-like condition is not necessarily a canonical large-fiber nodal model, but it is useful pedagogically because Nav1.7 has strong links with sensory-neuron excitability and pain phenotypes.

## Model construction

Start with a Channelpedia-derived sodium channel:

```python
Nav16 = axs.ChannelpediaHHChannel.from_name("Nav1.6")
```

or, if the real Channelpedia integration is not implemented yet:

```python
Nav16 = axs.HHChannel.from_parameters(
    name="Nav1.6_like",
    current="na",
    gates=[
        axs.ActivationGate("m", power=3, vhalf=-32 * axs.mV, slope=6 * axs.mV),
        axs.InactivationGate("h", power=1, vhalf=-60 * axs.mV, slope=-6 * axs.mV),
    ],
    reversal=55 * axs.mV,
)
```

Then build a nodal membrane:

```python
node_membrane = axs.CompositeMembrane(
    capacitance=2.0 * axs.uF / axs.cm**2,
    channels=[
        Nav16.with_density(3000 * axs.S / axs.m**2),
        axs.MRGFastK().with_density(80 * axs.S / axs.m**2),
        axs.MRGPersistentNa().with_density(10 * axs.S / axs.m**2),
        axs.Leak(g=7 * axs.S / axs.m**2, e=-80 * axs.mV),
    ],
)
```

Then insert it into the MRG fiber:

```python
axon = axs.MRGFiber(
    diameter=10 * axs.um,
    n_nodes=51,
    node_membrane=node_membrane,
)
```

The important conceptual point is that only the node membrane is replaced:

```text
NODE membrane       → custom Channelpedia-derived membrane
MYSA / FLUT / STIN  → MRG-like structure unchanged
myelin sheath       → MRG-like structure unchanged
solver              → unchanged
```

## Variants to compare

### 2.1 Standard MRG

```python
standard = axs.MRGFiber(
    diameter=10 * axs.um,
    n_nodes=51,
)
```

### 2.2 Nav1.6-like node

```python
nav16_node = make_node_membrane(
    sodium_channel=Nav16,
    sodium_density=3000 * axs.S / axs.m**2,
)
```

### 2.3 Gain-of-function Nav-like node

A simple gain-of-function perturbation:

```python
nav16_gof = Nav16.shift_gating(
    activation=-5 * axs.mV,
    inactivation=+5 * axs.mV,
).with_persistent_fraction(0.01)
```

Interpretation:

```text
activation shifted to more negative potentials → easier opening
inactivation shifted to more positive potentials → more availability
small persistent current → enhanced excitability
```

### 2.4 Loss-of-function Nav-like node

```python
nav16_lof = Nav16.shift_gating(
    activation=+5 * axs.mV,
    inactivation=-5 * axs.mV,
).scale_conductance(0.5)
```

Interpretation:

```text
activation shifted to more positive potentials → harder opening
inactivation shifted to more negative potentials → less availability
lower density → weaker sodium current
```

## Stimulation protocols

Use three protocols.

### Protocol A — single pulse propagation

Stimulate near one end and measure conduction:

```python
stim = axs.IntracellularStimulus.pulse(
    node=5,
    start=0.5 * axs.ms,
    duration=0.1 * axs.ms,
    amplitude=2.0 * axs.nA,
)
```

Metrics:

```text
- spike shape at several nodes
- conduction velocity
- propagation success
```

### Protocol B — threshold search

Use a binary search for threshold:

```python
threshold = axs.find_threshold(
    axon=axon,
    stimulus_template=stim_template,
    target_node=40,
    criterion=lambda r: r.vm[40].max() > 0 * axs.mV,
)
```

Metrics:

```text
- threshold current
- threshold charge
- rheobase / chronaxie if pulse-width sweep is included
```

### Protocol C — recovery / refractory behavior

Deliver paired pulses:

```python
paired = axs.Stimulus.paired_pulse(
    first_start=0.5 * axs.ms,
    second_delay=delay,
    duration=0.1 * axs.ms,
    amplitude=1.2 * threshold,
)
```

Metric:

```text
- minimum inter-pulse interval producing a second propagated spike
```

This is useful because MRG was originally designed to reproduce recovery-cycle behavior in mammalian myelinated fibers.

## Figures

Recommended figures:

1. **Imported channel curves**
   Plot `m_inf(V)`, `h_inf(V)`, and optionally `tau_m(V)`, `tau_h(V)` for each Nav variant.

2. **Nodal spike waveforms**
   Compare standard MRG, Nav1.6-like, GOF, and LOF nodes.

3. **Threshold comparison**
   Bar plot or table.

4. **Conduction velocity comparison**
   Shows whether the custom membrane still supports stable saltatory conduction.

5. **Paired-pulse recovery**
   Shows how modified Nav kinetics affect refractory behavior.

## Expected qualitative result

```text
standard MRG      → stable propagation, reference threshold
Nav1.6-like       → stable propagation after density tuning
GOF Nav-like      → lower threshold, shorter latency, possibly afterdepolarization
LOF Nav-like      → higher threshold, weaker propagation, possible failure
```

## What this demo proves

This demo proves that AxonScope can decouple:

```text
myelinated axon geometry
from
nodal membrane biophysics
```

The key message is:

> Users can keep a validated MRG-like axon structure and replace only the nodal membrane with a membrane assembled from external channel models.

## Caveats

State these clearly:

- Channelpedia channel models are usually local membrane models, often derived from expression systems and voltage clamp data.
- Conductance densities must be tuned before claiming physiological accuracy.
- Temperature, Q10, reversal potentials, and host-cell differences must be handled carefully.
- A Nav1.7-like node is pedagogically useful but should not be presented as the canonical channel composition of large myelinated nodes.
- Nav1.6 is the more defensible nodal candidate.

---

# 3. Demo 3 — Channelpedia-derived sensory C-fiber

## Suggested file name

```text
examples/advanced/custom_axons/03_channelpedia_sensory_c_fiber.py
```

## Scientific pitch

This demo builds an unmyelinated C-fiber from reusable sensory-neuron-like membrane channel models. Unlike the previous examples, the whole axon is created from a custom membrane composition.

The user story is:

> We assemble a nociceptor-like C-fiber membrane using Nav1.7, Nav1.8, Nav1.9, Kv, Kv7, HCN, leak, and optionally Na/K pump components. AxonScope then turns that local membrane into a spatially extended unmyelinated axon and evaluates whether it supports slow, stable propagation and activity-dependent slowing.

This is the cleanest way to demonstrate:

```text
Channelpedia membrane components → complete axon model
```

without drifting into sensory receptor transduction.

## Biological motivation

The C-fiber should be framed as:

```text
nociceptor-like unmyelinated C-fiber
```

not as a definitive human C-fiber model.

A credible sensory C-fiber membrane can include:

```text
Nav1.7  → threshold and subthreshold amplification
Nav1.8  → robust TTX-resistant spike upstroke and conduction in C-fiber axons
Nav1.9  → persistent / subthreshold sodium current, excitability tuning
Kdr     → repolarization
A-type K or Kv4-like current → spike shape and excitability tuning
Kv7/M-current → slow excitability brake
HCN/Ih  → repetitive firing and recovery effects
Leak    → resting membrane potential
Na/K pump and ion concentration dynamics → activity-dependent slowing
```

This choice is inspired by nociceptor and C-fiber modeling work, especially models that include multiple sodium channels, potassium channels, HCN current, leak currents, Na/K pump, and dynamic ion concentrations.

## Minimal first implementation

Start with a simpler version:

```text
Nav1.7 + Nav1.8 + Nav1.9 + Kdr + Kv7 + leak
```

Then add optional advanced mechanisms:

```text
HCN
Na/K pump
dynamic intracellular Na+
dynamic extracellular/intracellular K+
slow inactivation gates
```

## Model construction

Import channels:

```python
Nav17 = axs.ChannelpediaHHChannel.from_name("Nav1.7")
Nav18 = axs.ChannelpediaHHChannel.from_name("Nav1.8")
Nav19 = axs.ChannelpediaHHChannel.from_name("Nav1.9")
Kv72  = axs.ChannelpediaHHChannel.from_name("Kv7.2")
Kv21  = axs.ChannelpediaHHChannel.from_name("Kv2.1")
HCN1  = axs.ChannelpediaHHChannel.from_name("HCN1")
```

Then compose the membrane:

```python
sensory_c_membrane = axs.CompositeMembrane(
    capacitance=1.0 * axs.uF / axs.cm**2,
    channels=[
        Nav17.with_density(80 * axs.S / axs.m**2),
        Nav18.with_density(300 * axs.S / axs.m**2),
        Nav19.with_density(15 * axs.S / axs.m**2),
        Kv21.with_density(40 * axs.S / axs.m**2),
        Kv72.with_density(5 * axs.S / axs.m**2),
        axs.Leak(g=0.05 * axs.S / axs.m**2, e=-60 * axs.mV),
    ],
)
```

Then build an unmyelinated axon:

```python
axon = axs.UnmyelinatedAxon(
    length=30 * axs.mm,
    diameter=0.8 * axs.um,
    n_compartments=600,
    membrane=sensory_c_membrane,
    axial_resistivity=0.7 * axs.ohm * axs.m,
)
```

The values above are demonstration starting points, not validated final parameters. The conductance densities should be fitted to target behaviors.

## Target phenotype

The model should aim for:

```text
- unmyelinated propagation
- conduction velocity in the approximate C-fiber range
- broad action potentials compared with large myelinated axons
- higher threshold than large myelinated fibers
- possible activity-dependent slowing during repetitive stimulation
```

Possible target metrics:

```text
conduction velocity: ~0.5–2 m/s
fiber diameter: ~0.5–1.5 µm
stable propagation over centimeters
spike width: broader than MRG nodal spike
```

The exact targets should be treated as calibration targets, not hard-coded truths.

## Three C-fiber membrane variants

The demo becomes more interesting if it compares three membranes assembled from the same components.

### 3.1 Baseline C-nociceptor-like membrane

```python
baseline = make_c_fiber_membrane(
    nav17_scale=1.0,
    nav18_scale=1.0,
    nav19_scale=1.0,
    kv7_scale=1.0,
    hcn_scale=1.0,
)
```

Expected behavior:

```text
stable slow conduction
moderate threshold
limited activity-dependent slowing unless slow states are included
```

### 3.2 Sensitized C-fiber membrane

```python
sensitized = make_c_fiber_membrane(
    nav17_vshift=-5 * axs.mV,
    nav18_scale=1.2,
    nav19_scale=1.5,
    kv7_scale=0.7,
    hcn_scale=1.2,
)
```

Expected behavior:

```text
lower threshold
shorter latency
easier firing
possible afterdepolarization
greater excitability
```

This is a useful condition because it demonstrates how the same channel building blocks can be perturbed to create a disease-like or sensitized phenotype.

### 3.3 Nav1.8-reduced / conduction-fragile membrane

```python
nav18_reduced = make_c_fiber_membrane(
    nav17_scale=1.0,
    nav18_scale=0.4,
    nav19_scale=1.0,
    kv7_scale=1.0,
    hcn_scale=1.0,
)
```

Expected behavior:

```text
higher threshold
weaker spike upstroke
slower or less reliable propagation
possible conduction failure
```

This condition is biologically meaningful because Nav1.8 has been implicated as a major contributor to TTX-resistant conduction in distal somatosensory C-fiber axons.

## Stimulation protocols

### Protocol A — single-pulse threshold

Use a current clamp near one end:

```python
stim = axs.IntracellularStimulus.pulse(
    compartment=20,
    start=5 * axs.ms,
    duration=1 * axs.ms,
    amplitude=I,
)
```

Find the threshold for a propagated spike at the distal recording site:

```python
threshold = axs.find_threshold(
    axon=axon,
    stimulus_template=stim_template,
    target_compartment=-50,
    criterion=lambda r: r.vm[-50].max() > 0 * axs.mV,
)
```

### Protocol B — pulse-width threshold curve

Run thresholds for different pulse widths:

```python
pulse_widths = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0] * axs.ms
```

Plot:

```text
threshold current versus pulse width
```

This is useful because C-fibers are often more sensitive to longer pulses than large myelinated fibers.

### Protocol C — activity-dependent slowing

Deliver a repetitive stimulation train:

```python
stim = axs.Stimulus.train(
    frequency=2 * axs.Hz,
    n_pulses=20,
    pulse_width=1 * axs.ms,
    amplitude=1.2 * threshold,
)
```

Measure latency for each pulse:

```python
latency[n] = spike_time_recording_site[n] - stimulus_time[n]
```

Then:

```python
ads_percent[n] = 100 * (latency[n] - latency[0]) / latency[0]
```

This keeps the example purely axonal:

```text
repetitive electrical activation → propagated spikes → latency drift
```

No receptor transduction is required.

## Metrics

Measure:

```text
- single-pulse threshold
- pulse-width threshold curve
- conduction velocity
- spike width
- propagation success/failure
- latency per pulse
- activity-dependent slowing
- recovery after rest
```

For recovery, add a pause after a train:

```python
stim = train_2hz_for_20_pulses + rest_30s + test_pulse
```

Then measure whether the latency returns toward baseline.

## Figures

Recommended figures:

1. **Membrane composition diagram**
   Shows Nav1.7, Nav1.8, Nav1.9, Kv, Kv7, leak, optional HCN/pump.

2. **Channel curves**
   Plot `m_inf`, `h_inf`, and time constants for Nav1.7/1.8/1.9.

3. **Vm(x, t) heatmap**
   Shows slow unmyelinated propagation.

4. **Threshold versus pulse width**
   Compares baseline, sensitized, and Nav1.8-reduced variants.

5. **Latency versus pulse number**
   Shows activity-dependent slowing.

6. **Conduction velocity versus membrane variant**
   Summarizes the phenotype.

## Expected qualitative result

```text
baseline C-fiber        → stable slow propagation
sensitized C-fiber      → lower threshold, shorter latency, higher excitability
Nav1.8-reduced C-fiber  → higher threshold, fragile propagation
```

If slow sodium inactivation, Na/K pump, and sodium accumulation are included:

```text
repetitive stimulation → progressive latency increase
```

## What this demo proves

This demo proves that AxonScope can assemble a new axon type from membrane components:

```text
Channelpedia-like channels
→ CompositeMembrane
→ UnmyelinatedAxon
→ propagation, threshold, latency, ADS
```

The key message is:

> External membrane models are usually local. AxonScope makes them spatial, propagating, and stimulation-ready.

## Caveats

State these explicitly:

- The model is “nociceptor-like”, not a fully validated human nociceptor.
- Channelpedia channels may be derived from expression systems and patch-clamp protocols, not directly from intact axons.
- Conductance densities must be fitted to target axonal phenotypes.
- Temperature correction and Q10 assumptions are important.
- A credible ADS model likely requires slow inactivation, ion concentration dynamics, and Na/K pump activity.
- Without slow states, the model can still demonstrate membrane-to-axon assembly, but it may not reproduce microneurography-like ADS.

## Optional advanced version

Add dynamic sodium concentration:

```python
sensory_c_membrane = axs.CompositeMembrane(
    capacitance=1.0 * axs.uF / axs.cm**2,
    channels=[...],
    ion_dynamics=[
        axs.IntracellularSodium(),
        axs.NaKPump(),
    ],
)
```

This allows the demo to connect more directly to activity-dependent slowing.

---

# Recommended documentation structure

The three examples can be presented as a mini tutorial series:

```text
Custom Axon Models from Reusable Membranes
├── 01_mrg_focal_demyelination.py
├── 02_mrg_channelpedia_nav_nodes.py
└── 03_channelpedia_sensory_c_fiber.py
```

## Tutorial introduction

Suggested text:

> AxonScope separates axonal structure, membrane biophysics, stimulation, and solver execution. This makes it possible to assemble new axon models from reusable membrane components. The following examples demonstrate three increasingly flexible uses of this design: local pathological modification of an MRG fiber, replacement of the nodal membrane in an MRG fiber using Channelpedia-derived Nav channels, and construction of an unmyelinated sensory C-fiber from Channelpedia-derived membrane components.

## Common output table

Each example should generate a compact result table:

| Example | Condition | Threshold | CV | Success | Notes |
|---|---:|---:|---:|---:|---|
| MRG demyelination | severity 0.0 | ... | ... | yes | healthy |
| MRG demyelination | severity 0.9 | ... | ... | no/partial | conduction block |
| MRG Nav node | Nav1.6-like | ... | ... | yes | tuned density |
| MRG Nav node | GOF | ... | ... | yes | lower threshold |
| C-fiber | baseline | ... | ... | yes | slow conduction |
| C-fiber | Nav1.8 reduced | ... | ... | partial | fragile propagation |

## Common plots

Use consistent plotting functions across examples:

```python
axs.plot_vm_heatmap(result)
axs.plot_traces(result, locations=[...])
axs.plot_activation_times(result)
axs.plot_threshold_summary(results)
```

This reinforces that the same solver and analysis tools apply to very different axon models.

---

# Implementation notes for AxonScope

## 1. Membrane adapter layer

Add or emphasize:

```python
axs.ChannelpediaHHChannel
axs.CustomHHChannel
axs.CompositeMembrane
axs.MembraneAdapter
```

A channel should provide:

```python
channel.current(v, state)
channel.rhs(v, state)
channel.initial_state(v)
channel.reversal
channel.density
```

A composite membrane should sum currents:

```python
I_total = sum(channel.current(v, state[channel.name]) for channel in channels)
```

and concatenate state vectors.

## 2. Unit handling

The examples should be unit-safe:

```python
80 * axs.S / axs.m**2
1.0 * axs.uF / axs.cm**2
0.8 * axs.um
30 * axs.mm
```

This matters because membrane models imported from different sources may use inconsistent units.

## 3. Temperature handling

The examples should include a temperature argument:

```python
temperature = 37 * axs.degC
```

and channels should expose:

```python
channel.with_q10(q10=3.0, reference_temperature=22 * axs.degC)
```

or a similar correction method.

## 4. Calibration hooks

Provide calibration targets:

```python
targets = axs.AxonPhenotypeTargets(
    conduction_velocity=(0.5, 2.0) * axs.m / axs.s,
    threshold_range=(...),
    spike_width_range=(...),
)
```

Then later:

```python
fitted = axs.fit_conductance_densities(
    axon_template=axon_template,
    membrane_template=membrane_template,
    targets=targets,
)
```

Even if the fitting is not implemented yet, the examples can be designed so that this becomes a natural extension.

---

# Scientific references

## Core axon and MRG references

1. McIntyre, C. C., Richardson, A. G., & Grill, W. M. (2002). *Modeling the excitability of mammalian nerve fibers: influence of afterpotentials on the recovery cycle.* Journal of Neurophysiology, 87(2), 995–1006. https://doi.org/10.1152/jn.00353.2001
   - Basis for the MRG myelinated mammalian axon model.
   - Introduces a double-cable structure with explicit nodes, paranodes, internodes, and finite-impedance myelin.

2. ModelDB entry 3810. *Spinal Motor Neuron (McIntyre et al. 2002).* https://modeldb.science/showmodel?model=3810
   - Public implementation/reference entry for the MRG-type PNS myelinated axon model.

## Demyelination references

3. Reutskiy, S., Rossoni, E., & Tirozzi, B. (2003). *Conduction in bundles of demyelinated nerve fibers: computer simulation.* Biological Cybernetics, 89, 439–448. https://doi.org/10.1007/s00422-003-0447-x
   - Computational study of local myelin damage and conduction effects.

4. Naud, R., Longtin, A., & Maler, L. (2019). *Linking demyelination to compound action potential dispersion with a spike-diffuse-spike approach.* Journal of Mathematical Neuroscience, 9, 3. https://doi.org/10.1186/s13408-019-0071-6
   - Models how demyelination affects propagation delay, jitter, and transmission probability.

5. Coggan, J. S., et al. (2015). *Physiological dynamics in demyelinating diseases: unraveling complex relationships through computer modeling.* International Journal of Molecular Sciences, 16(9), 21215–21236. https://doi.org/10.3390/ijms160921215
   - Review of computational modeling in demyelinating diseases.

## Channelpedia and channel model references

6. Ranjan, R., et al. (2011). *Channelpedia: an integrative and interactive database for ion channels.* Frontiers in Neuroinformatics, 5, 36. https://doi.org/10.3389/fninf.2011.00036
   - Describes Channelpedia as a database for ion channel information and models.

7. Channelpedia. https://channelpedia.epfl.ch/
   - Web-based ion-channel knowledge base with annotated channels, electrophysiological data, and Hodgkin-Huxley models.

8. Channelpedia Nav1.6 page. https://channelpedia.epfl.ch/ionchannels/125
   - Candidate source for a Nav1.6-like nodal channel.

9. Channelpedia Nav1.7 page. https://channelpedia.epfl.ch/ionchannels/126
   - Candidate source for a sensory-neuron excitability channel.

10. Channelpedia Nav1.8 page. https://channelpedia.epfl.ch/ionchannels/127
   - Candidate source for a C-fiber / nociceptor sodium channel.

11. Channelpedia Nav1.9 page. https://channelpedia.epfl.ch/ionchannels/128
   - Candidate source for a persistent/subthreshold sodium channel.

12. Channelpedia Kv7.2 page. https://channelpedia.epfl.ch/ionchannels/24
   - Candidate source for a Kv7/M-current-like excitability brake.

## Nodal sodium channel references

13. Caldwell, J. H., Schaller, K. L., Lasher, R. S., Peles, E., & Levinson, S. R. (2000). *Sodium channel Nav1.6 is localized at nodes of Ranvier, dendrites, and synapses.* Proceedings of the National Academy of Sciences, 97(10), 5616–5620. https://doi.org/10.1073/pnas.090034797
   - Supports Nav1.6 as a biologically defensible nodal sodium-channel candidate.

## Sensory C-fiber and nociceptor references

14. Tigerholm, J., Petersson, M. E., Obreja, O., Lampert, A., Carr, R., Schmelz, M., & Fransen, E. (2014). *Modeling activity-dependent changes of axonal spike conduction in primary afferent C-nociceptors.* Journal of Neurophysiology, 111(9), 1721–1735. https://doi.org/10.1152/jn.00777.2012
   - Biophysical C-nociceptor axon model reproducing activity-dependent conduction changes.

15. Serra, J., Campero, M., Ochoa, J., & Bostock, H. (1999). *Activity-dependent slowing of conduction differentiates functional subtypes of C fibres innervating human skin.* Journal of Physiology, 515(3), 799–811. https://doi.org/10.1111/j.1469-7793.1999.799ab.x
   - Classic microneurography result: activity-dependent slowing differentiates human C-fiber classes.

16. Obreja, O., Ringkamp, M., Turnquist, B., Hirth, M., Forsch, E., Rukwied, R., Petersen, M., & Schmelz, M. (2010). *Patterns of activity-dependent conduction velocity changes differentiate classes of unmyelinated mechano-insensitive afferents including cold nociceptors.* Pain, 148(1), 59–69. https://doi.org/10.1016/j.pain.2009.10.013
   - Supports ADS as a functional marker of C-fiber subtype.

17. Ackerley, R., et al. (2018). *Microneurography as a tool to study the function of individual C-fiber afferents in humans: responses from nociceptors, thermoreceptors, and mechanoreceptors.* Journal of Neurophysiology, 120(6), 2834–2846. https://doi.org/10.1152/jn.00109.2018
   - Review of microneurography in human C-fiber afferents.

18. Hameed, S. (2019). *Nav1.7 and Nav1.8: role in the pathophysiology of pain.* Molecular Pain, 15. https://doi.org/10.1177/1744806919858801
   - Review supporting the relevance of Nav1.7 and Nav1.8 in sensory-neuron excitability and pain.

19. Klein, A. H., Vyshnevska, A., Hartke, T. V., De Col, R., Mankowski, J. L., Turnquist, B., Bosmans, F., Reeh, P. W., & Ringkamp, M. (2017). *Sodium channel Nav1.8 underlies TTX-resistant axonal action potential conduction in somatosensory C-fibers of distal cutaneous nerves.* Journal of Neuroscience, 37(20), 5204–5214. https://doi.org/10.1523/JNEUROSCI.3799-16.2017
   - Strong support for including Nav1.8 in a C-fiber axonal conduction model.

20. Maxion, A., et al. (2023). *A modelling study to dissect the potential role of voltage-gated ion channels in activity-dependent conduction velocity changes as identified in small fiber neuropathy patients.* Frontiers in Computational Neuroscience, 17, 1265958. https://doi.org/10.3389/fncom.2023.1265958
   - Shows how C-fiber computational models can be used to study ion-channel contributions to activity-dependent conduction changes.

---

# Final positioning statement

Suggested documentation blurb:

> These examples demonstrate that AxonScope treats membrane biophysics as a reusable component. A user can start from an existing membrane model, such as a Channelpedia Hodgkin-Huxley channel, combine it with other currents, and place the resulting membrane into a spatial axon template. The same solver can then evaluate propagation, threshold, conduction velocity, refractory behavior, pathological failure, and activity-dependent changes across very different axon models.
