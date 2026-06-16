# Recorders, Observers, Thresholds, and Recruitment

This document is a design proposal plus implementation-status note. CPU/post-hoc
analyses, threshold protocols, lightweight Vm observers, and the first
solver-side observer-only paths are implemented. Broader study APIs and
amplitude-batched GPU sweeps remain future work.

The goal is to make AxonScope useful for stimulation studies such as activation
thresholds and recruitment curves while keeping the GPU path sane. The central
rule is:

> Recorders store traces. Observers compute compact results during the solve.

This distinction matters because full voltage movies are expensive for large
pools and amplitude sweeps. A recruitment study may need thousands of solves,
but often only needs one boolean per fiber per amplitude.

## Implementation Status

Current status on 2026-06-16:

- Phase 1 is implemented for the current result surface:
  - `SimResult.recordings["Vm"]` is the conceptual home of membrane voltage.
  - `result.Vm` remains a convenience alias for existing examples and tests.
  - `SimResult.observations` stores compact observer outputs for observer-only
    scalar runs.
- Phase 2 is implemented in CPU/post-hoc mode:
  - `axs.analysis.ActivationCriterion` evaluates activation from recorded Vm
    traces.
  - `axs.analysis.ActivationEvent` stores the compact activation summary.
  - `axs.analysis.detect_activation(...)` is a convenience wrapper.
- Phase 6 is implemented for the current public analysis layer:
  - `axs.analysis.Activation(...)` and `axs.analysis.PeakVoltage(...)` expose
    structured analysis definitions with statuses and missing-input metadata.
  - `axs.analysis.ActivationObserver` and
    `axs.analysis.PeakVoltageObserver` consume streamed Vm chunks and are
    cross-validated against post-hoc results.
- Phase 7.5 is implemented for the first public observer-only workflow:
  - public `axs.analysis.Activation(...)` and
    `axs.analysis.PeakVoltage(...)` lower to compact backend observer state;
  - scalar kernels and compatible homogeneous batch kernels call observer
    updates at every `dt`;
  - `Recording.none()` plus observers avoids retaining full Vm traces and
    returns compact `result.observations`;
  - homogeneous double-cable batch observer-only runs are supported for current
    MRG-like hotpaths.
- Phase 4/5 protocol helpers are implemented in CPU/post-hoc mode:
  - `axs.protocols.find_activation_threshold(...)` runs a binary search over
    stimulus amplitude.
  - `axs.protocols.find_activation_threshold_curve(...)` evaluates thresholds
    over a sequence of pool items.
  - `axs.protocols.pool_sweep(...)` runs generic repeated pool observations.
  - `axs.protocols.recruitment_sweep(...)` evaluates activation over sampled
    amplitudes and returns a `RecruitmentCurve`.
- User-facing examples:
  - `examples/advanced/example_06_activation_criterion.py` demonstrates
    post-hoc activation criteria.
  - `examples/advanced/example_07_recruitment_curve.py` demonstrates an
    extracellular threshold and recruitment protocol workflow.
  - `examples/basic/example_07_threshold_vs_diameter.py` and
    `examples/basic/example_08_recruitment_curve_population.py` provide compact
    didactic workflows for threshold curves and mixed-population recruitment.

Not implemented yet:

- Amplitude-batched GPU sweeps.
- Solver-side observers beyond membrane-voltage `Activation` and
  `PeakVoltage`.
- Combining scalar solver-side observers with retained scalar Vm traces in one
  kernel call.
- Padded heterogeneous batch observer masks.
- A dedicated `thresholds_for_pool(...)` convenience wrapper.

## Target Use Cases

AxonScope should support, in increasing complexity:

1. Run one axon and inspect full `Vm(t, x)`.
2. Run one axon and record observables such as gates/currents/conductances.
3. Run a pool and retain only center/probe/explicit Vm columns.
4. Run one axon or a pool and return only compact observations:
   activation, peak voltage, first spike time, first spike position.
5. Find the activation threshold of one axon.
6. Find thresholds for a pool.
7. Build recruitment curves versus stimulus amplitude.
8. Eventually batch amplitude sweeps on GPU to reduce Python launch overhead.

The API should stay didactic for users, but the implementation should not force
CPU-GPU transfers of full traces when a compact observer result is sufficient.

## Vocabulary

### Recorder

A recorder describes arrays to retain from the simulation.

Examples:

- `Vm`
- `gates.m`
- `currents.I_na`
- `conductances.g_k`
- all variables in a group such as `gates`

Recorders produce trace-like outputs. These outputs usually have a time axis:

```text
Vm:                (Nt, Nrecorded)
gates.m:           (Nt, Nrecorded)
pool Vm:           one (Nt, Nrecorded) matrix per fiber/result
batch internal Vm: (B, Nt, Nrecorded)
```

Recorders are useful for plotting, debugging, model validation, and small
notebooks. They are not the preferred mechanism for large threshold/recruitment
studies.

### Observer

An observer describes a compact reduction computed during the solver loop.

Examples:

- activated or not
- first spike time
- first spike position/index
- peak Vm
- peak depolarization after a blanking period
- spike count

Observers produce compact outputs without necessarily materializing the full
recorded traces:

```text
activation.activated:          scalar bool
activation.first_time_ms:      scalar float
activation.first_index:        scalar int
pool activation.activated:     (B,) bool during batch execution
pool activation.first_time_ms: (B,) float during batch execution
```

Observers should have post-hoc equivalents where possible, so a user can debug
the same activation criterion on a full trace before using the GPU-efficient
observer path.

### Analysis Criterion

An analysis criterion is the NumPy/post-hoc version of an observer. It consumes a
`SimResult` that already has traces.

Example:

```python
criterion = axs.analysis.ActivationCriterion(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
)

event = criterion.evaluate(result)
```

The corresponding observer should use the same public options:

```python
observer = axs.analysis.ActivationObserver(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
)
```

### Protocol / Experiment

A protocol or experiment orchestrates repeated simulations.

Examples:

- `find_activation_threshold(...)`
- `thresholds_for_pool(...)`
- `recruitment_sweep(...)`

Protocols should not contain low-level numerical logic. They should build
simulations, call `simulate` or `simulate_pool`, and interpret recorder/observer
outputs.

## Proposed Result Model

`SimResult` should eventually store all retained traces in `recordings`.
`Vm` should be treated as a recorded variable, not as a special result category.

Proposed shape:

```python
result.recordings["Vm"]                    # voltage, shape (Nt, Nrecorded)
result.recordings["gates"]["m"]            # optional observable traces
result.recordings["currents"]["I_na"]
result.observations["activation"]          # compact observer output
result.observations["peak_voltage"]
result.t                                   # time vector, if a time-resolved recorder exists
result.record_indices                      # physical mapping for spatially filtered Vm
```

Short-term transition:

```python
result.Vm
```

can remain as a compatibility/convenience property that returns
`result.recordings["Vm"]`. The conceptual model should still be:

> Vm is a recording variable.

Long-term optional trace behavior:

```python
result.recordings is None
result.Vm raises a clear error or returns None
result.observations["activation"].activated
```

This enables:

```python
axs.simulate_pool(
    pool,
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[axs.analysis.ActivationObserver(...)]
)
```

## Recorder API

The current `Recording` object is the right public surface, but it should be
interpreted as a recorder policy.

Possible future shape:

```python
recording = axs.Recording(
    signals=[axs.signals.Vm, axs.signals.GATES, axs.signals.CURRENTS],
    spatial=axs.RecordingSpatial.indices([0, 50, 100]),
    temporal=axs.RecordingTemporal.every(10),
)
```

For now, the simpler API can remain:

```python
axs.Recording.voltage()
axs.Recording.full()
axs.Recording.only(axs.signals.Vm, axs.signals.GATES)
axs.Recording.center(axs.signals.Vm)
axs.Recording.probes(axs.signals.Vm, count=8)
axs.Recording.indices([0, 10, 20], axs.signals.Vm)
axs.Recording.none()
```

Important design points:

- `Recording` is public/user-facing.
- Solver-side batch retention remains a lower-level numerical policy.
- Pool observable groups can be future work; pool Vm retention is already useful.
- Position-based recording can be implemented later by resolving positions to
  compartment indices before the solver starts.
- Temporal subsampling should be implemented in the solver loop or batch kernel,
  not by storing everything and slicing on CPU.

## Observer API

Observers should be public objects under `axonscope.results` or a nearby module.
They describe what compact result should be computed, not how the solver computes
it.

Example:

```python
activation = axs.analysis.ActivationObserver(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
    require_propagation=True,
)

peak = axs.analysis.PeakVoltageObserver(
    target=axs.positions.ALL,
    blanking=0.0 * axs.ms,
)

results = axs.simulate_pool(
    pool,
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[activation, peak],
)
```

Result access:

```python
event = result.observations["activation"]
event.activated
event.first_time_ms
event.first_position_um
event.first_index

peak = result.observations["peak_voltage"]
peak.value_mV
peak.index
peak.time_ms
```

For batch solver internals, the same fields become vectorized arrays over the
batch axis.

## Observer Compilation Model

Public observer objects should compile to solver-side/JAX-compatible observer
implementations at runtime preparation.

Conceptual interface:

```python
class CompiledObserver:
    name: str

    def init_state(self, runtime) -> ObserverState:
        ...

    def update(self, state, *, Vm, t_ms, step_index, runtime) -> ObserverState:
        ...

    def finalize(self, state, runtime) -> Observation:
        ...
```

JAX constraints:

- `ObserverState` must be a JAX pytree with static structure.
- No Python callbacks inside `lax.scan`.
- No growing lists or dynamic dictionaries inside the time loop.
- Output shapes should be known before JIT compilation.
- Public units are converted before compilation; JAX receives plain numeric
  canonical units.

Single-cable and double-cable kernels should both expose the same observer
sample to observers:

```text
Vm_new_mV
t_ms
step_index
```

Additional optional samples can be added later:

```text
gates
currents
conductances
Ve
Vi
```

The first implementation should only require `Vm`. This keeps activation and
peak voltage independent of membrane model internals.

## GPU and Memory Rules

The GPU-efficient path should obey these rules:

1. Do not materialize full `Vm[B, Nt, Nx]` unless a recorder asks for it.
2. Do not transfer arrays to CPU at every time step.
3. Observers update inside the JAX time loop.
4. Transfer only finalized compact observation arrays at the end.
5. Use static shapes for observer state and output.
6. Vectorize observer states over batch rows.
7. Keep threshold/recruitment orchestration in Python initially, but avoid
   copying trace arrays back to CPU.

For example, activation in a batch should keep state like:

```text
activated:     bool[B]
first_time_ms: float[B]
first_index:   int[B]
peak_mV:       float[B]
```

not:

```text
Vm_full: float[B, Nt, Nx]
```

## Spatial Selection for Observers

Observers need to know which compartments to inspect.

Public options could be:

```python
target=axs.positions.ALL
target=axs.positions.CENTER
target=axs.positions.DISTAL
target=axs.positions.At([100.0 * axs.um, 500.0 * axs.um])
target=axs.positions.Indices([0, 10, 20])
```

Implementation should compile these to integer indices before the solver loop.

Suggested first implementation:

- `target=axs.positions.Indices(...)`
- `target=axs.positions.ALL`
- `target=axs.positions.CENTER`
- `target=axs.positions.DISTAL`

Later:

- a future node selector for myelinated axons;
- physical position arrays resolved through layout;
- named section/tag selectors.

For a filtered observer, the observer only evaluates the selected columns. This
keeps activation checks cheap:

```text
selected Vm shape for one row: (Nselected,)
selected Vm shape for batch:   (B, Nselected)
```

## Activation Semantics

Activation detection is subtle. A simple local threshold crossing can be
misleading for extracellular stimulation because the membrane under the
electrode can depolarize without producing a propagating action potential.

Therefore the activation API should distinguish simple and propagated criteria.

### Simple Activation

Triggered when selected `Vm` crosses a threshold after a blanking period:

```python
ActivationObserver(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.ALL,
)
```

State:

```text
activated
first_time_ms
first_index
```

This is useful for intracellular examples and simple debugging.

### Propagated Activation

Triggered when a spike is detected away from the stimulation site or in a target
zone:

```python
ActivationObserver(
    threshold=-20.0 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
    require_propagation=True,
)
```

For a first robust implementation, `require_propagation=True` can mean:

- inspect only distal/probe indices;
- detect threshold crossing after blanking;
- optionally require a positive local peak above threshold rather than only one
  sample above threshold.

Later refinements:

- require crossings at two spatially separated locations in time order;
- estimate velocity and reject implausible velocity ranges;
- ignore electrode-near compartments via an exclusion window.

## Activation Observer Pseudocode

Single row, conceptual NumPy/JAX logic:

```python
selected = Vm_new_mV[indices]
eligible = t_ms >= blanking_ms
crossing = eligible & any(selected >= threshold_mV)

newly_activated = crossing & ~state.activated
first_local_index = argmax(selected >= threshold_mV)

state.activated = state.activated | crossing
state.first_time_ms = where(newly_activated, t_ms, state.first_time_ms)
state.first_index = where(newly_activated, indices[first_local_index], state.first_index)
```

Batch shape:

```text
selected:          (B, Nselected)
crossing:          (B,)
newly_activated:   (B,)
first_local_index: (B,)
first_index:       (B,)
```

No host transfer is needed until `finalize`.

## Peak Voltage Observer

Peak voltage is the simplest observer and should probably be implemented before
activation to validate the observer infrastructure.

Public:

```python
axs.analysis.PeakVoltageObserver(
    target=axs.positions.ALL,
    blanking=0.0 * axs.ms,
)
```

State:

```text
peak_mV
peak_time_ms
peak_index
```

Update:

```python
selected = Vm_new_mV[indices]
local_peak = max(selected)
is_new_peak = eligible & (local_peak > state.peak_mV)
```

This observer is useful for recruitment summaries and sanity plots, even when
activation criteria are under discussion.

## Threshold Search

Threshold search should live above solvers, probably in a new module:

```text
src/axonscope/protocols/
  __init__.py
  activation.py
```

Public API:

```python
threshold = axs.protocols.find_activation_threshold(
    simulation_factory=lambda tested_current: make_simulation(
        electrode_current=tested_current,
    ),
    bounds=(1.0 * axs.uA, 500.0 * axs.uA),
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.2 * axs.ms,
        target=axs.positions.DISTAL,
    ),
    tolerance=1.0 * axs.uA,
    max_iterations=20,
)
```

Factory contract:

- the protocol only chooses the next tested value;
- the user-provided lambda receives that value with units;
- the lambda owns every modification between simulations;
- no protocol mutates an existing stimulus, electrode, context, or axon in
  place.

Returned object:

```python
threshold.amplitude
threshold.status
threshold.lower_bound
threshold.upper_bound
threshold.history
threshold.n_iterations
threshold.plot()
```

Algorithm, phase 1:

1. Evaluate lower bound.
2. Evaluate upper bound.
3. If lower already activates, threshold is below range.
4. If upper does not activate, threshold is above range.
5. Otherwise binary search until tolerance or max iterations.

Each evaluation should use:

```python
recording=axs.Recording.none()
observers=[activation_observer]
```

when observers are implemented. Before observers exist, the same protocol can
fall back to `Recording.probes(...)` or `Recording.full()` plus post-hoc
`ActivationCriterion`.

## Recruitment Curves

Recruitment can be built either by sweeping amplitudes or by computing per-fiber
thresholds.

### Sweep-Based Recruitment

Public:

```python
curve = axs.protocols.recruitment_sweep(
    pool_factory=lambda tested_current: make_pool(
        electrode_current=tested_current,
    ),
    amplitudes=np.linspace(0.0, 500.0, 21) * axs.uA,
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.2 * axs.ms,
        target=axs.positions.DISTAL,
    ),
)
```

As with threshold search, the pool factory is the mutation boundary. This keeps
the protocol generic enough for extracellular stimulation, intracellular
clamps, waveform families, axon position sweeps, or any later FEM-backed
context. The public protocol should never assume that "changing amplitude" is
implemented by editing one particular object.

Returned object:

```python
curve.amplitudes
curve.activated       # bool array, shape (Namplitudes, Nfibers)
curve.count           # recruited fiber count per amplitude
curve.fraction        # recruited fraction per amplitude
curve.threshold_like_uA # first activating sampled amplitude per fiber, or NaN
curve.plot()
```

Phase 1 implementation:

- Python loop over amplitudes;
- each amplitude calls `simulate_pool`;
- post-hoc `ActivationCriterion` evaluates the returned `SimResult`;
- return compact arrays.

This is simple and already useful.

### Threshold-Based Recruitment

Public:

```python
thresholds = axs.protocols.thresholds_for_pool(
    fiber_factory=lambda fiber_index, tested_current: make_simulation(
        fiber_index,
        electrode_current=tested_current,
    ),
    fiber_count=N,
    bounds=(1.0 * axs.uA, 500.0 * axs.uA),
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.2 * axs.ms,
        target=axs.positions.DISTAL,
    ),
)

curve = axs.protocols.RecruitmentCurve.from_thresholds(
    thresholds,
    amplitudes=np.linspace(0.0, 500.0, 101) * axs.uA,
)
```

This is more efficient than a dense sweep if the response is monotonic in
amplitude.

Important caveat:

> Recruitment is only threshold-like if activation is monotonic with stimulus
> amplitude under the chosen pulse family.

For unusual waveforms, electrode interactions, or pathological dynamics, a
sweep may be safer than assuming monotonic thresholds.

## Future GPU Optimization: Amplitude Batch Dimension

Phase 1 can run one amplitude per solver launch. That is acceptable for the API
and for correctness.

Later, recruitment sweeps can batch amplitudes:

```text
Vm shape internally:       (A, F, Nx) or (A * F, Nx)
observer activated shape:  (A, F)
```

This requires:

- stimulus scaling/broadcasting over amplitude rows;
- batch context builders that accept an amplitude axis;
- compatible axons grouped once, then evaluated across amplitudes;
- careful memory control so `Vm[A, F, Nt, Nx]` is never materialized unless
  explicitly requested.

This should be an optimization below the public protocol API. The user should
not need to rewrite threshold/recruitment code when amplitude batching lands.

## Stimulus Families

Threshold and recruitment protocols need a clean way to change stimulus
amplitude, but that change should stay inside the user factory.

Useful convenience methods can live on `Stimulus`, but protocols should not call
them directly:

```python
stimulus.scaled(factor)
```

For point-source extracellular stimulation:

```python
base_electrode = axs.PointSourceElectrode(...)

def make_simulation(electrode_current):
    stimulus = axs.Stimulus.pulse(
        start=0.2 * axs.ms,
        duration=0.3 * axs.ms,
        amplitude=electrode_current,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[base_electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )
    sim = axs.AxonInstance(axon)
    sim.add_extracellular_context(context=context)
    return sim
```

This keeps the stimulus attached to the electrode, consistent with the current
stimulation design. If a waveform does not have a simple amplitude setter, the
factory can rebuild it directly:

```python
threshold = axs.protocols.find_activation_threshold(
    lambda tested_current: make_simulation(
        stimulus=axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.3 * axs.ms,
            amplitude=tested_current,
        ),
    ),
    bounds=(1.0 * axs.uA, 500.0 * axs.uA),
    duration=2.0 * axs.ms,
    dt=0.01 * axs.ms,
    criterion=criterion,
)
```

## Proposed Package Layout

Near-term:

```text
src/axonscope/results/
  single.py          SimResult, recordings, observations
  analysis.py        post-hoc analysis functions
  activation.py      ActivationCriterion and activation result dataclasses
  observers.py       public observer specs and observation dataclasses

src/axonscope/solvers/
  observers.py       compiled observer protocol and JAX implementations

src/axonscope/protocols/
  __init__.py
  activation.py      threshold search and recruitment protocols
```

Current boundary:

Keep public observer and analysis specifications in `axonscope.analysis`.
Lower them into backend-specific compiled observer state during Phase 7.5.
This keeps user-facing concepts separate from numerical results while keeping
JAX-specific observer details out of the public analysis layer.

## Implementation Plan

### Phase 0: Documentation and API Agreement

- Done: validate this document as a strategy/proposal with current-status notes.
- Done: keep public policy named `Recording`.
- Done: keep compact observer outputs under `observations`.
- Done: keep `result.Vm` as a permanent convenience alias for retained Vm
  recordings, with an explicit error when Vm was not retained.

### Phase 1: Normalize `Vm` as a Recording Variable

- Store `Vm` in `SimResult.recordings["Vm"]`.
- Keep `result.Vm` as a property reading `recordings["Vm"]`.
- Update docs and examples to show `recordings["Vm"]` as the conceptual source.
- Keep existing helper methods:
  - `voltage_values`
  - `trace_values`
  - `peak_voltage_values`
  - `plot_trace`
  - `plot_map`

Validation:

- unit tests for direct `recordings["Vm"]`;
- old `.Vm` usage still works.

### Phase 2: Post-Hoc Activation Criterion

- Add `ActivationCriterion`.
- Add `ActivationEvent` dataclass.
- Implement NumPy evaluation on `SimResult`.
- Support threshold, blanking, indices, center/distal/all.

Validation:

- synthetic traces with known crossing times;
- filtered `record_indices`;
- no-activation cases;
- wrong-shape/error cases.

### Solver-Side Observer Infrastructure

- Done: add `observations` to `SimResult`.
- Done: add public observer specs. Current status: lightweight public
  `ActivationObserver` and `PeakVoltageObserver` specs exist for streamed Vm
  chunks.
- Done for `Activation` and `PeakVoltage`: add solver-side compiled observer
  interface as AxonScope Phase 7.5.
- Done for scalar observer-only runs: wire observers through scalar solver
  kernels.
- Done for compatible homogeneous batch observer-only runs: wire observers
  through batch kernels.
- Done: call observer updates at every solver `dt` inside the kernel/scan loop.
- Done for current compatible paths: support observer-only runs without
  retaining full Vm traces.
- Future: add more observer kinds, padded heterogeneous batch masks, and
  amplitude-batched study execution.

Validation:

- observer peak equals post-hoc peak on full recording;
- observer activation equals `ActivationCriterion` on test traces;
- batch observer shape is one compact output per pool row.

### Phase 4: Threshold Protocols

- Add `find_activation_threshold`.
- Use observer path if available.
- Fall back to post-hoc criterion if full/probe recording is requested.
- Return a structured threshold result with history.

Validation:

- monotonic toy stimulus response;
- lower-bound-active and upper-bound-inactive edge cases;
- units for amplitude/tolerance.

### Phase 5: Recruitment Protocols

- Add `recruitment_sweep`.
- Add `thresholds_for_pool`.
- Add `RecruitmentCurve`.
- Add plotting helpers for count/fraction versus amplitude.

Validation:

- small HH/Rattay pool example;
- deterministic synthetic pool;
- recruitment count monotonic for monotonic pulse family;
- no full Vm transfer in observer-only mode.

### Phase 6: Amplitude-Batched GPU Recruitment

- Add internal amplitude axis to runtime-batch builders.
- Batch compatible axons and amplitudes together.
- Keep public `recruitment_sweep` unchanged.
- Use observers to return only `(Namplitudes, Nfibers)` compact outputs.

This phase is an optimization, not a prerequisite for the public API.

## Example Roadmap

Current examples:

```text
examples/advanced/example_05_recording_options.py
examples/advanced/example_06_activation_criterion.py
examples/advanced/example_07_recruitment_curve.py
examples/advanced/example_18_solver_side_observers.py
```

`example_05_recording_options.py` shows the current recording policy surface:
`Vm`, observable groups, and pool Vm retention modes.

`example_06_activation_criterion.py` shows the post-hoc activation criterion
that shares semantics with `axs.analysis.Activation`.

`example_07_recruitment_curve.py` shows the high-level protocol API for one
extracellular binary activation threshold search and one sampled recruitment
curve from a point-source electrode.

`example_18_solver_side_observers.py` shows homogeneous single-cable and
double-cable observer-only runs with `Recording.none()` and compares compact
observer outputs against post-hoc analysis.

Future examples should focus on missing public orchestration rather than new
recording primitives:

- callable extracellular stimulation families for threshold/recruitment
  studies;
- optional full-trace reruns at threshold for inspection;
- solver-option presets once Phase 7.6.3 establishes benchmark-backed
  defaults.

## Open Questions

Remaining questions to decide before API freeze:

1. Should `recordings` be flat (`"Vm"`, `"gates.m"`) or mixed nested
   (`"Vm"`, `"gates": {"m": ...}`)?
2. What should the default activation criterion be for extracellular
   stimulation: simple threshold, distal threshold, or propagated crossing?
3. Should threshold protocols assume monotonicity by default and expose a
   `check_monotonic=True` diagnostic option?

Settled recommendations:

- Keep public policy named `Recording`.
- Treat `Vm` as `recordings["Vm"]`.
- Keep `result.Vm` as a convenience alias for notebooks and compatibility.
- Put public criteria/observers in `axs.analysis`.
- Put threshold/recruitment orchestration in `axs.protocols`.

Current recommendation for the remaining questions:

- Use nested recordings for grouped variables:
  - `recordings["Vm"]`
  - `recordings["gates"]["m"]`
  - `recordings["currents"]["I_na"]`
- Make propagated/distal activation the recommended extracellular criterion,
  while still allowing simple threshold crossing for didactic examples.
