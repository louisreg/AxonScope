# Results, Recording, Analysis

This layer has three separate responsibilities:

- `Recording` describes what the solver should store.
- `SimResult` stores what a simulation actually returned.
- `axonscope.analysis` turns returned arrays into scientific metrics with
  statuses and population denominators.
- `axonscope.analysis` also provides lower-level rasterization and velocity
  helpers used by plotting and validation workflows.
- `axonscope.results.visualization` provides plotting helpers for returned
  arrays.

Keeping these responsibilities separate matters because a result may not contain
the full `Vm[Nt, Nx]` matrix. Pool runs can record only the center compartment
or a small set of probes, and observer-only runs can return compact observations
without materializing the whole voltage movie.

## Recording

`Recording` is a public storage policy. It is passed to `simulate` or
`simulate_pool`, not attached to the axon model.

```python
recording = axs.Recording(
    signals=[axs.signals.Vm, axs.signals.GATES, axs.signals.CURRENTS],
)

result = axs.simulate(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=recording,
)
```

Current support:

- single-axon runs always return `Vm`;
- single-axon runs can include observable groups such as `gates`, `currents`,
  and `conductances` alongside `Vm`;
- pool runs currently support `Vm` recording with `full`, `center`, `probes`,
  or explicit compartment `indices` spatial modes;
- `Recording.none()` is supported when solver-side observers are supplied;
- single-axon spatial filters, position-based recording, temporal subsampling,
  and pool observable groups are explicit future work.

Current solver handling:

- `axs.simulate(...)` validates the public `Recording`, forwards
  `record_observables=True` to the scalar solver when observable groups are
  requested, then filters `SimResult.recordings` to the requested groups.
- `axs.simulate_pool(...)` translates public pool Vm recording policies to
  solver-level `BatchRecording` through `Recording.to_batch_options()`. Scalar
  fallback rows are filtered after the solve so public `record_indices` match
  the requested center/probe/index columns.
- Low-level solvers and batch kernels still receive numerical flags/options;
  they do not own the user-facing `Recording` contract.

Convenience constructors:

```python
axs.Recording.voltage()
axs.Recording.full()
axs.Recording(signals=[axs.signals.Vm, axs.signals.GATES])
axs.Recording.center(axs.signals.Vm)
axs.Recording.probes(axs.signals.Vm, count=8)
axs.Recording.indices([0, 5, 10], axs.signals.Vm)
```

Signals are descriptors, not a closed enum. Built-in descriptors live under
`axs.signals`, and custom descriptors can be built with `axs.Signal` and
`axs.SignalId` for future workflows that produce new result channels.

`Recording.only(axs.signals.GATES)` is a valid policy object, but current
public solvers still require Vm storage. Include `axs.signals.Vm` when
requesting observable groups.

`positions` must carry length units and is stored internally as `positions_um`.
`sample_dt` must carry time units and is stored internally as `sample_dt_ms`.
Pint quantities are accepted and normalized at construction time:

```python
recording = axs.Recording(
    signals=axs.signals.Vm,
    positions=[0.25 * axs.mm],
    sample_dt=10 * axs.us,
)
```

## SimResult And Pool Results

`SimResult` is the scalar single-axon result and remains intentionally small:

```python
result.t               # time samples in ms, shape (Nt,)
result.recordings["Vm"]  # voltage samples, shape (Nt, Nrecorded)
result.Vm              # convenience alias for recordings["Vm"]
result.recording       # Recording policy used by the public wrapper, if any
result.record_indices  # original axon indices represented by Vm columns, if filtered
result.recordings      # Vm plus optional gates/currents/etc.
result.observations    # compact observer outputs for observer-only runs
result.diagnostics     # optional metadata such as pool index/method
```

Results expose small unit-aware accessors and plot helpers for common notebook
workflows:

```python
result.time_values(unit=axs.ms)
result.position_values(unit=axs.um)
result.voltage_values(unit=axs.mV)
result.trace_values(position=500 * axs.um)
result.peak_voltage_values(unit=axs.mV)

result.plot_trace(position=500 * axs.um)
result.plot_map()
```

For a full recording, `result.Vm.shape[1] == result.axon.n_compartments`. For a
filtered recording, `result.record_indices` maps each `Vm` column back to the
original axon position. Analysis functions must use that mapping instead of
assuming that columns are contiguous compartments.

Pool runs return `AxonSimulationResult`, a cohort-backed container. Indexing or
iterating over it gives one `AxonResultView` per simulated row:

```python
results = axs.simulate_pool(
    pool,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
)

for result in results:
    assert result.Vm.shape[1] == 1
    print(result.record_indices)

dense_vm = results.signal(axs.signals.Vm)
first = results.axon(0)
vm_manifest = results.recording_manifest.signal(axs.signals.Vm)
standalone = first.to_sim_result()
```

For homogeneous recordings, `results.signal(axs.signals.Vm)` returns a dense
array indexed as `(axon, time, recorded_position)`. Heterogeneous pools remain
accessible through per-axon views and through `results.cohorts`.
`results.recording_manifest` records which signals were requested, which
signals are actually available, and the dense shape/dtype for each cohort.

The lower-level `run_pool` path returns private dispatch results. Those
containers keep `index`, `group_id`, and `method` before the public wrapper
converts the pool into `AxonSimulationResult`.

## Analysis

Structured post-hoc analysis lives under `axs.analysis`.

```python
report = result.report(
    axs.analysis.Activation(
        threshold=-20 * axs.mV,
        target=axs.positions.DISTAL,
    ),
    axs.analysis.PeakVoltage(target=axs.positions.CENTER),
)

activation = report["activation"]
activation.values
activation.statuses
activation.population.n_valid
```

Each public analysis definition declares requirements such as required signals,
positions, supported formulations, and algorithm version. Analysis results keep
values and statuses side by side, so missing inputs or undetermined metrics are
not hidden as anonymous NaNs.

Low-level helpers live under the same `axs.analysis` namespace.

```python
spike_t_ms, spike_x_um = axs.analysis.rasterize(
    result,
    threshold_mV=-10.0,
    min_distance_ms=1.0,
)

velocity_m_s = axs.analysis.conduction_velocity(result)
peaks_mV = axs.analysis.peak_voltage(result)
positions_um = axs.analysis.recorded_positions_um(result)
```

Threshold and timing arguments also accept Pint quantities:

```python
spike_t_ms, spike_x_um = axs.analysis.rasterize(
    result,
    threshold_mV=-10 * axs.mV,
    min_distance_ms=1 * axs.ms,
)
```

`recorded_positions_um(result)` is the shared guardrail. It returns the physical
positions represented by `Vm` columns. If a result is spatially filtered but does
not carry `record_indices`, it raises a `ValueError` rather than guessing.
Rasterization also validates that the minimum spike distance is non-negative.

The low-level post-hoc activation criterion also lives under `axs.analysis`:

```python
event = axs.analysis.ActivationCriterion(
    threshold=-20 * axs.mV,
    blanking=0.2 * axs.ms,
    target=axs.positions.DISTAL,
).evaluate(result)

event.activated
event.first_time_ms
event.first_position_um
```

These criteria are CPU/post-hoc companions to the lightweight online Vm
observers and the current solver-side observer path.

## Visualization

Shared plotting helpers live under `axs.results.visualization` and consume the
same analysis helpers. `SimResult` also exposes direct plot methods for the
common single-result voltage trace/map workflows.

```python
ax = axs.results.visualization.plot_raster(
    result,
    threshold_mV=-10.0,
    min_distance_ms=1.0,
)
```

Future plotting helpers should follow the same rule: they can consume
`SimResult`, `AxonResultView`, `AxonSimulationResult`, axon models, or
dispatcher outputs, and should reuse the same position/recording guardrails.

## Online Vm Observers

Activation and peak-voltage analyses can create lightweight online observers
that consume membrane-voltage chunks and finalize to the same `AnalysisResult`
shape as post-hoc definitions:

```python
activation = axs.analysis.Activation(
    threshold=-20 * axs.mV,
    target=axs.positions.DISTAL,
)

observer = activation.online_observer(
    positions=result.position_values(unit=axs.um) * axs.um,
    original_indices=result.record_indices,
)
observer.update(
    result.time_values(unit=axs.ms) * axs.ms,
    result.voltage_values(unit=axs.mV) * axs.mV,
)

online_activation = observer.finalize()
posthoc_activation = result.analyze(activation)
```

Solver-side observer execution is implemented for the current
`axs.analysis.Activation(...)` and `axs.analysis.PeakVoltage(...)` definitions
when they are passed as simulation observers with `Recording.none()`. Observer
state is updated at every solver `dt` inside the scalar or compatible batch
kernel, and the result carries compact `observations` rather than retained Vm
traces.

```python
result = axs.simulate(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[
        axs.analysis.Activation(threshold=-20.0 * axs.mV),
        axs.analysis.PeakVoltage(),
    ],
)
```

For pool runs, homogeneous single-cable and homogeneous double-cable groups can
use the compact observer-only batch path. Incompatible or padded groups may
still fall back to scalar execution or post-hoc observer evaluation depending on
the requested recording and group shape. Solver-side observers currently support
membrane-voltage based `Activation` and `PeakVoltage`; richer latency, block,
spike-count, and non-Vm signal observers remain future work.
