# Results, Recording, Analysis

This layer has three separate responsibilities:

- `Recording` describes what the solver should store.
- `AxonSimulationResult` stores what public execution returned.
- `AxonResultView` exposes one simulated axon row from that result.
- `axonscope.analysis` turns returned arrays into scientific metrics with
  statuses and population denominators.
- `axonscope.analysis` also provides lower-level rasterization and velocity
  helpers used by plotting and validation workflows.
- `axonscope.results.views` provides raw result value/plot helpers, and
  `axonscope.analysis.views` provides analysis-derived plots such as spike
  rasters.

Keeping these responsibilities separate matters because a result may not contain
the full `Vm[Nt, Nx]` matrix. Pool runs can record only the center compartment
or a small set of probes, and observer-only runs can return compact observations
without materializing the whole voltage movie.

## Recording

`Recording` is a public storage policy. It is passed to `AxonSimulation`, not
attached to the axon model.

```python
recording = axs.Recording(
    signals=[axs.signals.Vm, axs.signals.GATES, axs.signals.CURRENTS],
)

run = axs.AxonSimulation(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=recording,
).run()
result = run.single
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

- `AxonSimulation.run()` validates the public `Recording`, forwards
  `record_observables=True` to the scalar route when observable groups are
  requested, then filters the internal scalar recording payload to the
  requested groups before wrapping it in the canonical public result.
- Population `AxonSimulation.run()` translates public pool Vm recording
  policies to a backend-neutral `RecordingPlan`, then the JAX backend lowers
  that plan to solver-level `BatchRecording`. Scalar fallback rows are filtered
  after the solve so public `record_indices` match the requested
  center/probe/index columns.
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

## Canonical Simulation Results

All public execution returns `AxonSimulationResult`, including one-axon runs:

```python
run = axs.AxonSimulation(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
).run()
result = run.single      # or run[0]

result.t               # time samples in ms, shape (Nt,)
result.recordings["Vm"]  # voltage samples, shape (Nt, Nrecorded)
result.Vm              # alias for recordings["Vm"] on one-axon views
result.signal(axs.signals.Vm)
result.recording       # Recording policy used by the public wrapper, if any
result.record_indices  # original axon indices represented by Vm columns, if filtered
result.recorded_axis   # recorded intrinsic positions + original layout indices
result.recordings      # Vm plus optional gates/currents/etc.
result.observations    # compact observer outputs for observer-only runs
result.diagnostics     # metadata such as pool index/method
result.final_state     # final backend state, or None when not retained
```

Results expose small unit-aware accessors and plot helpers for common notebook
workflows:

```python
result.time_values(unit=axs.ms)
result.position_values(unit=axs.um)
result.recorded_axis.position_values(unit=axs.um)
result.recorded_axis.index_values()
result.voltage_values(unit=axs.mV)
result.trace_values(position=500 * axs.um)
result.peak_voltage_values(unit=axs.mV)

result.plot_trace(position=500 * axs.um)
result.plot_map()
```

For a full recording, `result.Vm.shape[1] == result.axon.n_compartments`. For a
filtered recording, `result.record_indices` maps each `Vm` column back to the
original axon position. `result.recorded_axis` is the canonical interpreted
view of that metadata: it contains intrinsic axon positions, never anatomical
or world placement. Analysis functions must use that mapping instead of
assuming that columns are contiguous compartments.

Pool runs return the same `AxonSimulationResult` container. Indexing or
iterating over it gives one `AxonResultView` per simulated row:

```python
results = axs.AxonSimulation(
    pool,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.center(axs.signals.Vm),
).run()

for result in results:
    assert result.Vm.shape[1] == 1
    print(result.recorded_axis.original_indices)

dense_vm = results.signal(axs.signals.Vm)
first = results.axon(0)
center_trace = first.signal(axs.signals.Vm)
vm_manifest = results.recording_manifest.signal(axs.signals.Vm)
row_recordings = results.recordings
row_axes = results.recorded_axes
row_final_states = results.final_states
```

For homogeneous recordings, `results.signal(axs.signals.Vm)` returns a dense
array indexed as `(axon, time, recorded_position)`. Heterogeneous pools remain
accessible through per-axon views; storage partitioning is an implementation
detail, not a second result workflow.
`results.recording_manifest` records which signals were requested, which
signals are actually available, and advanced storage shape/dtype metadata.

Public result surface audit:

| Object | Role |
| --- | --- |
| `AxonSimulationResult` | canonical execution result for one row or many rows; supports indexing, iteration, `signal(...)`, `analyze(...)`, `report(...)`, diagnostics, observations, recordings, recorded axes, and final-state aggregation. |
| `AxonResultView` | one simulated row; exposes `Vm`, `t`, `signal(...)`, `recorded_axis`, recordings, observations, diagnostics, final state, plots, and analysis/report helpers. |
| `RecordedSignal` and `RecordingManifest` | structured record of requested and available signals. |
| `RecordedAxis` | canonical interpretation of retained Vm columns as intrinsic axon positions plus original layout indices. |
| `VmRasterResult` | compact observer-only threshold raster stored under `observations["vm_raster"]`. |
| `AnalysisReport` and protocol summaries | separate scientific interpretations of results; they do not mutate or merge into raw numerical result objects. |

View/rendering helpers are intentionally separate from data containers:
`axs.results.views` owns one-axon value extraction and Vm plots,
`axs.analysis.views` owns analysis/report formatting, dataframe export, and
simple scalar report plots, and `axs.protocols.views` owns threshold,
recruitment, and generic sweep summaries. Methods such as
`result.plot_trace()`, `result.plot_traces()`, `results.plot_traces()`,
`result.plot_recorded_axis()`, `raster.plot()`, `report.to_dataframe()`,
`analysis_result.plot()`, `curve.plot()`, and `curve.plot_groups()` remain
ergonomic facades over those view modules.

The view modules share a small plotting helper layer for Matplotlib axis
creation, unit-aware labels, titles, grids, and legends. Plot semantics stay
domain-owned: raw simulation plots are named by signal or concept, summary
objects expose `plot()` only when there is a clear default visualization, and
example-specific scientific plots remain in examples.

Summary-like objects use one table contract:

- `rows()` returns tuple-of-dict rows without requiring pandas.
- `to_dataframe()` wraps the same rows when pandas is available.
- `format()` and `print()` produce compact human-readable text.
- `plot()` is available only when the object has a clear default plot.

This applies to analysis results/reports, protocol summaries, VmRaster
summaries, and simulation estimates. Raw simulation arrays remain raw arrays;
they are inspected through direct value helpers such as `voltage_values(...)`,
`trace_values(...)`, or `signal(...)` rather than forced into one generic
dataframe.

The lower-level `run_pool` path returns private dispatch records. Those
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

Status vocabularies are intentionally scoped:

- `AnalysisStatus` belongs to `AnalysisResult` rows. It answers whether a
  scientific metric is valid, missing required inputs, not applicable,
  numerically failed, or undetermined.
- Protocol statuses belong to protocol summaries. For example,
  `axs.protocols.ThresholdStatus` uses `threshold`, `below_range`, and
  `above_range` to describe a bisection outcome relative to the supplied
  bounds.

Do not compare protocol statuses to `AnalysisStatus`. A protocol may use
analysis definitions or keep analysis events in its history, but the protocol
result status describes the outer search or sweep, not the validity of a
single analysis metric.

When a protocol produces a per-row scientific metric, it can expose an explicit
analysis view without losing the richer protocol object:

```python
curve = axs.protocols.find_threshold(...)
threshold_metric = curve.to_analysis_result()

recruitment = axs.protocols.recruitment_sweep(...)
first_activation = recruitment.to_analysis_result()
```

The protocol object keeps search history, sampled values, and protocol status.
The derived `AnalysisResult` keeps values, `AnalysisStatus`, unit, population
denominators, and row labels for report/table workflows.

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

Shared plotting helpers live under the retained view modules. One-axon views
expose direct plot methods for common voltage trace/map workflows, while
analysis-derived plots live under `axs.analysis.views`.

```python
ax = axs.analysis.views.plot_spike_raster(
    result,
    threshold_mV=-10.0,
    min_distance_ms=1.0,
)
```

## Online Vm Observers

Activation analyses can create lightweight online observers that consume
membrane-voltage chunks and finalize to the same `AnalysisResult` shape as
post-hoc definitions. `PeakVoltage` remains a post-hoc analysis on recorded Vm:

```python
activation = axs.analysis.Activation(
    threshold=-20 * axs.mV,
    target=axs.positions.DISTAL,
)

observer = activation.online_observer(
    positions=result.recorded_axis.position_values(unit=axs.um) * axs.um,
    original_indices=result.recorded_axis.original_indices,
)
observer.update(
    result.time_values(unit=axs.ms) * axs.ms,
    result.voltage_values(unit=axs.mV) * axs.mV,
)

online_activation = observer.finalize()
posthoc_activation = result.analyze(activation)
```

Solver-side observer-only execution now uses one strict VmRaster primitive.
Threshold-style definitions such as `axs.analysis.Activation(...)` lower to
fixed membrane-voltage probes, the solver thresholds those probes at every
`dt`, and the result carries compact `observations["vm_raster"]` rather than
retained Vm traces. Activation, latency, velocity, threshold, and recruitment
summaries are post-processing of that raster.

The packed result container is `axs.results.VmRasterResult` and the canonical
observation key is `axs.results.VM_RASTER_OBSERVATION_KEY`. Solver/backend code
owns the packed-bit update loop, but CPU unpacking and result-side helpers live
with public results.

```python
run = axs.AxonSimulation(
    sim,
    duration=5.0 * axs.ms,
    dt=0.01 * axs.ms,
    recording=axs.Recording.none(),
    observers=[
        axs.analysis.Activation(threshold=-20.0 * axs.mV),
    ],
).run()
result = run.single
```

For pool runs, compatible single-cable and double-cable groups can use the
compact observer-only batch path. Row-specific probe tables and masks must be
lowered before the solver so padded rows do not force full Vm retention.
Population-level `result.observations["vm_raster"]` may be padded to a shared
probe width; an individual `result[i].observations["vm_raster"]` view exposes
that row's own probe width.
`PeakVoltage` and other rich analyses remain post-hoc on recorded Vm until a
dedicated solver-side implementation is designed, benchmarked, and kept off the
hot path.
