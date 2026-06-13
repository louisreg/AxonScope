# Results, Recording, Analysis

This layer has three separate responsibilities:

- `Recording` describes what the solver should store.
- `SimResult` stores what a simulation actually returned.
- `axonscope.results.analysis` and `axonscope.results.visualization` interpret
  those returned arrays.

Keeping these responsibilities separate matters because a result may not contain
the full `Vm[Nt, Nx]` matrix. Pool runs can record only the center compartment
or a small set of probes, and future solvers should be able to return observer
outputs without materializing the whole voltage movie.

## Recording

`Recording` is a public storage policy. It is passed to `simulate` or
`simulate_pool`, not attached to the axon model.

```python
recording = axs.Recording(
    variables=["Vm", "gates", "currents"],
    spatial_mode="full",
)

result = axs.simulate(
    sim,
    duration_ms=5.0 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=recording,
)
```

Current support:

- single-axon runs always return `Vm`;
- single-axon runs can filter observable groups such as `gates` and `currents`;
- pool runs currently support `Vm` recording with `full`, `center`, `probes`,
  or explicit compartment `indices` spatial modes;
- single-axon spatial filters, position-based recording, temporal subsampling,
  and pool observable groups are explicit future work.

Convenience constructors:

```python
axs.Recording.voltage()
axs.Recording.full()
axs.Recording.center(["Vm"])
axs.Recording.probes(["Vm"], count=8)
axs.Recording.indices([0, 5, 10], ["Vm"])
```

Plain numbers use AxonScope's canonical public units. `positions_um` is stored in
micrometers and `sample_dt_ms` is stored in milliseconds. Pint quantities are
accepted and normalized at construction time:

```python
recording = axs.Recording(
    variables="Vm",
    positions_um=[0.25 * axs.mm],
    sample_dt_ms=10 * axs.us,
)
```

## SimResult

`SimResult` is intentionally small:

```python
result.t               # time samples in ms, shape (Nt,)
result.recordings["Vm"]  # voltage samples, shape (Nt, Nrecorded)
result.Vm              # convenience alias for recordings["Vm"]
result.recording       # Recording policy used by the public wrapper, if any
result.record_indices  # original axon indices represented by Vm columns, if filtered
result.recordings      # Vm plus optional gates/currents/etc.
result.observations    # compact observer outputs, once observer support lands
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

Pool runs return plain result lists:

```python
results = axs.simulate_pool(
    pool,
    duration_ms=5.0 * axs.ms,
    dt_ms=0.01 * axs.ms,
    recording=axs.Recording.center(["Vm"]),
)

for result in results:
    assert result.Vm.shape[1] == 1
    print(result.record_indices)
```

The lower-level `run_pool` path returns private dispatch results. Those
containers keep `index`, `group_id`, and `method` before the public wrapper
converts each axon to `SimResult`.

## Analysis

Post-hoc analysis lives under `axs.results.analysis`.

```python
spike_t_ms, spike_x_um = axs.results.analysis.rasterize(
    result,
    threshold_mV=-10.0,
    min_distance_ms=1.0,
)

velocity_m_s = axs.results.analysis.conduction_velocity(result)
peaks_mV = axs.results.analysis.peak_voltage(result)
positions_um = axs.results.analysis.recorded_positions_um(result)
```

Threshold and timing arguments also accept Pint quantities:

```python
spike_t_ms, spike_x_um = axs.results.analysis.rasterize(
    result,
    threshold_mV=-10 * axs.mV,
    min_distance_ms=1 * axs.ms,
)
```

`recorded_positions_um(result)` is the shared guardrail. It returns the physical
positions represented by `Vm` columns. If a result is spatially filtered but does
not carry `record_indices`, it raises a `ValueError` rather than guessing.
Rasterization also validates that the minimum spike distance is non-negative.

Post-hoc activation criteria live under `axs.results`:

```python
event = axs.results.ActivationCriterion(
    threshold=-20 * axs.mV,
    blanking=0.2 * axs.ms,
    positions="distal",
).evaluate(result)

event.activated
event.first_time_ms
event.first_position_um
```

These criteria are CPU/post-hoc companions to the future solver-side observers.

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
`SimResult`, lists of `SimResult`, axon models, or dispatcher, and should reuse
the same position/recording guardrails.

## Future Observers

The long-term solver-side mechanism should be observer-style:

```python
result = axs.simulate(
    sim,
    duration_ms=5.0 * axs.ms,
    dt_ms=0.01 * axs.ms,
    observers=[
        axs.results.analysis.RasterObserver(threshold_mV=-10.0),
        axs.results.analysis.PeakVoltageObserver(),
    ],
)
```

Observers should share names and semantics with post-hoc analysis functions, but
they will run inside the solver loop and store compact derived outputs. That is
the path for rasterization, peak detection, thresholds, and summaries that do
not need full voltage storage.
