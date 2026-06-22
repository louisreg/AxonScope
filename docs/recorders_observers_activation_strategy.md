# Recorders, VmRaster Observers, Thresholds, And Recruitment

This document records the current recording/observer strategy. The older broad
solver-side observer design is superseded.

## Current Contract

Recorders retain traces. Analyses interpret traces or compact observations.

```text
Recording.voltage(...)
    retains Vm traces for plotting, validation, and post-hoc analyses

Recording.none() + threshold-style observers
    retains only packed observations["vm_raster"]
```

The active solver-side observer path is intentionally narrow:

- public threshold-style analysis definitions such as
  `axs.analysis.Activation(...)`, `axs.analysis.Latency(...)`, and
  `axs.analysis.ConductionBlock(...)` may lower to a VmRaster plan;
- backend kernels threshold fixed membrane-voltage probes at every solver `dt`;
- backend kernels return compact `observations["vm_raster"]`;
- `axs.results.VmRasterResult` and `axs.results.unpack_vm_raster_words(...)`
  own result-side unpacking;
- activation, latency, velocity, threshold, and recruitment summaries are
  post-processing of the packed raster.

`PeakVoltage` remains post-hoc on recorded Vm. Do not document or reintroduce a
generic `Activation`/`PeakVoltage` compiled observer interface unless a future
design has benchmark evidence and a separate hot-path review.

## Recording

Recording policies describe outputs to retain:

- full Vm for teaching and validation;
- center/probe/explicit Vm positions for lighter trace inspection;
- no Vm traces when compact VmRaster observations are enough.

The conceptual home of membrane voltage is `result.recordings["Vm"]` on a
one-axon result view. `result.Vm` remains a convenience alias.

Unsupported recording requests must fail explicitly. Do not create meaningless
arrays, fill unavailable rows silently, or retain full Vm only to recover
observer outputs later.

## Analyses

Post-hoc analyses consume recorded signals and return separate analysis objects:

```python
activation = axs.analysis.Activation(threshold=-20.0 * axs.mV)
event = result.analyze(activation)

peak = result.analyze(axs.analysis.PeakVoltage(target=axs.positions.CENTER))
```

Lightweight host-side streaming observers may exist for cross-validation against
recorded Vm chunks. They are not the solver-side observer architecture.

## VmRaster

The backend VmRaster plan is a JAX implementation detail. It is allowed to live
under `axonscope.backends.jax`, while public protocols and analyses consume only
the result-side `VmRasterResult`.

The packed state shape is:

```text
words[B, R, P, W] uint32
B = batch rows
R = threshold/probe definition count
P = static probe slots
W = ceil(Nt / 32)
```

For padded or heterogeneous groups, probe tables are row-aware and padded slots
are masked before the kernel runs.

## Protocols

Threshold and recruitment protocols should be written against simulation
results or public analysis definitions:

- `axs.protocols.find_activation_threshold(...)`;
- `axs.protocols.find_activation_threshold_curve(...)`;
- `axs.protocols.pool_sweep(...)`;
- `axs.protocols.recruitment_sweep(...)`.

When a protocol only needs threshold-style membrane-voltage events, it may use
`Recording.none()` plus a compatible analysis definition so the backend can
return `observations["vm_raster"]`.

## Future Work

Future additions should build on this narrow contract:

- richer public decoders from `VmRasterResult`;
- study-level threshold/recruitment orchestration;
- amplitude batching only after memory and route inspection are reliable.

Do not add a broad solver-side observer hierarchy as a fallback path. New
solver-side observations need a concrete user need, equivalence tests against
post-hoc analysis, and benchmark evidence that the hot path stays healthy.
