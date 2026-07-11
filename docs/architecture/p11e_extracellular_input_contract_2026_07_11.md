# P11E Extracellular Input Contract - 2026-07-11

This note formalizes the non-solver extracellular input contract that
single-cable and double-cable execution paths should share before more
low-level solver work.

## Purpose

Recent P100 traces show that the remaining single-cable observer-only GPU cost
is mostly input representation and orchestration, not the tridiagonal solve
itself. In the reference `threshold_curves` run with `Naxons=4096`, `Nx=89`,
fp32, observer-only, and different diameters, single-cable spends about
`26.7 ms/simulation` in `inputs.extracellular`, while the comparable
double-cable route spends about `2.2 ms/simulation`.

The target is one explicit preparation/lowering contract shared by both cable
formulations. Solvers should receive typed numeric payloads and capability
decisions; they should not inspect public stimulation objects, benchmark
workload shapes, or membrane model names.

## Scope

This is an internal runtime contract. It does not add public dense/factorized
options and does not change the public `ExtracellularFootprint` ->
`ExtracellularDrive` -> `ExtracellularStimulation` workflow.

The contract applies after public objects have been prepared into a cohort and
before cable kernels are called:

```text
prepared rows
    -> dispatch grouping by runtime shape and waveform compatibility
    -> extracellular lowering mode selection
    -> typed numeric runtime payload
    -> single-cable or double-cable solver kernel
```

## Dispatch Compatibility

Rows may share one dispatch group when these properties are compatible:

- cable formulation: single-cable or double-cable;
- runtime, device, precision, and solver policy family;
- compiled membrane structure and padded runtime `Nx`;
- `Nt`, `dt`, sampling convention, and drive count/order;
- stimulus waveform compatibility signature.

Rows may vary inside the group:

- diameter and other parameter-batch values;
- sampled static footprints;
- row amplitude scales;
- row-specific solver state.

Amplitude magnitude is not a grouping key when the row can be represented as a
scale of a compatible waveform. Amplitude is dynamic numeric payload.

## Lowering Modes

The common lowering contract should represent extracellular input with one of
these modes, ordered from cheapest/safest to broadest. `S` is the number of
stimulation drives in the group. The common point-source case is just `S=1`;
multi-contact or multi-source stimulation should keep the same contract with
`S>1` instead of introducing a separate solver path.

| Mode | Numeric payload | Intended use |
| --- | --- | --- |
| `zero` | no extracellular payload | no extracellular stimulation |
| `shared_current` | static footprints `(B, S, Nx)` plus current waveforms `(S, Nt)` | recruitment curves and true shared waveform runs |
| `scaled_shared_waveform` | static footprints `(B, S, Nx)`, base waveforms `(S, Nt)`, row scales `(B, S)` | single-cable and double-cable threshold curves with the same temporal shape but different row amplitudes |
| `current_table` | static footprints `(B, S, Nx)`, unique current table `(U, Nt)`, row current indices `(B, S)`, optional row scales `(B, S)` | arbitrary temporal waveforms that cannot be reduced to shared/scaled waveforms |
| `dense` | explicit `Vstim[B, Nt, Nx]` plus double-cable previous sample when needed | fallback for unsupported or non-separable cases |

Dense is a fallback, not the default internal route. Observer-only compact
paths should reject unsupported shapes rather than silently materializing a
large dense tensor unless an explicit measured fallback policy exists.

The old rank-K view is therefore a shape, not a semantic mode:

```text
Vext[B, t, x] = sum_S footprint[B, S, x] * current_or_scale[B, S, t]
```

When each drive has a shared waveform, this is `shared_current[S]`. When each
drive shares a waveform shape but each row has its own amplitude, this is
`scaled_shared_waveform[S]`. Only genuinely arbitrary row/drive currents need
`current_table`, with the worst case `U = B * S`.

## Shared Waveform Contract

A shared waveform signature describes temporal shape, not amplitude value.

It should include:

- time grid shape, dtype, sampling convention, and digest;
- stimulus mode and units;
- normalized waveform shape digest;
- drive order and drive count;
- a zero-waveform marker for the all-zero special case.

The corresponding scale is a numeric array. A threshold curve should be able to
keep one group while amplitudes diverge by updating row scales, not by mutating
thousands of public `Stimulus` objects or rebuilding row-local waveforms in
Python.

If a waveform cannot be normalized safely, for example because it has an
offset, incompatible units, non-finite samples, or unsupported multi-drive
semantics, lowering should choose `current_table` or `dense` explicitly and
record the reason.

Object identity caches are allowed as hot-path accelerators, but correctness
must not rely on identity. Equivalent immutable waveform content should lower
to the same compatibility class.

## Capability Declaration

Each runtime/cable implementation should expose an internal typed capability
record for extracellular lowering. The exact Python name can change, but the
contract should be equivalent to the initial code anchor in
`src/axonscope/runtime/input_contract.py`:

```python
ExtracellularLoweringCapabilities(
    supports_zero=True,
    supports_shared_current=True,
    supports_scaled_shared_waveform=...,
    supports_current_table=...,
    supports_dense_fallback=...,
    requires_initial_previous=...,
)
```

The lowering decision should be capability-driven. Hard-coded helpers such as
`supports_compact_double_cable_factorized(...)` should become capability
checks against the selected cable/runtime path.

## Single-Cable Obligations

Single-cable execution should consume the common lowering modes without asking
for row-local public stimuli to keep threshold rows batchable.

The JAX single-cable path should support:

- `zero`;
- `shared_current`;
- `scaled_shared_waveform`;
- `current_table`;
- current rank-K where already equivalent and measured;
- dense fallback for explicit dense/reference paths.

It may keep an internal pre-lowered forcing footprint cache, such as
`single_cable_forcing_footprint_mV_per_A`, but that cache is an implementation
detail of the single-cable JAX runtime and not part of dispatch semantics.

## Double-Cable Obligations

Double-cable execution should use the same lowering decision surface, while
remaining stricter about the modes it can consume compactly.

The current JAX double-cable compact path is validated for shared-current
rank-1 observer-only inputs with a scalar initial previous sample. That fast
path should remain unchanged while the common contract is introduced.

`scaled_shared_waveform` is part of the target double-cable contract, because
double-cable threshold curves have the same "one waveform shape, many row
amplitudes" structure as single-cable threshold curves. During the transition,
the JAX double-cable capability can report no compact support and fall back
explicitly until equivalence and benchmarks validate the compact route.

Broader compact double-cable modes, such as compact
`scaled_shared_waveform` or `current_table`, should be added only
after:

- dense-vs-compact equivalence tests;
- observer-only and recorded/probe output tests where relevant;
- fresh CPU/GPU benchmarks showing no regression on the existing shared-current
  double-cable route.

Until then, double-cable should report an explicit dense fallback reason for
unsupported compact modes.

The multi-drive case should not introduce a separate rank-K solver family.
`Nstim=1` and `Nstim>1` should exercise the same lowering modes, with optional
internal squeezing or precomposition only after the common contract has chosen a
mode.

## Recording And Observer Contract

Extracellular input lowering should not fork by output mode except where a
capability explicitly says a compact route is only valid for one output sink.

`observer_only`, `probe_vm`, and `full_vm` should share the same semantic input
contract. Recording policy can change retention and pruning, but it should not
require a different public stimulation representation or a solver-specific
benchmark workaround.

## Required Metadata

Inspection and benchmark traces should record:

- selected lowering mode;
- dense fallback reason, when dense is used;
- drive count and unique waveform count;
- whether grouping used shared waveform plus row scale;
- current payload shape and footprint payload shape;
- host/device transfer sizes where available;
- capability decision for single-cable or double-cable.

No speed or memory claim should be made without this metadata in the artifact.

## Acceptance Criteria

The implementation is acceptable when:

- dispatcher grouping can batch same-shape waveforms with different amplitudes
  without row-local stimulus mutation tricks;
- recruitment keeps the existing `shared_current` route and reports zero Python
  row updates;
- threshold curves lower to `scaled_shared_waveform[S]` or a clearly justified
  fallback, not repeated host scans over row-local waveforms;
- double-cable shared-current observer-only benchmarks remain on the compact
  fast path with no regression;
- unsupported multi-drive or non-separable inputs choose an explicit fallback
  with inspection metadata;
- the P100 single-cable reference case reduces `inputs.extracellular` from the
  current roughly `26.7 ms/simulation` toward the double-cable few-millisecond
  range before any low-level single-cable solver optimization is considered.

## Implementation Slices

1. Add internal lowering-mode and capability types near the runtime input
   lowering boundary.
2. Split waveform shape signatures from amplitude scale in dispatch planning.
3. Build a preparation-level `Nstim`-aware shared-waveform/scaled-waveform
   representation.
4. Teach JAX single-cable lowering to consume that representation without
   row-local public stimulus scans.
5. Keep JAX double-cable shared-current behavior unchanged, then add compact
   scaled/indexed modes only if equivalence and benchmarks justify them.
6. Update inspection, benchmark metadata, and focused tests before running the
   next Kaggle gate.
