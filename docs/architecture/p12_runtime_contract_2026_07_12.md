# P12 Runtime Contract And Cleanup Plan - 2026-07-12

P12 starts from the P11 result: JAX performance is good enough for the current
double-cable and single-cable benchmark surface. The next cleanup must preserve
that evidence while making the runtime boundary clean enough for a future real
NumPy/SciPy runtime.

## Goal

Homogenize everything outside the numerical solver as much as possible between
single-cable and double-cable execution:

- dispatch and batch grouping;
- prepared cohort shape and padding rules;
- recording and VmRaster observer lowering;
- intracellular input lowering;
- extracellular input semantics;
- benchmark and inspection metadata;
- result assembly.

The solver itself remains runtime-specific and cable-specific. Any cleanup that
touches hot solver kernels needs benchmark evidence before and after.

## Runtime Contract

A concrete runtime must accept a prepared cable batch with:

- one cable formulation: `single-cable` or `double-cable`;
- one target `Nx` after padding;
- one time grid and dtype policy;
- row-specific parameters allowed through prepared arrays, not Python loops in
  the kernel hot path;
- a typed per-cable solver request from `ExecutionPolicy.solvers`;
- a recording/output plan derived from public `Recording` and observer
  definitions;
- intracellular input in one of the semantic modes:
  `zero`, `dense`, or `sparse_current_clamp`;
- extracellular input in one of the semantic modes:
  `zero`, `shared_current`, `scaled_shared_waveform`, `current_table`, or
  `dense`;
- an optional `initial_previous` extracellular sample for runtimes/cable paths
  that need the `t=-dt/2` value.

The semantic contract lives in `src/axonscope/runtime/input_contract.py`.
Concrete runtimes may choose different array containers, kernels, compilers, or
solver algorithms, but they should consume the same semantic lowering modes.

## Runtime-Neutral Versus Runtime-Specific

Runtime-neutral code should own:

- public execution policy types;
- cable/input semantic contracts;
- dispatch grouping and signatures;
- prepared cohort structure;
- public recording and observer definitions;
- benchmark event vocabulary;
- result model and result-side analyses.

JAX-specific code should own:

- JAX arrays and device placement;
- JIT/cache behavior;
- JAX membrane program lowering;
- JAX input materialization;
- JAX kernels and solver engines;
- JAX profiler/device-memory support.

Future NumPy/SciPy code should add its own runtime namespace instead of using a
JAX-backed compatibility path.

## Cleanup Order

1. Keep the current JAX performance paths unchanged while adding guardrails for
   the shared runtime contract.
2. Move semantic-only concepts from `runtime/jax` to `runtime` when they do not
   mention JAX arrays, JIT, or solver internals.
3. Refactor `runtime/jax/group_runner.py` in small steps:
   common prepare/cohort/recording/observer/lowering metadata helpers first,
   kernel-specific calls last.
4. Homogenize extracellular lowering names and metadata around
   `shared_current` and `scaled_shared_waveform` for both cable families.
5. Delete dead or duplicate paths only after a structural test proves they are
   not imported by active examples, tests, or benchmark entry points.
6. Re-run hotpath benchmarks before claiming no performance loss.

## Non-Goals

- Do not choose a new GPU default policy in this cleanup.
- Do not add model-specific runtime branches.
- Do not implement the NumPy/SciPy runtime until the JAX contract is stable.
- Do not replace exact double-cable semantics with simplified surrogate paths.
