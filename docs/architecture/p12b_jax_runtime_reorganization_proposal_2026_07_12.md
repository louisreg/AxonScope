# P12B JAX Runtime Reorganization Proposal

Date: 2026-07-12

This note audits `src/axonscope/runtime/jax/` after the scalar route and direct
solver facade removal. The goal is not a behavior change yet. The goal is to
make the JAX runtime easier to reason about and to expose a clean contract for a
future NumPy/SciPy runtime.

## Current Findings

`runtime/jax` is still mostly active code, not obvious dead code. `vulture`
does not report high-confidence unused runtime files. The batch-kernel knot has
been split mechanically: `kernels/batch.py` was removed and its contents now
live under cable-specific and shared-support kernel modules. Remaining issues
are smaller responsibility-mixing points:

- `kernels/single_cable.py` and `kernels/double_cable.py` still own large active
  hot paths, but cable ownership is now explicit and shared chunking,
  factorized-input, input-coercion, and result helpers are separated.
- The old direct/reference scan primitives and direct
  `SingleCableKernel`/`DoubleCableKernel` classes were production-dead and
  have been deleted rather than moved.
- The old `kernels/common.py` bucket has been split: `cable_geometry.py` owns
  geometry/diffusion helpers, `double_cable_linear.py` owns double-cable
  linear-system layout and assembly, and `block_tridiagonal.py` owns the active
  CPU Thomas primitive.
- JAX input construction is now split between `inputs/extracellular.py` and
  `inputs/intracellular.py`. The remaining cleanup there is mostly footprint
  helper extraction and payload contract separation, not one monolithic builder
  split.
- `types.py` now owns the JAX prepared-runtime dataclasses.
- `membranes/compile.py` now owns public membrane-to-JAX compilation and
  membrane backend construction.
- Base JAX runtime construction now lives in `preparation/base.py`: membrane,
  cable, stimulation, extracellular, solver-runtime preparation, cache keys,
  and direct sampling helpers.
- `inputs/lowering.py` is already close to the desired shape: it selects
  semantic lowering modes, but it still imports JAX builders directly.
- Runtime-neutral pieces already exist and should be strengthened:
  `runtime/input_contract.py`, `runtime/input_planning.py`,
  `runtime/host_preparation.py`, `runtime/solver_axon.py`,
  `runtime/output_contract.py`, and `runtime/recording.py`.

## File-By-File Responsibility

| File | Current responsibility | Proposed action |
| --- | --- | --- |
| `policy/__init__.py` | JAX solver request constructors and CPU/GPU namespaces | Keep JAX-specific and thin; do not export runtime internals here. |
| `policy/execution.py` | Device, precision, cache validation, JAX execution context | Keep JAX-specific. |
| `policy/engine*.py` | Resolve typed public solver policy to JAX internal engine | Keep flattened under `policy/`; a nested solver-engine package is unnecessary. |
| `group_runner.py` | Main JAX batch group orchestration | Keep as the runtime entry point, but rename target could be `runner.py`; it should import high-level prepared inputs, not low-level builders directly. |
| `types.py` | JAX prepared-runtime dataclasses | Keep as the stable internal JAX type contract. |
| `membranes/compile.py` | Public membrane-to-JAX compilation and backend construction | Keep under the membrane compiler subpackage. |
| `preparation/base.py` | Base single-row JAX runtime preparation, cache keys, and sampling helpers | Keep under `preparation/`; later split cache keys only if it reduces coupling. |
| `preparation/runtime.py` | Batch `SolverRuntime` construction and batch runtime cache use | Keep JAX-specific. |
| `preparation/stacking.py` | JAX row stacking for cable, membrane, extracellular, and group `Cm` arrays | Keep JAX-specific; runtime-neutral NumPy helpers stay in `runtime/host_preparation.py`. |
| `inputs/lowering.py` | Select dense/sparse/factorized input formats | Move semantic decisions toward `runtime/input_contract.py` and `runtime/input_planning.py`; keep only JAX adapter wiring here. |
| `inputs/extracellular.py` | Dense/factorized extracellular JAX input builders and footprint caches | Keep; later extract footprint cache/sampling helpers only if it improves clarity without hiding hot-path behavior. |
| `inputs/intracellular.py` | Dense/sparse intracellular JAX input builders | Keep as the JAX current-density materialization module. |
| `inputs/payloads.py` | JAX materializers for sparse/factorized payload contracts | Done: dataclasses moved to runtime-neutral `runtime/input_payloads.py`; JAX materializers remain in `runtime/jax/inputs/payloads.py`. |
| `kernels/single_cable.py` | Active single-cable batch solver kernel and chunk routes | Keep; future cleanup should only extract repeated non-solver orchestration if benchmarks stay stable. |
| `kernels/double_cable.py` | Double-cable route wrapper, chunk routes, and CPU/GPU dispatch | Keep as the cable-level wrapper; avoid solver-specific code here. |
| `kernels/double_cable_cpu.py` | CPU double-cable scan body and Thomas solve binding | Keep CPU-specific. |
| `kernels/double_cable_gpu.py` | GPU double-cable scan body and tiled-Thomas/Triton solve binding | Keep GPU-specific. |
| `kernels/chunking.py` | Shared chunking and VmRaster chunk-state helpers | Keep; shared by both cable families. |
| `kernels/factorized.py` | Shared factorized-current and single-cable forcing helpers | Keep; shared by single/double cable without runtime-specific policy decisions. |
| `kernels/inputs.py` | Shared batch-kernel input coercion and recording helpers | Keep; later split only if NumPy/SciPy runtime can reuse runtime-neutral contracts. |
| `cable_geometry.py` | Array alias, cable coefficients, compartment geometry, diffusion operator | Keep outside `kernels/`; move runtime-neutral geometry only if NumPy/SciPy can share it directly. |
| `kernels/double_cable_linear.py` | Double-cable layout conversion, static-term preparation, system assembly, Triton solve bridge | Keep as the shared double-cable linear-system layer between GPU scans and solver core. |
| `kernels/block_tridiagonal.py` | Active CPU Thomas primitive for 2x2 double-cable systems | Keep CPU solver primitive isolated; do not reintroduce diagnostic batched/PCR variants here. |
| `kernels/double_cable_step.py` | GPU double-cable batch membrane/cable step helpers | Keep near the double-cable GPU scan body. |
| `kernels/triton_double_cable.py` | Triton implementation details | Keep isolated; later move under `linear_systems/triton_double_cable.py` only if linear systems become their own package. |
| `recording/observer.py` | VmRaster plan/state/update/finalize | Keep JAX-specific update code, but move observer plan validation/metadata toward runtime-neutral output contracts if needed. |
| `recording/lowering.py` | Observer lowering and VmRaster plan caching | Keep JAX adapter; later split cache key helpers if NumPy shares them. |
| `recording/results.py` | Trim/finalize batch kernel result payloads | Keep outside `kernels/`; this is JAX result glue rather than a kernel body. |
| `preparation/caches.py` | JAX runtime and factorized forcing caches | Keep JAX-specific; import runtime `types.py`, not the large `runtime.py`. |
| `preparation/shape_bucketing.py` | JAX double-cable shape bucketing | Keep JAX-specific. |
| `benchmarking/profile.py` and `benchmarking/metadata.py` | JAX profiling and benchmark metadata | Keep JAX-specific; move under `runtime/jax/benchmarking/` later. |
| `membranes/*` | JAX membrane compilation/lowering/backends/stacking | Keep as a subpackage; it is already close to the target shape. |

## Proposed Target Layout

```text
src/axonscope/runtime/
  input_contract.py          # semantic modes, capabilities, shapes
  input_payloads.py          # sparse/factorized payload dataclasses, array-agnostic
  input_planning.py          # shared waveform/current/footprint planning
  output_contract.py
  recording.py
  solver_axon.py
  host_preparation.py

  jax/
    __init__.py
    group_runner.py          # JAX batch group execution entry point
    types.py                 # SimulationGrid, SolverRuntime, CableRuntime...
    cable_geometry.py        # JAX cable geometry/diffusion helpers
    policy/
      __init__.py            # typed JAX solver request constructors
      execution.py           # former policy/execution.py
      engine.py              # platform dispatcher
      engine_common.py       # shared policy validation
      engine_cpu.py
      engine_gpu.py
      engine_types.py
    preparation/
      base.py                # base single-row runtime construction/sampling
      runtime.py             # batch SolverRuntime construction/cache use
      stacking.py            # JAX row stacking/materialization
      caches.py              # bounded JAX runtime/forcing caches
      shape_bucketing.py
    inputs/
      lowering.py            # JAX adapter from semantic lowering to payloads
      extracellular.py       # extracellular input materialization and footprint caches
      intracellular.py       # intracellular current-density materialization
      stimulus.py            # scalar JAX stimulus callable compilation
      payloads.py            # JAX materializers for sparse/factorized payloads
    recording/
      observer.py            # VmRaster plan/state/update/finalize
      lowering.py            # observer/recording lowering cache
      results.py             # BatchKernelResult wait/finalize/trim helpers
    kernels/
      double_cable_step.py   # GPU double-cable membrane/cable step helpers
      double_cable_linear.py # double-cable linear-system layout/assembly
      block_tridiagonal.py   # CPU Thomas primitive
      single_cable.py
      double_cable.py
      double_cable_cpu.py
      double_cable_gpu.py
      chunking.py
      factorized.py
      arrays.py
      linear_systems.py or linear_systems/
      triton_double_cable.py
    observers/
      vm_raster.py
      lowering.py
    membranes/
      compile.py             # membrane-to-JAX compiler bridge
      ...
    benchmarking/
      profile.py
      metadata.py
```

## NumPy/SciPy Runtime Contract Implications

The future NumPy/SciPy runtime should not import `runtime/jax`. It should share
only runtime-neutral contracts:

- `SolverAxon`: descriptive axon to numerical arrays.
- `RuntimeInputContract`: cable formulation, padded `Nx`, dtype/time grid,
  per-cable solver policy, recording and observer plan, intracellular mode,
  extracellular mode.
- `input_planning.py`: shared-current/scaled-waveform/current-table decisions.
- `input_payloads.py`: sparse intracellular and factorized extracellular
  payload shapes.
- `output_contract.py` and `recording.py`: output/recording lowering semantics.
- `host_preparation.py`: NumPy geometry and padding helpers.

Everything with `jax`, `jnp`, `lax.scan`, `jax.jit`, device placement, P100/Triton,
or JAX profiler stays inside `runtime/jax`.

## Recommended Migration Order

1. **No-risk boundary cleanup**
   - Keep current behavior.
   - Add architecture guardrails that `solvers/` contains only options.
   - Finish moving runtime-neutral `SolverAxon` and docs. This is already in
     progress in the working tree.

2. **Keep deleted direct kernels deleted**
   - Done: direct scan helpers and old direct kernel classes were removed from
     active runtime code.
   - Do not reintroduce `SingleCableKernel`, `DoubleCableKernel`, or
     `KernelResult`; tests should validate through public/batch routes or
     explicit dense numerical references.
   - Keep production code importing only the batch kernels and retained shared
     kernel helpers.

3. **Extract array/payload contracts**
   - Done: `SparseIntracellularCurrentDensityBatch` and
     `FactorizedExtracellularPotentialBatch` live in runtime-neutral
     `runtime/input_payloads.py`.
   - JAX materialization functions remain in `runtime/jax/inputs/payloads.py`
     with guardrails preventing that module from re-owning the payload classes.

4. **Split input builders**
   - Move JAX intracellular builders to `runtime/jax/inputs/intracellular.py`.
   - Move JAX extracellular builders and footprint cache to
     `runtime/jax/inputs/extracellular.py`/`footprints.py`.
   - Keep semantic selection in `runtime/input_planning.py` and
     `runtime/input_contract.py`.

5. **Split runtime dataclasses from preparation**
   - Done: `SimulationGrid`, `MembraneRuntime`, `CableRuntime`,
     `StimulationRuntime`, `ExtracellularRuntime`, and `SolverRuntime` live in
     `runtime/jax/types.py`.
   - Done: caches and batch hot-path modules depend on `types.py`, not
     `runtime.py`.
   - Done: membrane compilation moved to `runtime/jax/membranes/compile.py`.
   - Remaining: move preparation helpers and cache keys out of the large
     `runtime.py`.

6. **Split active batch kernels**
   - Done: `BatchKernelResult` lives in `runtime/jax/recording/results.py`.
   - Done: single-cable and double-cable kernel classes live in separate files.
   - Done: chunking, factorized forcing, and input coercion helpers are split
     out of the former batch-kernel monolith.

7. **Only then revisit dense recording support**
   - Batch-native dense recording for full Vm/gates/currents/state should be
     designed explicitly.
   - Do not use the deleted direct kernels as a shortcut; dense recording tests
     need dedicated batch-native behavior plus explicit numerical references.

## Acceptance Criteria For This Reorg

- No public API change unless explicitly planned.
- `AxonSimulation(...).run()` still enters JAX only through
  `runtime.execution -> runtime.jax.runner`.
- `runtime/jax` imports runtime-neutral contracts, but runtime-neutral modules
  never import `runtime/jax`.
- `solvers/` remains options-only.
- Existing P11-sensitive benchmark paths are unchanged until after the move,
  then revalidated with focused single-cable/double-cable CPU/GPU benchmark
  slices.
- The future NumPy/SciPy runtime can implement the same input/output contracts
  without importing JAX modules.
