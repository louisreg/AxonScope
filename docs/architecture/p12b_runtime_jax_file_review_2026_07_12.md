# P12B Runtime JAX File Review

Date: 2026-07-12

Scope: `src/axonscope/runtime/jax/` after scalar-route removal, direct kernel
removal, `types.py` extraction, and membrane compiler extraction.

## Summary Findings

- Most files are still active. `vulture` reports no high-confidence dead code
  in `runtime/jax` after the current cleanup.
- `runtime/jax/execution/__init__.py` has been deleted with a guardrail that
  keeps the empty package from coming back.
- The former `kernels/batch.py` monolith has been split into
  `single_cable.py`, `double_cable.py`, `chunking.py`, `factorized.py`,
  `inputs.py`, and `recording/results.py`.
- The former monolithic input builder has been split into `inputs/extracellular.py` and
  `inputs/intracellular.py`. Remaining input cleanup is smaller: payload
  contract separation and possible footprint helper extraction.
- The current hot paths should not be modified casually. The next cleanup
  should split files along existing boundaries while preserving imports and
  behavior first, then benchmark.
- Base runtime preparation has moved from the root to `preparation/base.py`.
  It still mixes cache keys, preparation, and direct sampling helpers, but the
  root JAX package is no longer carrying that implementation.

## Priority Actions

1. Move cache-key helpers out of `preparation/base.py` only if the remaining
   coupling becomes painful.

## File Review

| File | Status | Review | Recommendation |
| --- | --- | --- | --- |
| `__init__.py` | Keep | Thin public JAX namespace exposing typed solver policy constructors. Good shape. | Keep as-is. Do not export runtime internals here. |
| `inputs/payloads.py` | Keep, later split | Owns sparse intracellular and factorized extracellular payload dataclasses plus JAX materializers. The dataclasses are close to runtime-neutral, while materializers are JAX-specific. | Later move payload dataclasses to `runtime/input_payloads.py`; keep materializers in `runtime/jax/inputs/materialize.py`. |
| `kernels/single_cable.py` | Keep | Active single-cable batch hot path. Cable ownership is now explicit after the batch split. | Keep; only extract repeated orchestration after performance checks. |
| `kernels/double_cable.py` | Keep | Active double-cable batch hot path plus internal solver selection. Cable ownership is now explicit after the batch split. | Keep; preserve validated CPU/GPU solver performance. |
| `kernels/chunking.py` | Keep | Shared time-chunk and VmRaster chunk-state helpers. | Keep shared between cable families. |
| `kernels/factorized.py` | Keep | Shared factorized-current and single-cable forcing helpers. | Keep shared; do not move solver policy here. |
| `kernels/inputs.py` | Keep | Shared input coercion and recording helpers for batch kernels. | Keep; later review what can become runtime-neutral for NumPy/SciPy. |
| `recording/results.py` | Keep | Owns `BatchKernelResult`, result trimming/finalization/wait-target helpers. It is JAX runtime result glue, not a kernel implementation. | Keep outside `kernels/`. |
| `benchmarking/profile.py` | Keep | JAX-specific benchmark interface, profiler start/stop, trace annotation, memory profile save. Correctly stays out of runtime-neutral benchmark API. | Keep; later move under `runtime/jax/benchmarking/profile.py`. |
| `benchmarking/metadata.py` | Keep | JAX metadata collector for lowered inputs and memory estimates. It reads lowered JAX payloads but does not run kernels. | Keep; later move under `runtime/jax/benchmarking/metadata.py`. |
| `cable_geometry.py` | Keep | Small JAX cable-geometry and diffusion helper module split out of the former `common.py`. It is used by runtime preparation and kernels. | Keep outside `kernels/`; consider runtime-neutral sharing only when the NumPy/SciPy runtime needs the exact same contract. |
| `kernels/double_cable_linear.py` | Keep | Double-cable layout conversion, static-term preparation, system assembly, and Triton solve bridge split out of the former `common.py`. | Keep as the shared double-cable linear-system layer. |
| `kernels/block_tridiagonal.py` | Keep | Active CPU Thomas primitive split out of the former `common.py`. | Keep only supported CPU Thomas code here; rejected diagnostic solvers stay archived. |
| `execution/__init__.py` | Deleted | Empty package with no active imports. Historical `scalar_runner.py` is already removed. | Guardrail keeps `runtime/jax/execution/` absent. |
| `policy/execution.py` | Keep | JAX execution context, device resolution, precision validation, and policy-to-engine resolution. Good runtime-specific boundary. | Keep; later move to `policy/execution.py` if policy becomes a subpackage. |
| `group_runner.py` | Keep | Main JAX group orchestration. It prepares runtime, lowers inputs, calls kernels, waits, records metadata, and dispatches results. Still a little wide but coherent as the entry point. | Keep as orchestrator; later rename to `runner.py` and keep it thin. |
| `inputs/extracellular.py` | Keep | Builds dense/factorized extracellular JAX inputs, samples footprints, owns footprint caches, resolves positions/dtypes, and assembles factorized payloads. | Keep; later extract footprint helpers only if it reduces size without obscuring the hot path. |
| `inputs/intracellular.py` | Keep | Builds dense/sparse intracellular current-density batches, including the fast one-pulse sparse current-clamp path. | Keep; this is the JAX adapter for intracellular input materialization. |
| `inputs/lowering.py` | Keep | Good decision layer from semantic contract to concrete input format. Uses lazy imports to avoid coupling callers to concrete builders. | Keep. |
| `kernels/triton_double_cable.py` | Keep | Isolated optional Triton implementation. Large but bounded to one solver family and dependency surface. | Keep isolated; later move under `linear_systems/triton_double_cable.py` only if linear systems become their own package. |
| `kernels/double_cable_cpu.py` | Keep | CPU double-cable scan body and Thomas solve binding. | Keep CPU-specific. |
| `kernels/double_cable_gpu.py` | Keep | GPU double-cable scan body and tiled-Thomas/Triton solve binding. | Keep GPU-specific. |
| `membranes/__init__.py` | Keep | Empty subpackage marker. Harmless and useful for package clarity. | Keep. |
| `membranes/backend.py` | Keep | Core JAX membrane backend implementations: uniform, heterogeneous, padded, gated-leak stack, row-indexed. Active and conceptually cohesive. | Keep; later consider splitting stack/row-indexed backends only if the file grows again. |
| `membranes/compile.py` | Keep | New home for public membrane-to-JAX compilation and backend construction. Correct boundary. | Keep; consider sharing solver-option cache-key helper with `runtime.py` later. |
| `membranes/layout.py` | Keep | Heterogeneous membrane layout wrapper and observable-name aggregation. Small and cohesive. | Keep. |
| `membranes/model_ir_lowering.py` | Keep | JAX lowering from Model IR expressions/equations to executable membrane program callbacks. Active compiler code. | Keep; no runtime cleanup needed now. |
| `membranes/program.py` | Keep | JAX membrane program facade and observable aggregation helpers. Active and cohesive. | Keep. |
| `membranes/stacking.py` | Keep | JAX-specific gated/leak row stacking optimization. Active host-side optimization. | Keep; later review whether more of its signatures can be runtime-neutral. |
| `recording/observer.py` | Keep, minor rename later | VmRaster plan/state/update/finalize. Active observer-only path. The `scalar` helper names mean one-row table updates, not the removed scalar solver route. | Keep; later rename scalar helpers to row/probe-table wording to avoid confusion. |
| `policy.py` | Keep | Typed JAX solver constructors and CPU/GPU namespaces. Good user-facing runtime-specific policy surface. | Keep. CPU policy should stay minimal: double-cable auto/thomas only. |
| `recording/lowering.py` | Keep | Lowers observer/recording definitions to JAX VmRaster plans and caches them. Good adapter boundary. | Keep; later move generic observer-definition signatures toward runtime-neutral output contracts if NumPy shares them. |
| `preparation/base.py` | Keep | Base runtime preparation bridge: grid, membrane runtime, cable runtime, stimulation runtime, extracellular runtime, solver runtime, sampling helpers, and cache keys. | Keep under `preparation/`; split cache keys later only if needed. |
| `preparation/caches.py` | Keep | Small bounded caches for batch runtime and factorized forcing. Good boundary and now depends on `types.py`, not `runtime.py`. | Keep. |
| `preparation/runtime.py` | Keep | Batch `SolverRuntime` construction and runtime cache use. Active and performance-sensitive. | Keep as the JAX batch-runtime constructor. |
| `preparation/stacking.py` | Keep | Host-side cable, membrane, Cm, and extracellular row stacking for JAX batch runtimes. | Keep split from runtime construction. |
| `preparation/shape_bucketing.py` | Keep | Small double-cable batch/Nx bucketing helper with metadata. Isolated and useful. | Keep. |
| `kernels/double_cable_step.py` | Keep | Shared membrane/cable step helpers used by the GPU double-cable batch kernel. Cohesive and belongs near kernel code. | Keep near the cable-specific kernels. |
| `policy/engine.py` | Keep | Dispatches CPU/GPU runtime policy to concrete JAX solver engine. Small and clean. | Keep. |
| `policy/engine_common.py` | Keep | Shared JAX solver-policy validation for single/double cable requests. | Keep flattened under `policy/`; no nested solver-engine package. |
| `policy/engine_cpu.py` | Keep | CPU solver policy resolver. Should remain intentionally simple: single-cable tridiagonal, double-cable thomas. | Keep; remove unsupported CPU double-cable paths if any reappear. |
| `policy/engine_gpu.py` | Keep | GPU solver policy resolver. Owns auto/default mapping to JAX/Triton/PCR-style routes. | Keep; final defaults should be benchmark-driven. |
| `policy/engine_types.py` | Keep | `JaxSolverEngine` dataclass. Small and clean. | Keep. |
| `inputs/stimulus.py` | Keep | Compiles scalar descriptive stimuli to JAX callables. | Keep under inputs; extracellular and intracellular compilers consume it. |
| `types.py` | Keep | JAX prepared-runtime dataclasses. New stable internal type contract. | Keep and guardrail against moving dataclasses back into `runtime.py`. |

## Notes For NumPy/SciPy Runtime Preparation

- The future NumPy/SciPy runtime should not import any `runtime/jax/*` module.
- Good candidates to become runtime-neutral later:
  `inputs/payloads.py` payload dataclasses, some input/recording signatures,
  cache-key helpers based on host arrays, and shape/contract descriptions.
- JAX-only code should remain where it is: `jax.jit`, `jax.lax.scan`, device
  placement, Triton, profiler hooks, and JAX membrane program execution.
