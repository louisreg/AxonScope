# P11D Solver Engine Migration Plan

Date: 2026-07-09

This is the migration plan for flattening AxonScope solver execution around a
small typed policy surface:

```text
device: CPU / GPU
precision: FP32 / FP64
single-cable solver policy
double-cable solver policy
solver-specific options
recording / observers as one output policy
```

The migration is intentionally allowed to break pre-release APIs. The target is
one clean execution model, not compatibility with every intermediate P11
experiment.

## Goal

Current solver routing still exposes too many implementation axes:

```text
scalar vs batch
single-cable vs double-cable
CPU vs GPU
Thomas vs PCR vs PCR-SoA vs Triton
full Vm vs probe Vm vs observer-only
```

The target mental model should be:

```text
AxonSimulation(...)
    -> typed ExecutionPolicy
    -> CPU or GPU solver engine
    -> single-cable or double-cable equation path inside the engine
    -> one output/recording sink model
```

`scalar` should become `batch_size == 1`, not a separate solver family.

## Non-Goals

- Do not add a NumPy runtime in this phase.
- Do not make Triton default until the P11C-F policy benchmark is summarized.
- Do not add membrane-model-specific runtime branches.
- Do not expose benchmark-only solver probes as public API.
- Do not keep compatibility shims for the current pre-release
  `double_cable_block_solver` placement if the new API is cleaner.

## Triton Validation Objective

The practical objective is to validate the looped Triton tiled-Thomas route
deeply enough that it can become the preferred GPU double-cable route if the
evidence supports it. P11C-F and this migration should therefore produce the
data needed to make a real promotion decision, not merely keep Triton as an
interesting benchmark side path.

Promotion should be the default hypothesis for large GPU fp32 double-cable
populations, because the current warm evidence is strong. The benchmark gates
still have to disprove or constrain that hypothesis by checking small batches,
`Nx` buckets, recording modes, cold/warm behavior, memory pressure, dependency
failure modes, and physical curve agreement.

If the validation passes, the clean end state should be:

```text
GPU engine, double-cable, supported fp32 large-population shapes
    -> tiled Thomas / Triton route preferred
```

If validation fails or only holds for a narrow region, keep Triton as an
explicit solver option or benchmark-only route for that region, and keep the
current JAX GPU policy elsewhere.

## Public Typed API Target

The existing public typed values stay:

```python
axs.runtime.auto
axs.runtime.jax
axs.Device.cpu()
axs.Device.gpu(0)
axs.PrecisionPolicy.float32()
axs.PrecisionPolicy.float64()
axs.ExecutionPolicy(...)
```

Add solver selection to `ExecutionPolicy` through typed values, not raw strings.

Candidate public shape:

```python
policy = axs.ExecutionPolicy(
    runtime=axs.runtime.jax,
    device=axs.Device.gpu(0),
    precision=axs.PrecisionPolicy.float32(),
    solvers=axs.SolverPolicy(
        single_cable=axs.runtime.jax.SingleCableSolver.auto(),
        double_cable=axs.runtime.jax.DoubleCableSolver.auto(),
    ),
)
```

Explicit solver examples:

```python
cpu_policy = axs.ExecutionPolicy(
    device=axs.Device.cpu(),
    precision=axs.PrecisionPolicy.float64(),
    solvers=axs.SolverPolicy(
        double_cable=axs.runtime.jax.cpu.DoubleCableSolver.thomas(),
    ),
)

gpu_policy = axs.ExecutionPolicy(
    device=axs.Device.gpu(0),
    precision=axs.PrecisionPolicy.float32(),
    solvers=axs.SolverPolicy(
        double_cable=axs.runtime.jax.gpu.DoubleCableSolver.pcr_soa(),
    ),
)
```

If P11C-F supports Triton promotion, the future explicit route should look like
this rather than exposing the current benchmark label:

```python
gpu_policy = axs.ExecutionPolicy(
    device=axs.Device.gpu(0),
    precision=axs.PrecisionPolicy.float32(),
    solvers=axs.SolverPolicy(
        double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
            block_b=64,
            allow_fallback=False,
        ),
    ),
)
```

The current internal name `jax_triton_loop_xb` should remain an artifact label,
not a stable public name.

## Proposed Public Types

### `SolverPolicy`

```python
@dataclass(frozen=True)
class SolverPolicy:
    single_cable: SingleCableSolver = field(default_factory=SingleCableSolver.auto)
    double_cable: DoubleCableSolver = field(default_factory=DoubleCableSolver.auto)
```

This carries solver policy by cable family. `ExecutionPolicy.device` carries the
CPU/GPU axis, so the backend resolves the same `double_cable=auto` request to a
CPU Thomas route or a GPU adaptive PCR route depending on the active device.

### `SingleCableSolver`

```python
class SingleCableSolverKind(Enum):
    AUTO = "auto"
    JAX_TRIDIAGONAL = "jax_tridiagonal"

@dataclass(frozen=True)
class SingleCableSolver:
    kind: SingleCableSolverKind
```

Initial behavior:

```text
AUTO -> JAX tridiagonal for single-cable
```

### `DoubleCableSolver`

```python
class DoubleCableSolverKind(Enum):
    AUTO = "auto"
    THOMAS = "thomas"
    JAX_PCR = "jax_pcr"
    JAX_PCR_SOA = "jax_pcr_soa"
    TILED_THOMAS = "tiled_thomas"

@dataclass(frozen=True)
class DoubleCableSolver:
    kind: DoubleCableSolverKind
    pcr_options: PcrSolverOptions = field(default_factory=PcrSolverOptions)
    tiled_thomas_options: TiledThomasSolverOptions = field(
        default_factory=TiledThomasSolverOptions
    )
```

Initial behavior before Triton promotion:

```text
AUTO -> THOMAS on CPU, current JAX PCR policy on GPU
THOMAS -> exact Thomas route
JAX_PCR -> existing PCR route on GPU
JAX_PCR_SOA -> existing PCR-SoA route on GPU
TILED_THOMAS -> explicit experimental tiled-Thomas route on GPU
```

If P11C-F promotes Triton:

```text
AUTO may choose TILED_THOMAS only for proven fp32 GPU double-cable shapes.
Explicit TILED_THOMAS should fail loudly when dependency/device/dtype constraints
are not met, unless allow_fallback=True is explicitly requested.
```

### Solver-Specific Options

Keep options typed and local to each solver family:

```python
@dataclass(frozen=True)
class PcrSolverOptions:
    adaptive_threshold: int = 4096

@dataclass(frozen=True)
class TiledThomasSolverOptions:
    block_b: int | None = None
    allow_fallback: bool = False
    require_gpu: bool = True
```

Only add options after a benchmark shows they matter. Do not surface every
experimental knob from `benchmark/analysis`.

## What Moves Out Of `BatchOptions`

`BatchOptions` currently mixes:

- retained Vm policy;
- time chunking;
- double-cable block-solver choice.

Target split:

```text
Recording / observers
    -> RecordingPlan / ObserverPlan
    -> internal OutputPlan

ExecutionPolicy.solvers
    -> typed single-cable/double-cable solver selection

KernelExecutionOptions
    -> internal time chunking and backend execution controls
```

`BatchOptions.double_cable_block_solver` should be removed or made internal.
Public users should select solvers through `ExecutionPolicy.solvers`.

`BatchOptions.recording` should not be a public solver concept long-term. The
public output contract should be `Recording(...)` plus `observers=(...)`.

## Internal Engine Target

Add a backend-private solver engine layer:

```text
src/axonscope/runtime/jax/policy/
    __init__.py
    types.py
    policy.py
    cpu.py
    gpu.py
    layouts.py
    output.py
```

Internal protocol:

```python
class SolverEngine(Protocol):
    def solve_single_cable(
        self,
        batch: SingleCableBatch,
        output: OutputPlan,
    ) -> KernelOutput: ...

    def solve_double_cable(
        self,
        batch: DoubleCableBatch,
        output: OutputPlan,
    ) -> KernelOutput: ...
```

Engine selection:

```text
ExecutionPolicy.device -> CPU engine or GPU engine
ExecutionPolicy.precision -> dtype validation/lowering
ExecutionPolicy.solvers -> engine-local linear solver policy
```

Equation selection remains internal:

```text
single-cable equation path
double-cable equation path
```

But scalar/batch should not be a solver-engine distinction:

```text
one axon -> batch with B=1
many axons -> batch with B=N
```

## Shared Core

Share problem description and assembly as much as possible:

```text
src/axonscope/runtime/jax/solver_core/
    batches.py
    single_cable.py
    double_cable.py
    recording.py
    observers.py
    scan.py
```

Shared responsibilities:

- normalized batch state;
- membrane gate/current/conductance calls;
- single-cable RHS assembly;
- double-cable logical system assembly;
- extracellular RHS drive;
- output sink update;
- chunk/time-loop helpers;
- result packaging metadata.

Specialized responsibilities:

- CPU linear solve layout;
- GPU linear solve layout;
- GPU Triton `XB` packing/layout if promoted;
- low-level solver-specific fallback behavior.

## Layout Contract

Represent the double-cable linear system once, but allow engine-preferred
materialization:

```python
class DoubleCableLayout(Enum):
    BX = "batch_space"
    XB = "space_batch"
    TILED = "tiled"
```

Shared logical assembly should support:

```python
assemble_double_cable_system(..., layout=engine.preferred_layout)
```

CPU can request `BX`.
Current JAX PCR/SoA can request `BX`.
Triton tiled Thomas can request `XB` or `TILED`.

This avoids duplicating physiology and RHS logic while still allowing the GPU
hot path to use a useful memory layout.

## Unified Output Policy

Create one internal output model for full Vm, probe Vm, and observer-only:

```python
class OutputKind(Enum):
    FULL_VM = "full_vm"
    SAMPLED_VM = "sampled_vm"
    VM_RASTER = "vm_raster"
    NONE = "none"

@dataclass(frozen=True)
class OutputPlan:
    kind: OutputKind
    vm_indices: tuple[int, ...] | None
    observers: VmRasterPlan | None
    time_chunk_steps: int | None
```

Lowering:

```text
Recording + observers
    -> OutputRequest
    -> OutputPlan
    -> engine.solve_*(..., output=OutputPlan)
```

Observer-only should not have a separate orchestration route. It should be one
output sink choice inside the same time-loop structure.

Acceptance target:

```text
full Vm, probe Vm, and VmRaster observer-only differ only by output sink and
retained buffers, not by independent runtime orchestration.
```

## Migration Phases

### P11D-A - Public Policy Types

Status on 2026-07-09: implemented for the current JAX backend surface. Public
solver policy lives under `ExecutionPolicy.solvers`; `BatchOptions` no longer
contains double-cable solver selection.

- Add `SolverPolicy` plus runtime-specific constructors under
  `axs.runtime.jax` and solver-specific option dataclasses.
- Extend `ExecutionPolicy` with `solvers: SolverPolicy | None`.
- Keep `Device` and `PrecisionPolicy` unchanged; replace the root `Runtime`
  enum with named runtime targets under `axs.runtime`.
- Add validation:
  - CPU execution rejects GPU-only double-cable routes.
  - GPU Triton solver requires fp32, GPU device, and optional dependencies once
    it is public.
  - Mixed precision remains estimate-only until explicitly implemented.
- Add tests for typed construction, invalid combinations, equality/metadata,
  and public facade exports.

### P11D-B - Separate Output Options From Solver Options

Status on 2026-07-09: implemented for active simulation and curve benchmark
entry points. Benchmark CLI solver strings are translated to typed policy at the
benchmark boundary; internal benchmark-only overrides remain metadata-driven
and do not re-enter `BatchOptions`.

- Introduce internal `KernelExecutionOptions` for time chunking and backend
  execution controls.
- Introduce `OutputPlan` and make recording/observer lowering produce it.
- Remove double-cable solver selection from public `BatchOptions`.
- Update protocols and benchmarks to pass solver policy through
  `ExecutionPolicy`, not `BatchOptions`.
- Keep benchmark-only overrides separate from public policy metadata.

### P11D-C - Normalize B=1 And B>N

Status on 2026-07-12: implemented for batchable Vm/VmRaster outputs. One-row
groups use the batch route with `B=1`. `Recording.full()`/observable recordings
raise explicitly until dense gates/currents/conductances outputs are implemented
as batch-native result payloads.

- Make dispatch always produce normalized groups with a batch axis.
- Route one-axon execution through the same batch group lifecycle.
- Do not keep a temporary scalar fallback for unsupported features; unsupported
  dense observable recordings must fail before runtime dispatch.
- Ensure result records have one shape model for B=1 and B>N.

### P11D-D - Add CPU/GPU Engine Layer

Status on 2026-07-09: initial JAX engine-resolution layer implemented under
`src/axonscope/runtime/jax/policy/`. It resolves typed public policy
to CPU/GPU engine descriptors and feeds the current batch kernels; it is not yet
a full solver-core extraction.

- Add `policy/types.py` with the internal protocol.
- Add `policy/policy.py` to resolve a typed public policy to a concrete
  backend engine and internal solver route.
- Add `policy/cpu.py`:
  - single-cable JAX tridiagonal;
  - double-cable Thomas;
  - no public PCR choices initially.
- Add `policy/gpu.py`:
  - single-cable JAX tridiagonal;
  - double-cable current JAX PCR policy;
  - optional benchmark-only Triton route still gated.
- Route `group_runner.py` through the engine layer.

### P11D-E - Shared Single/Double-Cable Core

Status on 2026-07-09: initial shared core implemented. Batched membrane
gate/current/linearization/state operations and the batch-native double-cable
linear step now live in `src/axonscope/runtime/jax/solver_core.py`; the
stateful recorded-Vm and observer-only scans share those pieces while keeping
their output-specific loop logic local.

- Move duplicated scan/output logic from `batch_kernels.py` into
  `solver_core`.
- Factor shared double-cable system assembly with layout-aware output.
- Keep CPU/GPU differences localized to linear solve and layout conversion.
- Make `common.py` smaller:
  - production numerical primitives stay;
  - benchmark-only probes move to `benchmark/analysis` or a clearly marked
    backend experimental module.

### P11D-F - Recording And Observer Homogenization

Status on 2026-07-09: initial `OutputPlan` implemented for JAX group execution.
Full Vm, sampled Vm, and VmRaster observer-only now share a backend-local output
descriptor, while deeper loop unification remains part of P11D-E.

- Route full Vm, probe Vm, and observer-only through the same output-plan
  interface.
- Ensure VmRaster observer-only uses the same engine time loop as recorded Vm
  where feasible.
- Keep `observations["vm_raster"]` as the only solver-side public observer
  payload.
- Activation, threshold, latency, velocity, and recruitment remain
  result-side post-processing.

### P11D-G - Clean Public Surface

- Remove or privatize pre-migration names that only describe old routing:
  - public `BatchOptions.double_cable_block_solver`;
  - public `resolve_double_cable_block_solver(...)` if replaced by typed policy
    resolution;
  - docs/examples that teach forced string solver choices.
- Update examples to use:

  ```python
  execution_policy=axs.ExecutionPolicy(
      device=axs.Device.gpu(0),
      precision=axs.PrecisionPolicy.float32(),
      solvers=axs.SolverPolicy(
          single_cable=axs.runtime.jax.SingleCableSolver.auto(),
          double_cable=axs.runtime.jax.DoubleCableSolver.auto(),
      ),
  )
  ```

- Keep benchmark CLI string flags if useful, but map them to typed policies at
  the benchmark boundary and label them as benchmark vocabulary.

### P11D-H - Examples And Learning Path

Status on 2026-07-09: added
`examples/advanced/runtime/04_solver_policy.py` and updated
`examples/README.md`. Basic examples continue to use defaults.

- Update basic examples that mention runtime/device/precision/solver execution
  so they use typed `ExecutionPolicy`, `Device`, `PrecisionPolicy`, and
  `SolverPolicy` values.
- Keep introductory examples simple: most examples should use default solver
  policy and only show CPU/GPU or FP32/FP64 when that is the point of the
  example.
- Add or update targeted advanced examples when a solver concept is genuinely
  user-facing:

  ```text
  examples/advanced/runtime/
      solver_policy_cpu_gpu.py
      precision_and_device_policy.py
      gpu_solver_policy.py
  ```

- Keep benchmark/profiling examples under `benchmark/`, not under public
  examples, unless the example teaches a normal user-facing runtime workflow.
- If Triton is promoted later, add a dedicated advanced example showing the
  explicit solver policy, constraints, and fallback behavior. Do not document
  `jax_triton_loop_xb` as a public option.
- Update `examples/README.md` so the learning path points users to the right
  basic or advanced example instead of exposing old `BatchOptions` solver
  strings.

### P11D-I - Benchmark And Validation Gates

Before accepting the migration:

- local unit tests pass for public policy types, recording lowering, dispatch,
  and batch solver semantics;
- NRV tests run only if numerical behavior changes;
- CPU and GPU quick benchmark artifacts confirm no hidden route regression;
- P11C-F still remains the authority for deciding Triton public/default policy;
- no speed claim is made from migration alone unless fresh benchmark artifacts
  are generated.

## Breaking Changes To Accept

These are acceptable if the new API is cleaner:

- remove `BatchOptions.double_cable_block_solver` from public use;
- replace string solver names with typed solver policy objects;
- stop exporting `resolve_double_cable_block_solver(...)` as a user-facing
  helper if the new policy resolver supersedes it;
- remove scalar-vs-batch documentation language from user docs;
- require benchmarks to use explicit benchmark-only override flags for internal
  routes rather than sneaking them through public runtime options.

## Acceptance Criteria

The migration is complete when:

- one-axon and population simulations use the same normalized batch/result
  lifecycle except for explicitly documented unsupported fallbacks;
- high-level users can choose CPU/GPU, FP32/FP64, single-cable solver policy,
  double-cable solver policy, and solver-specific options through typed
  objects;
- recording/probe/observer-only outputs lower through one output-plan model;
- production code no longer exposes benchmark-only solver probes as runtime
  choices;
- current CPU auto policy is still Thomas-oriented;
- current GPU auto policy remains benchmark-backed and does not silently switch
  to Triton before P11C-F;
- Triton, if promoted later, has a clean public name and fails loudly when its
  constraints are not met.

## Suggested Implementation Order

1. Add typed public policies and tests without changing runtime behavior.
2. Move solver choice out of `BatchOptions` into policy resolution.
3. Add `OutputPlan` and unify recording/observer lowering.
4. Add CPU/GPU engine protocol and route current kernels through it.
5. Normalize B=1 through the batch lifecycle.
6. Split shared core from CPU/GPU linear-solver specializations.
7. Update examples, advanced examples, and `examples/README.md`.
8. Clean docs and benchmark CLI mapping.
9. Run focused tests and quick benchmarks.
10. Revisit Triton promotion only after P11C-F.
