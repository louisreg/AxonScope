# AxonScope Phase 4 Backend Boundary

Status: working inventory for Phase 4. This is not user documentation.

Update on 2026-06-14: PR 4.2 through PR 4.5 have moved batch group
execution, JAX batch input materialization, and scalar Crank-Nicholson
execution under `axonscope.backends.jax`. Low-level kernel/runtime modules
remain under `solvers/` for now because they still own tested numerical kernels
and are not duplicate public execution paths.

## Goal

Phase 4 isolates JAX runtime/lowering behind backend modules without changing
the public AxonScope API. The first move should be small: extract the current
batch group execution boundary, then move lower-level JAX helpers only when a
real responsibility moves with them.

This follows `GUIDELINES.md`: do not create empty backend packages, do not keep
obsolete forwarding modules, and keep planning/preparation backend-neutral.

## Current JAX-Owned Surface

The current JAX-owned implementation is spread across these areas:

- `src/axonscope/solvers/runtime.py`
  - Builds `SimulationGrid`, `MembraneRuntime`, `CableRuntime`,
    `StimulationRuntime`, `ExtracellularRuntime`, and `SolverRuntime`.
  - Converts descriptive axon/stimulation objects into JAX arrays/callables.
- `src/axonscope/solvers/kernels.py`
  - Scalar single-cable and double-cable JAX kernels.
- `src/axonscope/solvers/batch_kernels.py`
  - Batch single-cable and double-cable JAX kernels.
- `src/axonscope/solvers/crank_nicholson.py`
  - Public solver wrapper that prepares JAX runtime and invokes JAX kernels.
- `src/axonscope/stimulation/runtime.py`
  - JAX compiled stimuli, intracellular clamps, and extracellular contexts.
- `src/axonscope/icm/backends.py` and `src/axonscope/icm/membrane_layout.py`
  - JAX membrane backend objects used by solver kernels.
- `src/axonscope/channel_models/`
  - Current channel equations are JAX-compatible implementations rather than
    fully backend-neutral descriptions.
- `src/axonscope/dispatcher/runtime_batches.py`
  - Mixed responsibility: NumPy host preparation plus JAX array materialization
    for batch inputs.
- `src/axonscope/dispatcher/execution.py`
  - Mixed responsibility: dispatch orchestration plus JAX runtime preparation,
    parameter-batched runtime stacking, kernel invocation, and result splitting.
- `src/axonscope/simulation.py`
  - Mostly public orchestration, but still uses `jnp.take` for result slicing.

Benchmarking/profiling modules may import JAX for metadata/profiler support;
that is observability, not solver backend ownership.

## Smallest Useful Boundary

The first backend boundary should sit between prepared dispatch groups and
backend execution:

```text
DispatchPlan + PreparedCohort
        |
        v
JAX batch/scalar group runner
        |
        v
DispatchResult rows
```

The dispatcher should decide which group runs. The JAX backend should own:

- preparing `SolverRuntime`;
- preparing JAX input arrays from `PreparedCohort`;
- stacking row-specific runtimes for parameter batches;
- invoking scalar/batch kernels;
- synchronizing JAX outputs when hotpath benchmarking asks for it;
- converting backend arrays into backend-neutral result arrays at the boundary.

The dispatcher should keep:

- pool validation and input order;
- `DispatchPlan` construction;
- progress lifecycle;
- result placement back into input order;
- public diagnostic labels.

## Proposed Phase 4 PRs

### PR 4.1: Backend Boundary Inventory

Deliverables:

- This inventory.
- TODO roadmap for the first backend extraction.
- No code movement unless it directly reduces ambiguity.

### PR 4.2: JAX Group Runner Wrapper

Status: implemented.

Create the first real backend package only when moving execution code:

```text
src/axonscope/backends/
src/axonscope/backends/jax/
src/axonscope/backends/jax/group_runner.py
```

Move these responsibilities out of `dispatcher/execution.py`:

- `_run_single_cable_batch_group`
- `_run_double_cable_batch_group`
- `_with_batched_single_cable_runtime`
- `_with_batched_double_cable_runtime`
- runtime stacking/padding helpers used only by JAX batch execution

Keep `dispatcher/execution.py` as the orchestrator that calls the JAX group
runner and places returned `DispatchResult` rows.

### PR 4.3: JAX Input Lowering Split

Status: implemented as `axonscope.backends.jax.input_batches`, with
`dispatcher/runtime_batches.py` reduced to host-side row helpers.

Split `dispatcher/runtime_batches.py`:

- host/NumPy preparation stays in `preparation/`;
- JAX materialization and `jnp.asarray` helpers move under `backends/jax/`.

The fast paths added during Phase 3 should remain available through the JAX
backend runner.

### PR 4.4: Guardrails

Status: implemented for the current boundary.

Add tests that prevent new JAX imports from entering descriptive/public layers:

- `axons/`
- `membranes/`
- `recording.py`
- `population.py`
- `results/`
- `preparation/signatures.py`

Allowed JAX zones during the transition:

- `solvers/`
- `icm/`
- `channel_models/`
- `dispatcher/runtime_batches.py`
- `dispatcher/execution.py`
- `stimulation/runtime.py`
- `backends/jax/`
- `benchmarking/` for profiler/device metadata only

The allowed list should shrink as code moves.

### PR 4.5: Scalar Solver Boundary

Status: implemented as `axonscope.backends.jax.scalar_runner`.

After batch group execution is isolated, move scalar Crank-Nicholson execution
behind the same JAX backend family. This keeps scalar and batch execution from
diverging.

## Non-Goals For Phase 4

- Do not introduce a generic kernel IR.
- Do not add a NumPy reference backend yet.
- Do not change public simulation, recording, axon, or stimulation APIs.
- Do not move channel models before the group runner boundary is proven.
- Do not delete old solver modules until their responsibilities have moved and
  tests cover the new import path.

## Acceptance Criteria

Phase 4 is ready to close when:

- dispatcher execution no longer imports concrete JAX kernels directly;
- batch JAX execution lives behind a backend-owned group runner;
- descriptive/public modules remain backend-neutral by guardrail;
- hotpath traces still expose the same stage names or a documented replacement;
- the unit suite and hotpath smoke run pass after each move.
