# P11 Solver Policy And Runtime Cleanup Decisions - 2026-07-11

This note freezes the current solver-policy decisions before deleting or
renaming runtime paths. It complements the P11E extracellular input contract:
`docs/architecture/p11e_extracellular_input_contract_2026_07_11.md`.

## Current Decision

The immediate objective is a smaller runtime surface, not a new solver zoo.
AxonScope should expose one typed policy surface and keep benchmark-only probes
behind benchmark entry points until a full policy matrix promotes them.

### CPU

- Single-cable CPU keeps the JAX tridiagonal route.
- Double-cable CPU keeps only Thomas as a production route.
- `auto` on CPU resolves to Thomas for double-cable.
- CPU PCR, PCR-SoA, tiled-Thomas, and Triton routes are not production routes.
  They should not be advertised as supported CPU choices.

Some low-level CPU execution paths may remain temporarily as diagnostic or unit
equivalence helpers, but they should be moved behind explicit benchmark/test
boundaries or deleted during cleanup. They must not influence public policy.

### GPU

- Single-cable GPU keeps the JAX tridiagonal route for now.
- Double-cable GPU keeps typed explicit solver choices while the policy matrix
  is completed.
- The looped Triton tiled-Thomas route is the preferred promotion candidate for
  supported fp32 large-population double-cable GPU workloads.
- `auto` should not be changed blindly. It can prefer Triton only after the
  benchmark matrix confirms shape, dtype, recording-mode, cold/warm,
  dependency, memory, and correctness bounds.

### Input And Output Contract

- `observer_only`, `probe_vm`, and `full_vm` should share the same semantic
  extracellular input lowering contract.
- The preferred non-dense extracellular route is factorized:
  `factorized_footprint` plus `shared_current` or `scaled_shared_waveform`.
- `shared_current` is the canonical recruitment-style path.
- `scaled_shared_waveform` is the canonical threshold-style path when rows
  share temporal waveform shape and differ only by amplitude scale.
- Recording mode should affect the output sink, not the input representation,
  unless a capability explicitly rejects a compact input mode.

## Fresh Validation

Commit `199b7b9` clarified benchmark metadata so reports separate solver
kernel identity from input lowering identity.

Local validation:

- `compileall` on touched modules passed.
- Focused unit tests passed: `118 passed`.
- Local micro campaign checks confirmed the new summary fields.

Kaggle P100 mini validation from branch `kaggle-bench/199b7b9`:

- `axs-p11e-contract-single-gpu-199b7b9`: single-cable GPU,
  recruitment curves, `Naxons=512`, `Nx=89`, fp32,
  `observer_only` and `probe_vm`, same and different diameters, `4/4 passed`.
- `axs-p11e-contract-double-gpu-199b7b9`: double-cable GPU,
  recruitment curves, `Naxons=512`, `Nx=89`, fp32,
  `observer_only` and `probe_vm`, same and different diameters, tiled Thomas
  `block_b=64`, `4/4 passed`.

The resulting summaries report:

| cable | kernel | input format | input mode | output sinks |
| --- | --- | --- | --- | --- |
| single-cable | `jax_tridiagonal` | `factorized_footprint` | `shared_current` | `observer_only`, `probe_vm/probes` |
| double-cable | `jax_triton_loop_xb` | `factorized_footprint` | `shared_current` | `observer_only`, `probe_vm/probes` |

This validates the reporting contract and the shared-current recruitment path.
It does not by itself close the full threshold/scaled-waveform policy matrix.

## Cleanup Rules

1. Do not keep public aliases for rejected routes.
2. Do not keep CPU double-cable PCR/PCR-SoA/Triton as production choices.
3. Keep Triton GPU double-cable selectable through typed policy while it is
   being validated; promote to default only after the full matrix.
4. Keep benchmark-only labels such as `jax_triton_loop_xb` out of stable public
   user-facing names.
5. Prefer one shared input-lowering contract for single-cable and double-cable.
6. Delete tests for removed production behavior; keep only focused numerical
   equivalence tests for routes that remain supported or explicitly diagnostic.

After the first P11F cleanup pass, active curve benchmarks also follow this
rule: users select the Triton/tiled-Thomas candidate with
`--double-cable-block-solver tiled_thomas` and `--tiled-thomas-block-b ...`.
The old `--benchmark-double-cable-block-solver jax_triton_loop_xb` hook is
removed; `jax_triton_loop_xb` remains only a runtime/artifact label.

The next cleanup pass removed the backend-local
`runtime/jax/solver_engines/block_solvers.py` resolver. `auto` is now resolved
by typed solver policy before kernel dispatch; low-level double-cable kernels
accept only concrete backend-private routes (`thomas`, `pcr`, `pcr_soa`,
`pcr_adaptive`) or explicitly permitted benchmark/internal labels.

The solver route is now carried through JAX orchestration as one
`JaxSolverEngine` value. `DoubleCableBatchKernel.run(...)` consumes that engine
instead of parallel raw arguments such as `double_cable_block_solver`,
`allow_internal_double_cable_block_solver`, and
`double_cable_tiled_thomas_block_b`.

## Next Cleanup Slice

- Make CPU double-cable policy and docs say one thing everywhere:
  `auto == thomas`, and Thomas is the only supported CPU double-cable solver.
- Move any remaining CPU PCR/PCR-SoA/Triton usage into benchmark-only or
  diagnostic tests, then delete stale public-facing coverage.
- Keep the GPU double-cable policy matrix before making Triton the default.
- Keep the single-cable route simple unless a future GPU benchmark shows a real
  low-level solver bottleneck.
