# AxonScope TODO

Living execution plan for AxonScope cleanup, benchmark evidence, validation,
documentation, and next runtime phases.

`GUIDELINES.md` is the architecture reference. `AGENTS.md` is the agent working
guide. Source, tests, runnable examples, and fresh benchmark reports remain the
implementation truth.

This file is intentionally compact. The full pre-cleanup ledger was archived in
`docs/architecture/todo_archive_before_cleanup_2026_07_12.md`.

## Snapshot

Updated on 2026-07-12 during P12B runtime/JAX cleanup.

Current state:

- P7 is closed: public membrane authoring is class-based through
  `axs.membranes.Model`; built-ins live under
  `src/axonscope/membranes/models/`.
- Historical `channel_models`, `icm`, `model_ir/models`, and
  `model_ir/builtins.py` paths are removed and must stay absent.
- Model IR remains internal compiler/runtime vocabulary. Users write membrane
  models, equations, parameters, gates, currents, and observables.
- `axs.runtime.numpy` is reserved for a future real NumPy/SciPy reference
  runtime. It must not become a JAX-backed compatibility path.
- Solver-side observer-only execution is the strict VmRaster path under
  `observations["vm_raster"]`; activation, latency, velocity, threshold, and
  recruitment summaries remain post-processing.
- `PeakVoltage` remains post-hoc on recorded Vm unless a dedicated benchmarked
  solver-side design is accepted.
- P3 is paused after current-docs cleanup. Tutorials/Sphinx/docstrings remain
  open.
- P11 is closed for the current JAX runtime, benchmark, and solver-policy
  stabilization pass. Deferred runtime, benchmark, and solver-policy work is
  tracked in `docs/architecture/p11_closeout_2026_07_12.md`.
- Current solver-policy decisions are tracked in
  `docs/architecture/p11_solver_policy_cleanup_decisions_2026_07_11.md`:
  CPU double-cable keeps only Thomas as a production route; GPU double-cable
  currently resolves `auto` through the Triton/tiled-Thomas route while the
  full policy matrix remains the benchmark gate; single-cable stays on the JAX
  tridiagonal route for now.

Fresh local validation from the 2026-07-02 audit:

```text
python -m compileall -q src tests/unit
pytest -q tests/unit --tb=short
587 passed, 1 skipped in 424.89s
```

## Non-Negotiables

- AxonScope is pre-release with one active user. Prefer clean deletion and
  direct convergence over retrocompatibility, shims, aliases, or deprecated
  wrappers.
- One concept, one public name, one execution path, one canonical public result
  model.
- Public examples must not import solver/runtime internals.
- Every public feature, option, workflow, analysis, runtime mode, inspection
  view, or advanced concept must be documented in runnable examples or removed
  from the public surface.
- World/anatomical coordinates, trajectories, nerve geometry, electrode CAD,
  surgical placement, and FEM solving stay outside AxonScope core. AxonScope
  consumes intrinsic positions and sampled footprints.
- Do not remove unfinished TODO items unless they are completed, rejected, or
  moved to a named tracking document.

## Active Plan

### P12 - Runtime Cleanup, Studies, Serialization, Integration

- [x] P12A runtime contract and sanity benchmark gate:
  use `docs/architecture/p12_runtime_contract_2026_07_12.md` and
  `docs/architecture/p12a_jax_runtime_audit_2026_07_12.md` as the completed
  local CPU plus Kaggle GPU smoke gate. This validates that the initial
  runtime-contract cleanup still runs on the P11-sensitive single-cable and
  double-cable observer-only paths.
- [ ] P12B runtime/JAX cleanup:
  use `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md` as the active
  migration note. Homogenize non-solver preparation, recording/observer
  lowering, input semantics, benchmark metadata, and result assembly between
  single-cable and double-cable paths as much as possible without losing P11
  performance.
  - [x] Remove unused rate-table option, the direct public
    `CrankNicholson`/`Solver` execution facade, and the JAX scalar fallback;
    one-row public simulations use the batch route with `B=1`.
  - [x] Split the former JAX batch-kernel monolith into explicit
    `runtime/jax/kernels/single_cable.py`,
    `runtime/jax/kernels/double_cable.py`, shared chunking/factorized/input
    helpers, and `runtime/jax/recording/results.py`; remove the old
    `runtime/jax/batch_kernels.py`/`kernels/batch.py` path.
  - [x] Split active shared numerical primitives out of the old
    `runtime/jax/kernels/common.py` bucket into
    `runtime/jax/cable_geometry.py`,
    `runtime/jax/kernels/double_cable_linear.py`, and
    `runtime/jax/kernels/block_tridiagonal.py`; remove legacy PCR/PCR-SoA and
    diagnostic batched-Thomas helpers from active runtime code, and keep
    double-cable scan bodies split into CPU Thomas and GPU tiled-Thomas/Triton
    files.
  - [x] Review `chunking`, `factorized`, `inputs`, `results`, `core`, and
    `cable_geometry` one by one: keep `chunking`, `factorized`, and `inputs`
    as kernel-only shared helpers; move JAX geometry to
    `runtime/jax/cable_geometry.py`, move result synchronization to
    `runtime/jax/recording/results.py`, and rename the former vague kernel
    `core.py` to `runtime/jax/kernels/double_cable_step.py`.
  - [x] Archive historical P11B/P11C solver probes under
    `benchmark/legacy/p11_solver_exploration/`, delete the
    `jax_triton_cold_start_audit` runner/test surface, and keep the active
    double-cable routes limited to CPU Thomas and GPU looped Triton/tiled
    Thomas.
  - [x] Clean the single-cable JAX kernel surface: split scan bodies into
    `runtime/jax/kernels/single_cable_scans.py`, remove the unsupported
    observer-only sparse-current plus dense-Vstim route, and keep only dense,
    factorized, factorized-sparse, and zero-sparse routes that are reachable
    from the runtime lowering contract.
  - [x] Reorganize the remaining JAX runtime modules by responsibility:
    typed runtime policy in `runtime/jax/policy/`, input payload/build/lowering
    in `runtime/jax/inputs/`, host-side batch preparation and caches in
    `runtime/jax/preparation/`, observer/recording/result synchronization in
    `runtime/jax/recording/`, and JAX profiling/metadata helpers in
    `runtime/jax/benchmarking/`.
  - [ ] Reintroduce dense recording only as a batch-native result path:
    `Recording.full()`, gates, currents, conductances, and state variables must
    lower through the batch route for `B=1` and `B>N`, with explicit signal
    names, result manifests, memory accounting, tests, examples, and benchmark
    evidence. Do not reintroduce scalar fallback execution for this.
- [x] Audit `src/axonscope/runtime/jax/` for dead, duplicate, or cable-specific
  host-side code. Delete unused paths, keep solver/kernel-specific code inside
  the JAX runtime, and move semantic-only reusable contracts to
  `src/axonscope/runtime/` when they can support a future NumPy/SciPy runtime.
  Use `docs/architecture/p12b_jax_runtime_reorganization_proposal_2026_07_12.md`
  as the proposed file-responsibility map before moving more modules.
  Method: do the whole audit/move/delete pass first, without running the full
  test suite after each file or folder. Validate once at the end with
  `compileall`, `tests/unit`, `git diff --check`, `vulture`, and only then
  targeted benchmarks if hotpath behavior changed.
  For every directory or root file below, verify that all retained paths are
  still used, that there is no dead or duplicate code, that responsibility is
  not split across redundant routes, and that the code is genuinely
  JAX-specific. If a contract, planning rule, metadata shape, or host-side
  semantic helper is runtime-neutral, move it to `src/axonscope/runtime/` so it
  can serve the future NumPy/SciPy runtime too.
  - [x] Root JAX files: `__init__.py`, `group_runner.py`, `types.py`, and
    `cable_geometry.py`.
  - [x] `runtime/jax/policy/`: typed JAX solver requests, execution context,
    device/precision lowering, and solver-engine resolution.
  - [x] `runtime/jax/inputs/`: payload dataclasses, dense/sparse/factorized
    builders, footprint caches, and semantic input lowering.
  - [x] `runtime/jax/preparation/`: batch runtime materialization, caches,
    shape bucketing, host-to-device array preparation, and row stacking.
  - [x] `runtime/jax/recording/`: VmRaster observer plan/state/update,
    recording lowering, result synchronization, waits, trimming, and
    finalization.
  - [x] `runtime/jax/kernels/`: single-cable, double-cable CPU/GPU, shared
    chunking/factorized/input helpers, double-cable linear-system helpers, and
    Triton integration.
  - [x] `runtime/jax/membranes/`: membrane compiler bridge, Model IR lowering,
    membrane backend implementations, layout aggregation, programs, and
    stacking optimizations.
  - [x] `runtime/jax/benchmarking/`: JAX profiling hooks, memory profiling,
    benchmark metadata, and estimate/inspection support helpers. Source audit
    pass started: the runtime-neutral batch memory-estimate math now lives in
    `runtime/memory_estimates.py`.
- [ ] Define and enforce the runtime input contract before implementing
  `axs.runtime.numpy`: prepared batches must expose one cable formulation, one
  padded `Nx`, a dtype/time grid, typed per-cable solver policy, recording and
  observer plans, intracellular modes, and extracellular modes
  (`zero`, `shared_current`, `scaled_shared_waveform`, `current_table`,
  `dense`).
- [ ] Before claiming P12 cleanup has no performance loss, re-run the relevant
  P11 hotpath/realistic benchmark slices for single-cable and double-cable,
  CPU/GPU where applicable, with fresh artifact directories and git metadata.
- [ ] Post-P11 runtime/benchmark backlog:
  continue only the deferred items tracked in
  `docs/architecture/p11_closeout_2026_07_12.md`. Main follow-ups are GPU
  double-cable Triton/tiled-Thomas policy thresholds, shared-waveform/scaled
  extracellular input lowering, adaptive time-chunk policy, GPU dispatch
  scheduling, model/compiler optimizer closeout, dense fallback decisions, and
  NRV validation only when numerical behavior changes.
- [ ] Evaluate targeted GPU kernels for remaining non-solver device-side
  bottlenecks, without turning the whole host/runtime path into Triton:
  first prototype an `extracellular_scaled_shared_waveform` path that writes
  forcing directly in the solver layout, then prototype observer-only
  VmRaster/probe packing that extracts or aggregates on GPU without CPU
  round-trips. Keep this behind the JAX GPU runtime boundary and accept it only
  with before/after stage benchmarks showing that the cost is device-side and
  not just Python/JIT/transfer overhead.
- [ ] After the runtime contract, benchmark surface, and hot-path cleanup are
  stable, revisit cold-run optimization separately. Focus on JIT/lowering,
  membrane/runtime preparation caches, pool rebuild costs, and optional
  persistent compilation caches; do not mix cold-start policy decisions into
  the current hot-path cleanup.
- [ ] Continue hardening NRV integration only where the package contract is
  stable: keep geometry construction in `examples/with_nrv` or benchmarks, and
  promote future pieces only when they do not duplicate the canonical
  sampled-footprint path already in `axonscope.integrations.nrv`.
- [ ] Studies: callable threshold curves, block-threshold curves, recruitment
  curves, conduction validation, parameter sweeps, reuse policies, retention
  policies, and study results.
- [ ] Serialization: final schemas, typed serialization, and persistence
  strategy.
- [ ] Work on HPC integration.
- [ ] Work on FEM footprint integration, see
  `ideas/fem_axon_gpu_coupling_design.md`. Start with the CPU/NRV path before
  thinking GPU FEM: split benchmarks into FEM solve, first footprint, cached
  footprint sampling, and AxonScope solve; cache reusable FEM field bases;
  avoid repeated point-location by introducing an axon embedding/projection
  representation; then choose between full precomputed footprints, chunked
  projection, and future fused projection-solver paths by memory budget.

### P3 - Documentation And Examples

- [x] README rewritten after post-P7 stabilization.
- [x] Manual cleanup of `docs/`, `GUIDELINES.md`, and `AGENTS.md`.
- [x] Public examples audited after benchmark flattening.
- [ ] Write real notebook tutorials under `examples/tutorials/` following the
  indexed mini-course sequence.
- [ ] Add a didactic basic example for high-frequency block after block
  detection exists, so the example distinguishes propagation, activation
  failure, and true conduction block.
- [ ] Prepare proper Sphinx documentation.
- [ ] Do/update all public docstrings.

## Future Phases

### P8 - Future Bonus NumPy/SciPy Reference Solver Runtime

This is intentionally not the next implementation phase. The NumPy/SciPy
runtime remains valuable as a future reference/debug backend, but only after
the model/compiler surface and the current JAX runtime contract are clean
enough. The goal is a real reference solver runtime, not a JAX-backed
compatibility path.

- [ ] Keep `axs.runtime.numpy` reserved/non-executable until this phase reaches
  executable behavior through the same `AxonSimulation(...).run()`,
  `.estimate()`, and `.inspect()` lifecycle as JAX.
- [ ] Do not start implementation before P10 model/compiler cleanup and P11
  realistic JAX solver benchmarking/optimization are stable enough that the
  reference runtime has a clean contract to implement.
- [ ] Define the first supported scope explicitly: scalar/tiny simulations
  first, not population batching, GPU parity, or a second public workflow.
- [ ] Implement the reference solver behind the backend execution facade, using
  Model IR semantics and SciPy/NumPy numerical primitives rather than JAX
  membrane backends.
- [ ] Use the tridiagonal Crank-Nicholson solver path as the first numerical
  primitive for single-cable tiny simulations; choose SciPy banded/sparse
  helpers where they make the implementation clearer and deterministic.
- [ ] Decide and document the v1 model/input subset: single-cable first,
  intracellular current, sampled extracellular footprints, recording modes,
  observer support, and whether double-cable waits for a later slice.
- [ ] Add cross-backend validation against JAX on small deterministic cases:
  Vm traces, activation/block/latency observers, thresholds, probe recordings,
  retained membrane recordings, and model-step equivalence.
- [ ] Wire `ExecutionPolicy(runtime=axs.runtime.numpy)` only after executable
  behavior, examples, docs, estimates, inspection records, and tests exist.
- [ ] Document when to use the reference runtime: debugging tiny simulations,
  semantic validation, backend comparison, and numerical regression tests;
  document when not to use it.

## Completed Phase Summary

Detailed completed ledgers live in the archive and referenced architecture
docs. Keep only high-level state here.

- P0-P6: public API cleanup, one simulation workflow, protocols/results/views,
  examples-as-docs, inspection/runtime reports, validation policy, and
  backend/lowering cleanup are complete for the current JAX path.
- P7: class-based public membrane models, source compiler, generated JAX/NumPy
  model-step artifacts, generated-code cache/reporting, direct
  `JaxMembraneProgram` execution, and old membrane-stack deletion are complete.
- P9: cold-run micro baseline, scalar/batch span normalization, explicit
  hotpath chunk controls, and closeout decisions are recorded.
- P10: model/compiler cleanup and optimizer prep are complete enough for the
  current runtime work.
- P11: benchmark reset, JAX solver optimization, large-population Triton
  exploration, solver-engine flattening, single/double-cable cartography, and
  runtime cleanup closeout are complete for the current pass.

## Key References

- Architecture reference: `GUIDELINES.md`
- Agent guide: `AGENTS.md`
- Full pre-cleanup TODO archive:
  `docs/architecture/todo_archive_before_cleanup_2026_07_12.md`
- P11 closeout:
  `docs/architecture/p11_closeout_2026_07_12.md`
- Solver policy cleanup:
  `docs/architecture/p11_solver_policy_cleanup_decisions_2026_07_11.md`
- P12 runtime contract:
  `docs/architecture/p12_runtime_contract_2026_07_12.md`
- P12A runtime audit:
  `docs/architecture/p12a_jax_runtime_audit_2026_07_12.md`
- P12B runtime/JAX cleanup:
  `docs/architecture/p12b_runtime_jax_cleanup_2026_07_12.md`
- Benchmark surface map: `benchmark/README.md`
