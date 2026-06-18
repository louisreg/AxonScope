# AxonScope TODO

Living operational roadmap for AxonScope documentation, API cleanup, examples,
benchmarks, solver/backend work, and study APIs.

Read this file at the start of cleanup/API/performance work. Keep it actionable,
chronological, and free of long benchmark prose. Detailed evidence belongs in
dedicated reports under `benchmark/reports/` or focused roadmap files under
`ideas/`.

## How To Use This File

- `GUIDELINES.md` is the master architecture and product-boundary reference.
- `agent.md` captures project working rules for future agents.
- `todo.md` is the current execution plan, not the full historical log.
- Source, tests, and runnable examples remain the truth for current behavior.
- AxonScope is pre-release: prefer clean breaking changes over compatibility
  shims, and delete superseded paths once replacements are in use.

## Current Snapshot

Updated on 2026-06-18 after closing the double-cable GPU solver optimization
campaign and cleaning the benchmark surface.

| Area | Status | Notes |
| --- | --- | --- |
| Phases 0-7.5 | Done | Guardrails, object model, typed contracts, JAX boundary, pool results, analysis layer, performance estimates, and solver-side observers are implemented for the current public layer. |
| Phase 7.6.1 | Done | Benchmark evidence matrix exists under `benchmark/hotpaths/`. |
| Phase 7.6.2 | Done | Memory-transfer and long-run cleanup landed for current hotpaths. |
| Phase 7.6.3 | Closed | Exact double-cable GPU solver optimization pass is complete. No new public solver route; see `benchmark/reports/double_cable_solver_optimization_2026_06.md`. |
| Phase 7.6.4 | Standby | Pseudo-double/pseudo-MRG remains validation-only under `benchmark/pseudo_double/`; not public, not `auto`. |
| Phase 7.6.5 | Next | `Vext` materialization and realistic workflow performance. |
| Phase 7.7 | Next | Stimulation and placement API cleanup against `GUIDELINES.md`. |
| Phase 7.8 | Later | Examples learning-path cleanup after API and Vext work. |
| Phase 8 | Later | Callable studies, reuse policies, retention policies, and study results. |
| Phase 9 | Later | Serialization schemas and reference backend validation. |

Current solver surface:

- `BatchOptions.double_cable_block_solver` accepts exactly `auto`, `thomas`,
  `pcr`, `pcr_soa`, and `pcr_adaptive`.
- `auto` resolves on CPU/default backends to `thomas`; GPU-like backends use
  `pcr_adaptive`.
- `pcr_adaptive` uses `pcr_soa` for `B <= 4096`, then matrix-layout `pcr`.
- Pallas, Triton, JAX-Triton, CUDA FFI, split iterative, associative-transfer,
  and pseudo-double candidates are archived/standby evidence, not active solver
  routes.

## Immediate Queue

Work should start here unless the user asks otherwise.

- [x] Close Phase 7.6.3 solver optimization campaign.
- [x] Clean active solver package and move non-retained custom-kernel tests/code
  to benchmark/archive locations.
- [x] Add a clean solver-campaign summary report with a small speedup plot.
- [x] Add workflow-level benchmark based on basic examples 06/07/08:
  `benchmark/realistic_examples/bench_basic_examples.py`.
- [ ] Run `benchmark/realistic_examples/bench_basic_examples.py` on CPU and GPU
  for the bounded `standard` matrix, then add the resulting CSV/JSON paths to
  the solver/Vext report.
- [ ] Phase 7.6.5: profile and optimize `Vext` materialization for realistic
  threshold, activation, recruitment, and conduction workflows.
- [ ] Phase 7.7: clean stimulation and placement APIs after the first Vext pass.

## Phase 7.6.5 Vext Plan

Goal: reduce complete workflow time now that solver-only custom-kernel work is
closed. The working hypothesis from E2E benchmarks is that dense `Vext`
materialization, transfer, and repeated input construction dominate many
realistic GPU cases.

1. Baseline realistic workflows.
   - Run example 06 velocity, example 07 threshold, and example 08 recruitment
     with `benchmark/realistic_examples/bench_basic_examples.py`.
   - Compare CPU vs GPU by workflow, fiber type, run count, and population size.
   - Record build time, first run, warm run, backend, and devices.

2. Add `Vext` timing visibility.
   - Separate public object construction, extracellular footprint evaluation,
     dense `Vext` array materialization, host-to-device movement, solver time,
     and result packaging.
   - Keep measurements available in CSV/JSON, not only profiler traces.

3. Reduce avoidable dense inputs.
   - Preserve the current public API while testing internal representations for
     shared point-source/electrode drives.
   - Avoid materializing dense zero `Iinj`.
   - Reuse or cache `Vext` when protocols sweep only current amplitude.
   - Explore on-device/lazy `Vext` generation for analytical point sources.

4. Validate behavior.
   - Re-run unit tests for stimulation, dispatcher, protocols, and solvers.
   - Re-run relevant NRV comparisons if `Vext` semantics change.
   - Keep `pcr_adaptive` as the GPU solver baseline during Vext work.

5. Decide next branch.
   - If `Vext` dominates after easy wins, continue with representation/API work.
   - If solver time becomes dominant again, reopen custom kernels only with a
     clear validation gate and a target device that supports the required stack.

## Solver Campaign References

- Summary report: `benchmark/reports/double_cable_solver_optimization_2026_06.md`
- Plot: `benchmark/reports/double_cable_solver_optimization_2026_06_speedups.svg`
- Active solver README: `benchmark/solvers/README.md`
- Kaggle runner README: `benchmark/kaggle/README.md`
- Solver roadmap archive: `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`

Archived experiment locations:

- `benchmark/archived_solver_spikes/`
- `benchmark/triton_solver/`
- `benchmark/jax_triton_solver/`
- `benchmark/cuda_ffi_solver/`
- `tests/archive/solver_spikes/`

## Phase 7.7 Stimulation And Placement API Cleanup

Goal: make the public API match the product boundary before Phase 8 studies.

- [ ] Re-read `GUIDELINES.md` before implementation.
- [ ] Audit public stimulation/context API names after Vext work clarifies the
  internal representation.
- [ ] Keep user-facing examples simple: clamps, point-source electrodes,
  extracellular drives, footprints, stimulation protocols, and populations.
- [ ] Avoid exposing solver/backend implementation details in public examples.

## Phase 7.8 Examples Learning Path

- [ ] Update basic examples after Vext/API cleanup.
- [ ] Add a solver-options example only for retained public options:
  `auto`, `thomas`, `pcr`, `pcr_soa`, `pcr_adaptive`.
- [ ] Do not add pseudo-double or custom-kernel examples unless a candidate
  leaves standby and becomes public.

## Phase 8 Studies

- [ ] Add callable study objects for threshold curves, recruitment curves,
  conduction validation, and parameter sweeps.
- [ ] Define reuse policies for prepared populations and stimulation contexts.
- [ ] Define retention policies for recordings and derived analysis outputs.

## Phase 9 Serialization And Reference Backend

- [ ] Finalize serialization schemas for public objects.
- [ ] Add NumPy/reference backend validation where it improves trust in JAX
  lowering or custom kernels.

## Recent Verification

- 2026-06-15: non-NRV unit run after dispatch cleanup: `314 passed, 1 skipped`.
- 2026-06-18: solver optimization campaign closed; active solver surface cleaned.

Update this section only with high-signal final checks, not every exploratory
benchmark run.
