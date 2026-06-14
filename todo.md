# AxonScope TODO

Living checklist for AxonScope documentation, API cleanup, examples, benchmarks,
and GPU-readiness work.

Use this file as the step-by-step source of truth. At the start of each cleanup
session, read this file first. When a new mismatch is found, add it here. When a
task is done, check it only after code/docs/tests have been verified.

## First Task

- [ ] Audit `/docs` against the current code before writing Sphinx docs.
- [x] For each doc page, mark whether it is current, partially stale, proposal-only, or redundant.
- [x] Move confirmed mismatches into the sections below with concrete file paths.

## Master Product And Solver Direction

`GUIDELINES.md` is the project philosophy and strategic target for the next
large refactors. It defines this TODO's direction. Keep using `src/`, `tests/`,
and runnable examples as the truth for current behavior, but update
`GUIDELINES.md` when the product boundary, architecture philosophy, or target
object model changes.

- [x] Register `GUIDELINES.md` as the master architecture/philosophy direction in `agent.md`.
- [x] Convert the guideline roadmap into incremental implementation issues before starting the large object-model refactor.
- [ ] Keep the current pre-release policy: no backward-compatibility aliases for prototype APIs unless they are strictly temporary inside the repo.
- [ ] Do not run or require NRV validation for the next cleanup steps unless explicitly requested; keep using the fast unit suite for non-NRV work.
- [ ] Treat Sphinx setup as paused until the current API and architecture direction are stable enough to document.

### Phase 0 Guardrails Before Big Changes

- [x] Add architecture guardrail tests that prevent old and new public concepts from coexisting as permanent aliases.
  - [x] Guard root `GUIDELINES.md` as the project philosophy reference used by `agent.md` and `todo.md`.
  - [x] Guard against reintroducing removed top-level aliases such as `axs.analysis`, `axs.visualization`, and `axs.run_batch`.
  - [x] Guard against reintroducing removed public unit-suffix arguments on the stabilized public facade.
- [x] Add public API guardrails for “no raw string” closed-domain APIs where replacements already exist.
  - Current status: the initially tracked string-based public domains in this group now have typed replacements.
  - Done for activation target selectors: `ActivationCriterion.positions`/`indices` were replaced with typed `target=axs.positions.*`; the guardrail now rejects reintroducing those string parameters.
  - Done for recording selectors: `Recording.variables`/`Recording.spatial_mode` were replaced with typed `signals`/`spatial` inputs, `axs.signals.*`, `Signal`, and `RecordingSpatial`; the guardrail now rejects reintroducing those string parameters.
  - Done for formulation selectors: public axon constructors now use `axs.axons.CableFormulation`; the guardrail rejects raw formulation strings.
- [x] Add import-boundary guardrails so internal modules do not import top-level `axonscope` or visualization modules.
- [x] Inventory obsolete benchmark formats and exploratory scripts before deleting or migrating them.
  - Initial inventory on 2026-06-14: ignored generated caches exist under `benchmark/**/__pycache__/`, `examples/**/__pycache__/`, and `.DS_Store`; keep them out of source changes unless explicitly cleaning generated files.
  - Historical benchmark outputs live under `benchmark/results/` and `benchmark/reports/` as CSV/JSON/figures. Do not delete blindly; decide which outputs become dated baselines after the benchmark schema is finalized.
  - Current benchmark entry points are split between suite runners (`benchmark/runtime/run.py`, `benchmark/nrv_performance/run.py`) and standalone exploratory scripts/notebooks. Classify standalone scripts and `examples/benchmarks/*.ipynb` before the Phase 7 benchmark cleanup.
- [x] Record the latest non-NRV baseline after the next meaningful architecture guardrail pass.
  - Fresh run on 2026-06-14 after Phase 0 guardrails: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit` (`241 passed, 1 skipped`).

### Phase 1 Object Model Preparation

- [x] Draft the concrete migration from current `AxonSimulation` to target `AxonInstance` without keeping a compatibility alias.
  - [x] PR 1.1: rename `src/axonscope/axon_simulation.py` to `src/axonscope/axon_instance.py`.
  - [x] PR 1.1: rename class `AxonSimulation` to `AxonInstance` and update imports, examples, tests, docs, and changelog in the same change.
  - [x] PR 1.1: remove the old class/module/export instead of keeping a forwarding alias.
  - [x] PR 1.1: keep current behavior during the rename only; no extracellular footprint or result-model redesign in this PR.
- [x] Identify every current use of world placement (`x_offset`, `y`, `z`, point-source coordinates) and mark which pieces are transitional.
  - Transitional instance placement: `AxonInstance` initially keeps `x_offset`, `y`, `z`, `set_position(...)`, and internal `*_um` fields only as carry-over behavior from the current prototype.
  - Transitional analytical fields: `PointSourceElectrode(x, y, z, min_distance)` and analytical contexts remain helpers until Phase 2 rewrites them as footprint builders.
  - Solver/batch internals may continue using local transverse offsets while the public object model is being renamed.
- [x] Design the new root `AxonSimulation` as the executable definition: axons/population, stimulation, duration, dt, recording, and runtime options.
  - [x] PR 1.2: add a new root `AxonSimulation` object that owns `axons`, `duration`, `dt`, `recording`, and runtime/solver options.
  - [x] PR 1.2: delegate root `.run()` through the current `simulate(...)` and `simulate_pool(...)` execution paths.
  - [x] PR 1.2: avoid adding old-name aliases; examples teach the new root object directly.
- [x] Plan `AxonPopulation` so a single axon is the smallest population case, not a separate lifecycle.
  - [x] PR 1.3: add `AxonPopulation` as a typed collection of `AxonInstance` objects.
  - [x] PR 1.3: support one-instance populations and heterogeneous collections with the same execution lifecycle.
  - [x] PR 1.3: keep dispatch planning behind the population/simulation API rather than exposing separate scalar and pool workflows.
- [x] Rewrite affected examples and tests in the same change as the object-model rename.

### Guideline Roadmap Issue Queue

- [x] Phase 0 issue: add guardrails for guidelines reference, public compatibility aliases, public signatures, import boundaries, raw-string inventory, and non-NRV baseline.
- [x] Phase 1 issue: direct `AxonSimulation` -> `AxonInstance` rename with no alias.
  - Fresh non-NRV run on 2026-06-14 after the `AxonInstance` rename: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit` (`242 passed, 1 skipped`).
- [x] Phase 1 issue: add root executable `AxonSimulation`.
  - Added `examples/advanced/example_08_root_axon_simulation.py` as the required didactic advanced demo for this new concept.
  - Fresh non-NRV run on 2026-06-14 after adding the root `AxonSimulation`: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit` (`247 passed, 1 skipped`).
- [x] Phase 1 issue: add `AxonPopulation` and unify single/population lifecycle.
  - Added `examples/advanced/example_09_axon_population.py` as the required didactic advanced demo for this new concept.
  - Fresh non-NRV run on 2026-06-14 after adding `AxonPopulation`: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit` (`251 passed, 1 skipped`).
- [x] Phase 1 issue: keep constructor geometry directly inspectable on public axon models.
  - Added `axon.diameter` and `axon.diameter_values(unit=...)`; updated `examples/advanced/example_09_axon_population.py` to use the simple accessor.
  - Fresh non-NRV run on 2026-06-14 after adding public axon diameter accessors: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit` (`253 passed, 1 skipped`).
- [x] Phase 2 issue: add typed enums/selectors/signals/opaque identifiers, then remove remaining raw-string public domains.
  - [x] PR 2.1: replace public recording strings with typed `axs.signals.*`, `Signal`, and `RecordingSpatial`; update tests, docs, and examples.
  - [x] PR 2.1: add `examples/advanced/example_10_typed_recording_signals.py` as the required didactic advanced demo for this concept.
  - Fresh non-NRV run on 2026-06-14 after typed recording signals: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit` (`254 passed, 1 skipped`).
  - [x] PR 2.2: add typed position selectors for activation criteria.
  - [x] PR 2.2: add `examples/advanced/example_11_typed_position_selectors.py` as the required didactic advanced demo for this concept.
  - [x] PR 2.3: add `CableFormulation` for unmyelinated/myelinated formulation selection.
  - [x] PR 2.3: add `examples/advanced/example_12_cable_formulation.py` as the required didactic advanced demo for this concept.
  - [x] PR 2.4: add opaque `AxonId` and `DriveId` identifiers for typed public contracts.
  - [x] PR 2.4: fix `examples/advanced/example_11_typed_position_selectors.py` by removing the `PositionSelector.indices` storage/method collision.
- [x] Phase 2 issue: add `ExtracellularFootprint`, `ExtracellularDrive`, and `ExtracellularStimulation`.
  - [x] PR 2.5: add `ExtracellularFootprint`, `ExtracellularDrive`, `ExtracellularStimulation`, and explicit dense `ExtracellularPotential`.
  - [x] PR 2.5: require typed `axs.DriveId(...)` identifiers for extracellular drives.
  - [x] PR 2.5: add `examples/advanced/example_13_extracellular_footprint_drive.py` as the required didactic advanced demo for this concept.
  - Fresh unit run on 2026-06-14 after completing Phase 2: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit --tb=short` (`263 passed, 1 skipped`).
  - Fresh NRV run on 2026-06-14 after completing Phase 2: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/nrv --tb=short` (`116 passed, 516 warnings`).
- [x] Phase 2 issue: rewrite analytical point-source/context helpers as footprint builders.
  - [x] PR 2.5: add `AnalyticalExtracellularContext.build_footprint(...)` and `PointSourceElectrode.build_footprint(...)`.
  - [x] Keep the existing analytical context runtime path until Phase 3 planning/preparation moves solver lowering onto prepared footprints.
- [ ] Phase 3 issue: split planning and preparation, add signatures and reusable prepared cohorts.
- [ ] Phase 4 issue: isolate JAX runtime under backend modules and delete old dispatcher/solver paths once empty.
- [ ] Phase 5 issue: replace list-based pool results with canonical cohort-backed results and per-axon views.
- [ ] Phase 6 issue: move scientific analyses into a dedicated requirements/status/provenance layer.
- [ ] Phase 7 issue: finalize benchmark/performance story, footprint reuse, and memory estimates.
- [ ] Phase 8 issue: add callable studies, reuse policies, retention policies, and study results.
- [ ] Phase 9 issue: finalize serialization schemas and add NumPy reference backend validation.

### Phase 2 Contract Preparation

- [x] Design `ExtracellularFootprint` as a static spatial transfer object with intrinsic axon positions and units, not electrode CAD/world geometry.
- [x] Design `ExtracellularDrive` as footprint plus temporal `Stimulus`.
- [x] Design `ExtracellularStimulation` as the aggregate of one or more drives.
- [x] Plan migration of analytical point-source helpers into footprint builders outside the core solver dependency path.
- [x] Sketch typed selectors/signals/enums before replacing remaining string-based public domains.

### Later Target Architecture

- [ ] Split planning, preparation, execution, and backend lowering so JAX-specific code is isolated under backend runtime modules.
- [ ] Replace eager `list[SimResult]` pool results with a canonical cohort-backed result model and per-axon views.
- [ ] Move activation/velocity analysis into a dedicated analysis layer with applicability/status/provenance metadata.
- [ ] Add callable studies, reuse policies, retention policies, and final serialization only after the object model/result model settle.

## Initial Docs Audit Snapshot

Started from `agent.md` and a code/docs grep on 2026-06-13. This is a first
pass, not a completed audit.

| Page | Status | Notes / Next Action |
| --- | --- | --- |
| `docs/axon_model_organization.md` | partially current | Mostly matches the descriptive layer and unit-boundary direction. Re-check examples against `src/axonscope/axons/` before treating as Sphinx-ready. |
| `docs/solver_organization.md` | likely current | File list and time-grid behavior match current solver modules and `simulation_step_count`; keep as a candidate for Sphinx with light verification. |
| `docs/membranes.md` | mostly current | Built-in membrane namespace and unit normalization match `src/axonscope/membranes/`; still verify `Composite`, `SectionLayout`, and examples against tests. |
| `docs/stimulation.md` | mostly current | Known `HodgkinHuxley(length_um=..., diameter_um=...)` snippet was updated to `length=...` and `diameter=...`; still do a final full-page pass before Sphinx. |
| `docs/pool_dispatch.md` | mostly current | Public `simulate_pool`, dispatch diagnostics, and `build/print/plot_dispatch_plan` exist. Review for overlap with README and for any advanced batch API drift. |
| `docs/results_recording_analysis.md` | mostly current | Good conceptual split for `Recording`, `SimResult`, analysis, visualization. Future observer section now states that solver-side observers are not implemented. |
| `docs/recorders_observers_activation_strategy.md` | proposal | Implementation status refreshed on 2026-06-13: CPU/post-hoc activation and protocol sweeps exist; observer-only/GPU observer work remains future. |
| `docs/api_public_draft.md` | proposal-only | Clear proposal/roadmap warning added at the top. Later split implemented API from proposal if this document remains user-facing. |
| `docs/validation.md` | mostly current | Removed default GitHub Actions and stale NRV pass-count claims. No fresh NRV result is recorded yet. |

## Confirmed Mismatches From First Sweep

- [x] `README.md` lists `examples/basic/example_06_velocity_vs_diameter_batch.py`, but the current file is `examples/basic/example_06_velocity_vs_diameter.py`.
- [x] `tests/unit/test_examples.py` imports `examples.basic.example_06_velocity_vs_diameter_batch`; update to `examples.basic.example_06_velocity_vs_diameter`.
- [x] `examples/basic/example_06_velocity_vs_diameter.py` still shows the removed `_batch.py` filename in its module docstring run command.
- [x] `README.md` should list the new basic examples after the rename/additions: `example_06_velocity_vs_diameter.py`, `example_07_threshold_vs_diameter.py`, and `example_08_recruitment_curve_population.py` if they are intended to be part of the public learning path.
- [x] `docs/stimulation.md` should replace `length_um`/`diameter_um` axon constructor snippets with `length`/`diameter` quantity-based calls.
- [x] `docs/api_public_draft.md` still uses target snippets with `length_um`, `diameter_um`, `Recording.none()` as runnable behavior, and solver-side observers. Label these as proposal or move them to roadmap sections.
- [x] `docs/recorders_observers_activation_strategy.md` mentions `ActivationObserver`, `PeakVoltageObserver`, observer-only runs, amplitude-batched GPU sweeps, and `thresholds_for_pool`; current code has post-hoc `ActivationCriterion`, `detect_activation`, `find_activation_threshold`, `find_activation_threshold_curve`, `pool_sweep`, and `recruitment_sweep`.
- [x] `docs/results_recording_analysis.md` has future observer examples under `axs.results.analysis.*Observer`; keep clearly future or adjust to implemented post-hoc analysis only.
- [x] `docs/validation.md` says the default GitHub Actions workflow runs checks, but this checkout has no `.github/` directory.
- [x] `docs/validation.md` hard-codes `116 passed` for NRV validation; replace with a dated validation note only after rerunning in an NRV-ready environment.
- [x] `CHANGELOG.md` references absent paths/features including `axonscope.stimulation.evaluation`, `axonscope.solvers.stimulus_runtime`, `euler.py`, and a GitHub Actions workflow.

## Documentation Mismatches Already Found

- [x] Fix the example 06 rename everywhere: `README.md`, `examples/basic/example_06_velocity_vs_diameter.py`, and `tests/unit/test_examples.py`.
- [x] Update the README package map to point to `results/analysis.py` and `results/visualization.py`, or explicitly explain the top-level compatibility aliases.
- [x] Normalize constructor examples in docs to the implemented public names: `length`, `diameter`, `position`, `positions`, `sample_dt`, `duration`/`dt` for public wrappers, and `tsim`/`dt` for direct solver calls. Current docs are aligned outside the explicitly proposal-only `docs/api_public_draft.md`.
- [x] Add a clear warning at the top of `docs/api_public_draft.md` so stale target snippets are not mistaken for current runnable API.
- [ ] Later split `docs/api_public_draft.md` into implemented API versus proposal if it remains part of the user-facing docs.
- [x] Refresh `docs/recorders_observers_activation_strategy.md` implementation status with the current protocol functions and keep observer-only/GPU observer work marked future.
- [x] Audit `CHANGELOG.md` against files that actually exist in this checkout; remove or reword absent module names and CI claims.
- [x] Re-run `python -m pytest -q tests/unit` after doc/example/API cleanup fixes and record only fresh results. Fresh run on 2026-06-14 after typed recording signals: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit` (`254 passed, 1 skipped`).
- [ ] Re-run NRV validation only in an NRV-ready environment; record dated validation notes after a fresh run.
- [x] Remove duplicated narrative between README, `docs/pool_dispatch.md`, and `docs/results_recording_analysis.md` by making README a short entry point and keeping detailed contracts in `docs/`.

## Units And Public API

- [ ] Stabilize the public API before adding Sphinx or larger feature work.
- [ ] Prefer clean user-facing interfaces over backward compatibility shims; AxonScope is pre-release and not deployed as a stable dependency yet.
- [ ] Remove temporary compatibility aliases and old argument names once examples/tests/docs use the clean API.
- [x] Audit current compatibility aliases and decide which to remove before release: top-level `analysis`/`visualization`, time aliases, old constructor names, and any legacy convenience wrappers.
- [x] Keep constructor geometry easy to inspect on axon models: `axon.diameter` returns the nominal/uniform diameter in micrometers, and `axon.diameter_values(unit=...)` covers heterogeneous layouts.
- [ ] When removing or renaming public API, update affected examples in the same change and keep `tests/unit/test_examples.py` aligned.
- [ ] Pass units everywhere at public boundaries, including membranes, axons, stimulation, protocols, recording, and solver wrappers.
  - [x] Require unit-bearing current values for activation threshold bounds, tolerances, threshold-curve bounds, and recruitment amplitudes.
- [ ] Audit membrane constructors for plain-number ambiguity and decide where units should be required versus normalized.
  - [x] Rename public membrane template geometry arguments from `diameter_um` to `diameter` for `Tigerholm`, `Schild94`, and `Schild97`; require unit-bearing lengths while keeping internal `params["diameter_um"]`.
- [x] Audit stimulation constructors for plain-number ambiguity and decide where units should be required versus normalized.
  - [x] Require unit-bearing time inputs for public `Stimulus` starts, durations, sample grids, and shifts while keeping amplitudes generic until consumed by a clamp/electrode.
  - [x] Rename public intracellular clamp position inputs from `position_um` to `position`; require length units while keeping internal `position_um`.
  - [x] Require unit-bearing conductivity for explicit `AnalyticalExtracellularContext(sigma=...)`; omitting `sigma` keeps the ergonomic default of `0.3 S/m`.
  - [x] Replace `PointSourceElectrode` constructor coordinates with quantity-oriented `x`, `y`, `z`, and `min_distance`; remove public `x_um`/`y_um`/`z_um`, SI aliases, and plain-number coordinate interpretation.
- [x] Rename public recording temporal filter input from `sample_dt_ms` to `sample_dt`; require time units at construction while keeping internal `sample_dt_ms`.
- [x] Rename public recording spatial filter input from `positions_um` to `positions`; require length units while keeping internal `positions_um`.
- [x] Rename public simulation placement inputs from `x_offset_um`/`y_um`/`z_um` to `x_offset`/`y`/`z`; require length units while keeping internal `*_um` fields.
- [x] Add/update tests for unit-bearing membrane `diameter` parameters and public examples.
- [x] Make docs consistent about public names with units versus internal canonical suffixes.

### API Compatibility Audit

Started on 2026-06-13. Goal: remove pre-release compatibility shims before they
become accidental public contracts.

| Surface | Current State | Decision / Next Action | Affected Files |
| --- | --- | --- | --- |
| `axs.analysis` / `axs.visualization` | Top-level aliases duplicated `axs.results.analysis` and `axs.results.visualization`. | Done: remove top-level aliases and keep analysis/plotting under `axs.results.*`. | `src/axonscope/__init__.py`, `tests/unit/test_public_api.py` |
| `axs.run_batch(...)` | Public wrapper around `simulate_pool(...)`; used by README, `example_06`, and public API tests. | Done: removed from the public facade. Use `simulate_pool(...)` as the only pool wrapper. | `src/axonscope/simulation.py`, `src/axonscope/__init__.py`, `README.md`, `examples/basic/example_06_velocity_vs_diameter.py`, `tests/unit/test_public_api*.py` |
| Public wrapper time names | `simulate(...)` and `simulate_pool(...)` used to accept `duration_ms`/`dt_ms` aliases. | Done: public wrappers now use `duration`/`dt` with Pint quantities; internal names like `tsim_ms` stay below the public boundary. | `src/axonscope/simulation.py`, `README.md`, `docs/`, `examples/`, `tests/unit/test_public_api*.py`, `tests/unit/test_dispatcher.py` |
| Direct solver time aliases | Solver helpers used to accept `tsim`/`dt` and compatibility `duration_ms`/`dt_ms`. | Done: direct solvers use solver-level `tsim`/`dt` only. | `src/axonscope/solvers/common.py`, solver tests, NRV tests |
| Analysis threshold aliases | `rasterize`, `conduction_velocity`, and `average_velocity` used to accept `threshold`/`min_distance` aliases for older call sites. | Done: use explicit `threshold_mV`/`min_distance_ms` in post-hoc analysis helpers. Keep `threshold` on activation protocol objects where it is the domain term. | `src/axonscope/results/analysis.py`, README/docs/examples/tests using post-hoc analysis |
| `AxonInstance.intracellular_clamps` | Old alias of `intracellular_contexts`; runtime fallback and tests used it. | Done: removed alias and updated runtime/tests to use `intracellular_contexts`. Keep `add_current_clamp(...)` as an ergonomic stable shortcut. | `src/axonscope/axon_instance.py`, `src/axonscope/stimulation/runtime.py`, `tests/unit/test_public_api_facade.py`, `tests/unit/test_units.py` |
| `IntracellularCurrentClamp(position_um=...)` / `add_current_clamp(position_um=...)` | Public clamp placement exposed the internal micrometer suffix and accepted plain numbers. | Done: use `position` with required length units. Internal runtime state remains `position_um`. | `src/axonscope/stimulation/contexts.py`, `src/axonscope/axon_instance.py`, docs/examples/tests using intracellular clamps |
| `Recording(sample_dt_ms=...)` | Public constructor exposed an internal canonical-unit suffix. | Done: public input is `sample_dt` with required time units; normalized storage remains `sample_dt_ms` internally for solver/batch plumbing. | `src/axonscope/recording.py`, `docs/results_recording_analysis.md`, `docs/api_public_draft.md`, `tests/unit/test_units.py` |
| `Recording(positions_um=...)` | Public recording filter exposed the internal micrometer suffix and accepted plain numbers. | Done: public input is `positions` with required length units; normalized storage remains `positions_um`. | `src/axonscope/recording.py`, recording docs, tests/unit recording examples |
| `AnalyticalExtracellularContext(sigma=...)` | Explicit plain numbers were interpreted as S/m. | Done: explicit conductivity must carry units. Omitting `sigma` still defaults to `0.3 S/m`. | `src/axonscope/stimulation/contexts.py`, docs/examples/tests using analytical contexts |
| `PointSourceElectrode(x_um=..., x0_m=...)` | Constructor mixed public micrometer names, SI aliases, and plain-number coordinate interpretation. | Done: constructor uses `x`, `y`, `z`, and `min_distance` with required length units. Internal `x_um` fields and `x0_m` properties remain read-only canonical views. | `src/axonscope/stimulation/electrodes.py`, README/docs/examples/tests using point-source electrodes |
| `AxonInstance(..., y_um=...)` / `set_position(y_um=...)` | Public placement API exposed internal canonical suffixes and accepted plain numbers. | Done: use `x_offset`, `y`, and `z` with required length units. Internal `x_offset_um`, `y_um`, and `z_um` remain runtime/result fields. | `src/axonscope/axon_instance.py`, docs/examples/tests using positioned instances |
| Activation protocol currents | `find_activation_threshold`, `find_activation_threshold_curve`, and `recruitment_sweep` accepted plain current magnitudes as implicit microamperes. | Done: bounds, tolerances, callable bounds, vector bounds, and recruitment amplitudes must carry current units. Returned threshold fields still expose internal `*_uA` arrays plus quantity properties. | `src/axonscope/protocols/activation.py`, `tests/unit/test_protocols.py` |
| `Stimulus` explicit time inputs | Public waveform constructors accepted plain start/duration/sample/shift times as implicit milliseconds. | Done: public constructors require unit-bearing explicit times. Omitted `start` still means 0 ms, and amplitudes remain generic until a clamp/electrode consumes them. | `src/axonscope/stimulation/stimuli.py`, stimulation docs/examples/tests |
| `SimResult.Vm` | Convenience property/field used broadly in examples and tests. | Keep as a stable notebook-friendly convenience, not a temporary compatibility shim. Ensure errors stay clear when future observer-only runs do not carry Vm. | `src/axonscope/results/single.py`, recording/observer docs |
| `clear_extracellular_contexts(...)` | Plural name while the current instance stores one extracellular context internally. | Done: renamed to singular `clear_extracellular_context(...)`. The runtime tuple property remains `extracellular_contexts` for lower-level batch helpers. | `src/axonscope/axon_instance.py`, `tests/unit/test_public_api_facade.py` |

## Tutorials And Realistic Examples

- [ ] Write tutorials after the docs/code audit so they match the real API.
- [ ] Add realistic examples with NRV context/validation where appropriate.
- [ ] Keep basic examples didactic and compact.
- [ ] Keep advanced examples realistic enough to show actual stimulation studies and pool workflows.

## Results, Recording, And Observables

- [ ] Implement solver-side observables/observers, starting with peak voltage, then activation.
- [x] Decide and document whether `Recording` options are handled directly by solvers, translated before solver entry, or both.
- [ ] Verify single-axon and pool recording behavior for Vm, gates, currents, conductances, and state variables. Current docs state that pool observable groups are future work; keep this open until behavior and tests are complete across supported groups.
  - [x] Lock the current public `Recording` contract with tests: scalar runs require `Vm` and may include observable groups, pool runs support `Vm` spatial modes only, and unsupported position/temporal/pool-observable filters raise explicit errors.
- [ ] Keep post-hoc `ActivationCriterion` semantics aligned with future solver-side observers.

## Pool And Batch UX

- [x] Document and surface existing dispatch inspection helpers: `axs.dispatcher.build_dispatch_plan`, `print_dispatch_plan`, and `plot_dispatch_plan`.
- [ ] Add plotting helpers for batch groups and retained recording layouts.
- [x] Make batch diagnostics discoverable from `SimResult.diagnostics`.

## Benchmarks, CPU/GPU, And Bottlenecks

- [ ] Rework benchmark strategy; current benchmark story is not convincing enough --> see AXONSCOPE_BENCHMARKING_AGENT_SPEC.md
- [ ] Find a robust way to benchmark CPU versus GPU for representative workloads.
- [ ] Identify current bottlenecks and where the GPU path will likely hit memory, compilation, transfer, or batching limits. --> see AXONSCOPE_CPU_GPU_BOTTLENECK_ANALYSIS.md
- [ ] Separate correctness validation from performance benchmarking in docs and scripts.
- [ ] Record environment/device metadata for benchmark runs.

## Documentation Platform

- [x] Rewrite `README.md` from scratch as a short, current entry point.
- [ ] Provide extensive doctrings everywhere
- [ ] Set up Sphinx documentation after the `/docs` audit.
- [ ] Decide what belongs in Sphinx pages versus README versus examples.
- [x] Keep proposal/roadmap docs clearly labeled so users do not run future API snippets as current API.

## Cleanup

- [ ] Do a general cleanup pass after docs, examples, recordings, observers, and benchmarks are aligned.
- [ ] Remove stale aliases, removed file references, duplicate docs, and dead benchmark/example paths.
- [ ] Keep `agent.md` and this `todo.md` synchronized after each cleanup step.
