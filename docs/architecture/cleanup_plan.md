# AxonScope Architecture Cleanup Plan

Status on 2026-07-03: historical cleanup plan. Many P0-P5 items described
below are now complete or superseded. Use `README.md`, `GUIDELINES.md`,
`todo.md`, `benchmark/README.md`, and the active examples for current behavior.

This document is retained as an audit trail for the post-P7 cleanup. It is not
the active checklist. `GUIDELINES.md` remains the target architecture reference,
while `src/`, `tests/`, examples, and fresh benchmark reports remain the source
of truth for current behavior.

The goal is to converge the repository toward:

- one public concept per user-facing idea;
- one canonical public execution path;
- one public simulation result model;
- clear backend boundaries;
- explicit archive boundaries for experiments and historical benchmark paths.

`todo.md` should eventually become a short active queue that points here for the
larger cleanup plan.

---

## 1. Current Implementation Map

### 1.1 Public simulation path

The current user-facing execution path is now singular:

```text
AxonSimulation.run(...)
        |
        v
src/axonscope/simulation.py
        |
        v
axonscope.runtime.execution.execution_context(...)
        |
        v
scalar: CrankNicholson.solve(...)
pool:   dispatcher.execution.run_pool(...)
        |
        v
runtime/jax/scalar_runner.py
runtime/jax/group_runner.py
        |
        v
internal results -> AxonSimulationResult
```

The public API generally returns `AxonSimulationResult`. Internal bridge types
still exist and are expected today:

- `SolverOutput`
- `DispatchRowRecord`
- `DispatchCohortRecord`
- `_ResultBlock`

These internal types are not necessarily a problem, but documentation should not
claim that the whole codebase has only one result model. The correct claim is:
one canonical public result model, with internal execution result blocks.

### 1.2 Backend boundary reality

Current public simulation, estimate, and inspection entry points route concrete
backend details through `axonscope.runtime.execution`. The historical notes
below explain why that boundary was added.

The public simulation layer mostly respects the target boundary:

- `simulation.py` enters through `axonscope.runtime.execution`.
- backend-heavy JAX runtime and kernels live under `runtime/jax`.
- timebase and public recording concepts are mostly backend-neutral.

The dispatcher group-runner exception has been removed: dispatch planning now
delegates concrete batch execution through `axonscope.runtime.execution`.
The later P4 cleanup also moved estimate/inspection lowering summaries behind
the same runtime boundary.

### 1.3 Extracellular stimulation reality

The public target path is:

```text
ExtracellularFootprint
ExtracellularDrive
ExtracellularStimulation
axs.analytical.point_source_footprint(...)
axs.analytical.point_source_drive(...)
axs.analytical.point_source_stimulation(...)
```

The old solver-facing contract has been removed from active code:

```text
Electrode
AnalyticalElectrode
ExtracellularContext
AnalyticalExtracellularContext
ExtracellularStimulationContext
NRVExtracellularContext
```

`add_extracellular_stimulation(...)` now stores typed
`ExtracellularStimulation` directly, and backend lowering consumes stimulation
rows instead of context/electrode adapters.

Historical mentions of the old names are retained in this plan only as decision
and migration evidence; they are not supported public API.

---

## 2. Priority Principles

### P0 means correctness of project direction

P0 work fixes contradictions that make the next cleanup pass ambiguous:

- stale agent-guide file names;
- wrong version claims;
- docs that point to missing files;
- guardrails that encode obsolete names;
- public API claims that conflict with implementation truth.

### P1 means public surface convergence

P1 work decides which concepts users should see and which concepts are internal.
This includes extracellular contracts, batch options, result types, and backend
boundary language.

### P2 means removing or hiding legacy paths

P2 work deletes, privatises, or archives paths only after active tests, examples,
and docs have moved to the retained path.

### P3 means complexity reduction

P3 work splits large modules and clarifies execution internals without changing
public behavior.

### P4 means evidence and documentation hygiene

P4 work keeps benchmark evidence useful while preventing old experiments from
looking like supported runtime choices.

---

## 3. Decision Register

These decisions should be made explicitly before large edits.

### D1. Extracellular public contract

Decision: accepted and implemented.

`ExtracellularFootprint`, `ExtracellularDrive`, and
`ExtracellularStimulation` are the only public solver-facing extracellular
contract.

Consequences:

- `Electrode`, `AnalyticalElectrode`, `ExtracellularContext`,
  `AnalyticalExtracellularContext`, `ExtracellularStimulationContext`, and
  `NRVExtracellularContext` are not exported by `axonscope` or
  `axonscope.stimulation`.
- `add_extracellular_context(...)` is removed from active instance APIs.
- tests and examples construct sampled footprints/drives/stimulation.
- backend input lowering consumes typed stimulation directly.

Residual risk:

- none: AxonScope has one active user and prefers direct clean breaks over
  transition paths.

### D2. NRV handoff contract

Decision: accepted and implemented for public surface.

`axonscope.integrations.nrv` is the official NRV bridge. It should convert NRV
fiber populations into intrinsic AxonScope populations and sampled electrode
footprints. AxonScope should not expose an `NRVExtracellularContext` placeholder
as a supported runtime path.

Consequences:

- remove or hide `NRVExtracellularContext` until there is a concrete executable
  implementation;
- keep NRV geometry and electrode ownership outside core AxonScope objects;
- keep `examples/with_nrv/` as the learning surface for NRV-owned geometry.

### D3. BatchOptions and BatchRecording visibility

Recommended decision:

Treat `Recording` and `RecordingPlan` as the public recording language. Treat
`BatchOptions` and `BatchRecording` as advanced/internal solver-lowering
objects unless a clear user-facing need remains.

Consequences:

- remove root-level exports if tests/examples no longer require them;
- keep protocols accepting public `Recording` where possible;
- route lowering through backend or preparation facades.

### D4. Dispatcher to backend ownership

Decision: Option B.

The dispatcher should plan and orchestrate, but concrete backend execution
should route through `axonscope.runtime.execution`. `dispatcher` should not
import `runtime.jax.group_runner` directly.

Alternatives considered:

Option A, current shortcut:

- `dispatcher` may call the JAX group runner because JAX is the only executable
  backend today.
- Short-term simple, but makes `dispatcher` a JAX-aware execution layer.

Option B, stricter target boundary:

- `dispatcher` owns planning, progress, grouping, and dispatch records;
- `backends.execution` owns concrete backend group execution;
- `dispatcher` no longer imports `runtime.jax`.

Implications:

- easier future `axs.runtime.numpy`/SciPy backend insertion;
- cleaner dependency boundary for tests and guardrails;
- slightly more boilerplate now, but less backend leakage long term.

### D5. axs.runtime.numpy and mixed precision

Recommended decision:

Keep reserved public values only if docs clearly say they are not executable
for current solves. Otherwise, remove or hide them until the implementation
exists.

Consequences:

- avoid estimate-only public claims;
- update docs and errors to distinguish "reserved" from "implemented".

### D6. Experimental shape bucketing

Recommended decision:

Keep experimental double-cable shape bucketing as opt-in evidence only, or move
it to benchmark/experimental if it has no near-term product path.

Consequences:

- active runtime code stays easier to reason about;
- benchmark-only paths do not silently become supported runtime complexity.

---

## 4. Workstreams

## P0 - Consistency and Guardrails

Purpose: remove contradictions before deeper cleanup.

Tasks:

- Replace stale agent-guide references with `AGENTS.md` in:
  - `AGENTS.md`
  - `GUIDELINES.md`
  - `todo.md`
  - `tests/unit/test_architecture_guardrails.py`
- Align version statements:
  - Python requirement in docs versus `pyproject.toml`
  - JAX requirement in docs versus `pyproject.toml`
- Fix README and example docstrings that point to missing files, especially the
  NRV realistic fascicle example path.
- Update `AGENTS.md` module-purpose descriptions so they reflect current
  namespaces:
  - `analysis/` is real;
  - result types are public only at the final result layer;
  - stimulation no longer centers on public "contexts".
- Add or update guardrails for current public API names:
  - prevent public examples from importing backend internals;
  - prevent old placement/world-coordinate shortcuts from returning;
  - prevent archived solver choices from returning to public options;
  - prevent direct dispatcher -> JAX imports; route execution through the
    backend facade.

Validation:

- run architecture guardrails;
- run fast unit tests touched by docs/path checks;
- verify README/example paths with `rg` and example smoke tests where practical.

Exit criteria:

- docs no longer contradict the package metadata or current file names;
- guardrails encode the chosen current architecture, not obsolete names.

## P1 - Public Surface Convergence

Purpose: decide what belongs to users versus internals.

Tasks:

- Define the retained public stimulation names:
  - `Stimulus`
  - `ExtracellularFootprint`
  - `ExtracellularDrive`
  - `ExtracellularStimulation`
  - `axs.analytical` helpers
- Classify old stimulation names:
  - deleted from active public API;
  - historical mentions kept only in changelog/architecture notes.
- Decide root exports for:
  - `BatchOptions`
  - `BatchRecording`
  - old extracellular names are removed from root exports.
- Make result model language precise:
  - public model is `AxonSimulationResult`;
  - internal execution blocks are allowed and documented as internal.
- Decide whether `inspection.py` is public backend-neutral inspection or public
  JAX inspection. Route imports accordingly.

Validation:

- import tests for root `axonscope` public names;
- example import/smoke tests;
- documentation search for old names after each removal.

Exit criteria:

- public namespace tells one story;
- examples teach only retained concepts;
- old names are not accidentally advertised as current API.

## P1/P2 - Extracellular Contract Flattening

Purpose: remove the largest old/new architecture overlap.

Phase 1, inventory:

- complete: active call sites migrated away from:
  - `add_extracellular_context`
  - `ExtracellularContext`
  - `AnalyticalExtracellularContext`
  - `ExtracellularStimulationContext`
  - `Electrode`
  - `footprint_for_electrode`
  - `NRVExtracellularContext`
- remaining mentions are classified as:
  - negative public API guardrails;
  - historical docs/changelog;
  - architecture notes.

Phase 2, backend lowering target:

- complete: backend input lowering consumes `ExtracellularStimulation` objects
  directly;
- represent compatible static-footprint rows as the existing internal
  factorized path without exposing a public mode;
- preserve dense equivalence tests for all migrated cases.

Phase 3, tests and examples:

- complete: active tests use sampled footprints/drives/stimulation;
- complete: analytical helper tests stay under `axs.analytical`;
- complete: approximate double-cable validation scripts were removed from the
  active benchmark and test surfaces.

Phase 4, deletion or hiding:

- complete: public exports for old context/electrode classes removed;
- complete: `add_extracellular_context` removed from active instance APIs;
- complete: `NRVExtracellularContext` removed until a real implementation is
  scheduled.

Validation:

- unit tests for extracellular stimulation;
- dense/factorized Vext equivalence tests;
- NRV adapter tests that do not require NRV import where possible;
- focused examples using `axs.analytical` and `examples/with_nrv`.

Exit criteria:

- no public example uses context/electrode APIs;
- backend lowering no longer needs to reconstruct old context objects from new
  typed stimulation;
- old path is deleted, private, or archived with an explicit label.

## P2 - Runtime Boundary Cleanup

Purpose: make execution ownership visible and enforceable.

Tasks:

- Introduce a backend execution facade for batch groups. Done for the dispatcher
  batch-group execution path.
- Move `run_jax_batch_group` calls behind `backends.execution`. Done for
  `dispatcher/execution.py`.
- Keep dispatch plan construction backend-neutral.
- Add a guardrail preventing dispatcher modules from importing
  `axonscope.runtime.jax`. Done for `dispatcher/execution.py`.
- Move public inspection lowering through the same boundary or label it as JAX
  inspection.
- Ensure planning, performance estimates, and inspection do not import heavy JAX
  numerical helpers unless explicitly in backend modules.

Validation:

- architecture guardrails;
- focused pool dispatch tests;
- inspection tests;
- import-time smoke test for `import axonscope`.

Exit criteria:

- import graph matches the chosen boundary;
- future NumPy/reference backend work has a clear insertion point, or docs say
  JAX is the only execution backend for now.

## P2 - Runtime Policy and Estimate Honesty

Purpose: align public policy values with executable behavior.

Tasks:

- Audit `ExecutionPolicy`, `Runtime`, `Device`, and `PrecisionPolicy` docs
  against actual runtime behavior.
- Clearly mark `axs.runtime.numpy` as reserved if it remains non-executable.
- Clearly mark mixed precision as unsupported until implemented.
- Review performance estimates for factorized Vext:
  - avoid implying dense materialization when the retained execution path uses
    factorized footprints;
  - report dense-equivalent cost separately from actual materialized cost.

Validation:

- unit tests for policy validation and error messages;
- estimate tests for dense versus factorized Vext cases;
- docs search for overclaiming language.

Exit criteria:

- every public runtime/precision value is either executable or explicitly
  reserved;
- estimates match actual materialization choices.

## P2/P3 - Results and Recording Boundary

Purpose: keep the public result model stable while making internals honest.

Tasks:

- Document internal result blocks as implementation detail:
  - `SolverOutput`
  - `DispatchRowRecord`
  - `DispatchCohortRecord`
  - `_ResultBlock`
- Keep `AxonSimulationResult`, `AxonResultView`, `RecordedSignal`, and
  manifests as the public result vocabulary.
- Move any result conversion helpers that are still exposed in public modules to
  internal modules.
- Keep `Recording` and `RecordingPlan` public; lower to backend-specific batch
  recording internally.
- Decide whether `BatchOptions` can remain as an advanced execution knob or
  should be hidden behind `Recording` plus execution policy.

Validation:

- result API tests;
- scalar and pool result shape tests;
- examples using `.single`, indexing, `signal(...)`, observations, and reports.

Exit criteria:

- public docs no longer imply internal bridge classes are user concepts;
- scalar and pool APIs teach the same result model.

## P3 - Protocol Module Split

Purpose: reduce the size and conceptual load of `protocols/activation.py`.

Current responsibilities mixed in one module:

- threshold search result dataclasses;
- recruitment curve dataclasses;
- pool sweep results;
- threshold curves;
- single-axon threshold search;
- pool threshold curves;
- recruitment sweeps;
- generic pool sweep helpers;
- activation observer optimization path;
- VmRaster decoding;
- progress reporting.

Proposed split:

```text
protocols/activation/
    __init__.py
    criteria.py
    threshold.py
    threshold_curve.py
    recruitment.py
    pool_sweep.py
    observer_path.py
    progress.py
    results.py
```

Migration rules:

- preserve public imports from `axonscope.protocols` during the split only if
  they are part of the retained API;
- keep observer-only execution internal and analysis-oriented;
- do not expose backend observer knobs as public protocol controls.

Validation:

- protocol unit tests;
- recruitment and threshold examples;
- import guardrail tests for retained public protocol names.

Exit criteria:

- each protocol module has one obvious responsibility;
- activation observer fast path is testable without reading the whole protocol
  stack.

## P3 - Backend Module Decomposition

Purpose: make backend execution auditable without changing behavior.

Large modules to split after public-surface decisions:

- `runtime/jax/batch_kernels.py`
- `runtime/jax/common.py`
- `runtime/jax/group_runner.py`
- `inspection.py`

Suggested decomposition:

- keep numerical kernels separate from host-side assembly;
- keep input lowering separate from recording/observer lowering;
- keep cache keys and prepared-cohort cache logic in focused modules;
- keep result assembly separate from kernel launch;
- keep plotting/printing inspection views separate from planning records.

Validation:

- compile/import checks;
- focused batch tests;
- no behavior changes without before/after tests.

Exit criteria:

- no single module needs to explain planning, lowering, execution, caching, and
  result assembly at once.

## P4 - Benchmarks and Archives

Purpose: preserve evidence without confusing it with current architecture.

Current classification is tracked in `benchmark/registry.py` and summarized in
`benchmark/README.md`.

Paths to classify explicitly:

- `benchmark/hotpaths/`: active profiling entry point.
- `benchmark/runtime/`: active named runtime suites.
- `benchmark/nrv_performance/`: active AxonScope-vs-NRV and realistic
  fascicle suites.
- `benchmark/realistic_examples/`: active workflow-level public-example
  benchmarks.
- `benchmark/solvers/`: validation-only retained double-cable solver evidence.
- `benchmark/triton_solver/`: archive.
- `benchmark/jax_triton_solver/`: archive.
- `benchmark/cuda_ffi_solver/`: archive.
- `benchmark/cute_dsl/`: archive.
- `benchmark/archived_solver_spikes/`: archive.
- `tests/archive/solver_spikes/`: archive.
- `benchmark/results/`: ignored generated output, not documentation source.

Tasks:

- add README labels where missing:
  - active;
  - validation-only;
  - archive;
  - experimental;
  - generated output.
- remove archive paths from active TODO language;
- keep performance claims tied to fresh benchmark reports;
- avoid copying full repository snapshots under `benchmark/results/`.

Validation:

- `rg` for stale public solver options and old benchmark claims;
- docs review rather than numerical tests unless benchmark code changes.

Exit criteria:

- a contributor can tell which benchmark paths are current without reading
  historical context.

## P5 - Documentation and Learning Path

Purpose: make docs and examples teach the retained API.

Tasks:

- keep `todo.md` short:
  - current active queue;
  - next decisions;
  - links to this plan and benchmark reports.
- label proposal documents clearly, especially `docs/api_public_draft.md`.
- refresh docs after public-surface decisions:
  - `docs/stimulation.md`
  - `docs/results_recording_analysis.md`
  - `docs/pool_dispatch.md`
  - `docs/solver_organization.md`
  - `README.md`
  - `examples/README.md`
- write tutorials only after the API names settle.

Validation:

- documentation search for old names;
- example smoke tests;
- no user-facing snippets that import backend internals.

Exit criteria:

- docs teach the same public story as examples and `axonscope.__init__`.

---

## 5. Legacy Path Register

This table should be updated as cleanup proceeds.

| Path or concept | Current role | Proposed state | Blocking work |
| --- | --- | --- | --- |
| agent-guide references | fixed docs/test name | use `AGENTS.md` | done |
| `Electrode` | removed old stimulation primitive | deleted from active public API | done |
| `AnalyticalElectrode` | removed old analytical helper | replaced with `axs.analytical` helpers | done |
| `ExtracellularContext` | removed old solver-facing contract | direct lowering from typed stimulation | done |
| `AnalyticalExtracellularContext` | removed old analytical context | analytical behavior preserved through footprints | done |
| `ExtracellularStimulationContext` | removed adapter from new API to old context | deleted after lowering migration | done |
| `NRVExtracellularContext` | removed placeholder context | hide until real implementation exists | future NRV/FEM phase |
| `add_extracellular_context` | removed old instance API | use `add_extracellular_stimulation` | done |
| `BatchOptions` root export | public advanced runner knob | keep during transition | later `SolverTuning` review |
| `BatchRecording` root export | public advanced runner object | keep during transition | make internal once `Recording` covers needs |
| dispatcher direct JAX import | current batch execution shortcut | remove via backend execution facade | D4 accepted |
| inspection direct JAX lowering | public inspection shortcut | route through backend or label JAX-specific | inspection boundary decision |
| `axs.runtime.numpy` | reserved/non-executable runtime | dedicated future NumPy/SciPy solver phase | real implementation |
| mixed precision | reserved/non-executable policy | mark unsupported or implement | precision design |
| experimental shape bucketing | opt-in runtime experiment | keep experimental or archive | benchmark evidence |
| Triton/JAX-Triton/CUDA FFI/Cute DSL benchmarks | solver experiments | archive/label clearly | README/docs cleanup |
| `benchmark/results/` repo copies | generated output/noise | keep ignored, avoid using as source | local cleanup policy |

---

## 6. Current `todo.md` Shape

`todo.md` is now a short active queue:

- P0 consistency is complete.
- P1 extracellular and BatchOptions decisions are recorded.
- P2 active extracellular legacy removal is complete; no migration path is
  needed.
- P3/P4 remain the next structural cleanup work.

The detailed rationale stays in this file; `todo.md` should remain an execution
queue rather than a second architecture document.

---

## 7. Recommended Attack Order

### Slice 1 - Low-risk consistency pass

Do first because it clarifies the workspace and should not change runtime
behavior.

- stale agent-guide references -> `AGENTS.md`
- version statements
- missing README/example paths
- stale module-purpose wording
- guardrail wording

Expected tests:

- architecture guardrails;
- import smoke tests;
- no NRV or benchmark run required.

### Slice 2 - Public surface decision pass

Status: complete for extracellular and BatchOptions transition decisions.

- mark retained public names;
- classify old extracellular names;
- classify `BatchOptions`/`BatchRecording`;
- dispatcher boundary target remains open.

Expected output:

- updated `todo.md`;
- maybe updated `GUIDELINES.md`;
- old extracellular public paths removed.

### Slice 3 - Extracellular migration

Status: complete for active code paths.

- call sites migrated away from `add_extracellular_context`;
- backend lowering consumes direct typed stimulation;
- dense/factorized equivalence tests preserved;
- old exports deleted or hidden from active public API.

Expected tests:

- extracellular unit tests;
- scalar and pool stimulation tests;
- focused examples;
- NRV adapter tests where possible.

### Slice 4 - Boundary cleanup

Do after stimulation no longer depends on context/electrode adapters.

- route dispatcher and inspection through the chosen runtime boundary;
- update guardrails.

Expected tests:

- pool dispatch tests;
- inspection tests;
- architecture guardrails.

### Slice 5 - Module decomposition

Do after behavior is stable.

- split protocols;
- split backend host-side orchestration from kernels and lowering;
- split inspection records from views.

Expected tests:

- broad unit suite;
- compile/import checks;
- focused protocol and dispatcher tests.

### Slice 6 - Documentation flattening

Do last, after names are stable.

- refresh docs;
- rewrite `todo.md` as short queue;
- update examples learning path;
- keep benchmark archives labelled.

Expected tests:

- example smoke tests;
- docs search for old names.

---

## 8. Stop Conditions

Pause and re-evaluate if any of these occur:

- removing old extracellular contexts would require rewriting large backend
  numerical kernels rather than only input lowering;
- NRV validation depends on a context path that cannot be represented as sampled
  footprints;
- `BatchOptions` is still needed by public protocols in a way that `Recording`
  and `ExecutionPolicy` cannot express;
- a guardrail must be weakened to make cleanup pass;
- benchmark evidence contradicts the intended factorized/dense execution story.

---

## 9. Definition of Done

This cleanup campaign is complete when:

- `todo.md` is short and actionable;
- docs, examples, root exports, and guardrails use the same public vocabulary;
- typed extracellular stimulation is the only documented public path;
- old context/electrode objects are deleted, private, or explicitly archived;
- public execution returns `AxonSimulationResult` everywhere;
- internal result blocks are documented as internal implementation details;
- backend boundaries are either enforced or explicitly documented;
- benchmark experiments are labelled as active, validation-only, archive, or
  experimental;
- no public tutorial imports solver/backend internals.
