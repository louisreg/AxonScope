# Results, Analysis, And Plot Architecture Audit

Status: P3 audit snapshot, updated on 2026-06-29.

This document maps the current result, analysis, protocol, observer, estimate,
inspection, benchmark, and plotting surfaces. It is an execution aid, not a
public API tutorial. `GUIDELINES.md` remains the architecture reference and
`todo.md` remains the active work queue.

## Target Boundary

AxonScope should keep five different concepts separate:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Raw simulation result | `AxonSimulationResult`, `AxonResultView`, retained recordings, solver observations, axes, diagnostics, final states. | Scientific interpretation, protocol decisions, rich rendering. |
| Observer output | Compact solver-side observations such as `VmRasterResult`. | Dense Vm impersonation or analysis status semantics. |
| Analysis result | Values, statuses, units, events, missing-input diagnostics, population denominators. | Solver routing, protocol iteration history, raw recording storage. |
| Protocol result | Sweep/search/recruitment histories and summaries over one or more simulation runs. | Raw solver arrays or backend diagnostics. |
| Diagnostic report | Estimate, inspection, benchmark, and profiling records. | Scientific result semantics. |

Rendering must stay in view modules:

- `results.views` for raw result values, Vm plots, recorded-axis plots, and
  `VmRasterResult` summaries.
- `analysis.views` for analysis/result/report rows, dataframe export, text, and
  scalar plots.
- `protocols.views` for threshold, recruitment, and sweep result views.
- `inspection_views` for inspection text/rich/plot rendering.
- `performance_views` owns estimate text/rich rendering.
- Benchmark views can stay benchmark-owned, but should follow the same
  data-first pattern.

## Surface Inventory

| Surface | Current role | Used by | Current mismatch | Decision |
| --- | --- | --- | --- | --- |
| `AxonSimulationResult` | Canonical public run result for one or many rows. Aggregates raw recordings, observations, diagnostics, axes, final states. | `AxonSimulation.run()`, examples, analyses, protocols. | It has raw-result helpers and `plot_traces()`, but no generic row/table export. That is acceptable for now because raw Vm arrays are not a compact report. | Keep. Do not make it an analysis/protocol report. Add only raw-result views that are genuinely generic. |
| `AxonResultView` | One-row view into `AxonSimulationResult`. | Iteration/indexing, examples, analysis definitions. | Owns ergonomic plot/analyze/report methods through `SingleAxonResultMixin`; good pattern, but can grow too wide if every plot lands here. | Keep as one-row facade. New methods only when they are common raw-result operations. |
| `RecordingManifest`, `RecordedSignal`, `RecordedAxis` | Describes available recordings and maps Vm columns to intrinsic positions/original indices. | Results, analyses, examples. | Strong and useful; no obvious duplicate. | Keep. Treat `RecordedAxis` as the canonical spatial axis for retained Vm. |
| `_ResultBlock` | Internal dense cohort storage. | `results.pool` only. | Name is generic but private. It stores raw batches plus observations. | Keep private. Do not document as public result model. |
| `VmRasterResult` | Compact threshold-window observation stored under `observations["vm_raster"]`. Can also be derived from dense Vm through `from_result()`. | Observer-only runs, protocols, examples. | It is raw observer data, not an `AnalysisResult`. | Keep as raw observer result. It now follows the common `rows()` / `to_dataframe()` / `format()` / `print()` / `plot()` summary surface. |
| `AnalysisResult` | Per-analysis values, statuses, messages, events, units, row labels, input requirements. | Post-hoc definitions, online observers, and protocol-derived metrics. | Good contract. Protocol statuses remain separate, but protocol results can now project per-row metrics into `AnalysisResult`. | Keep. Use row labels when protocol/analysis alignment matters. |
| `AnalysisReport` | Bundle of `AnalysisResult` objects attached to a simulation result. | `result.report(...)`, examples. | Good concept. Plotting currently skips nonnumeric values silently except when none are plottable. | Keep. Improve plot/report views after row metadata decision. |
| `ActivationEvent` | Detailed activation/latency event payload. | Analysis definitions, threshold protocol history. | Protocol history depends on an analysis event object directly. That is reasonable but should be intentional. | Keep. Document event payload as analysis-owned. |
| `ThresholdSearchResult` | Single binary threshold search result with bounds and history. | `find_threshold`, examples/tests. | Uses separate `ThresholdStatus`, not `AnalysisStatus`; this is now documented as protocol outcome status rather than analysis validity. | Keep. Consider a shared status vocabulary bridge only if protocols start embedding richer per-row analysis diagnostics. |
| `ThresholdCurve` | Per-row thresholds over row labels. | Threshold-vs-parameter examples/tests. | Duplicates some sweep concepts but has threshold-specific bounds/status/history. The ambiguous result field `rows` was renamed to `row_labels` so `rows()` can mean table rows like every other summary. | Keep as protocol result. It exposes `to_analysis_result()` for per-row threshold metrics. |
| `RecruitmentCurve` | Activation matrix over amplitudes. | Recruitment examples/tests. | `activated` is raw boolean matrix with domain-specific `count/fraction`. Good. First sampled activation amplitude is a derived metric, not a true bisection threshold. | Keep. `first_activation_uA` and `to_analysis_result()` expose the per-row metric without pretending it is interpolated. |
| `PoolSweepResult` | Generic `(value, row)` observation matrix. | Generic protocol sweep. | Observations are `object`/arbitrary; plotting only works for scalar numeric. Good fallback but not enough semantic metadata. | Keep as generic substrate. Add optional observation name/unit/status later if used beyond simple sweeps. |
| `SimulationInspection` and child records | Structured pipeline dry-run report. | `AxonSimulation.inspect()`, runtime examples/tests. | Already split from `inspection_views`. Good target pattern. | Keep. Use as reference for estimate split. |
| `SimulationEstimate` and groups/items | Memory/workload estimate. | `AxonSimulation.estimate()`, runtime examples/tests. | No default `plot()`, because there is no single obvious plot. | Done: text/Rich rendering and item/group table rows live in `performance_views.py`; records/data stay in `performance.py`. |
| `BenchmarkReport` / `SolverBenchmarkResult` | Benchmark/profiling outputs. | Benchmark package/scripts. | Benchmark-owned reports use `to_dict()`/`format()` but not the public view contract. Mostly acceptable because benchmark is its own domain. | Keep benchmark-owned for now. Revisit only when benchmark reports become public docs/API. |
| `results.views` | Raw result value extraction and generic plots. | Result mixins, examples. | Now includes VmRaster rendering too, and shared plotting decoration lives in `plotting.py`. | Keep. Add only raw-result plots here. |
| `analysis.views` | Analysis rows, dataframe, text, scalar plots, and spike raster plots derived from `analysis.rasterize`. | `AnalysisResult`, `AnalysisReport`, examples/tests. | Coherent and backed by shared row/dataframe/unit/plotting helpers. Does not yet include richer activation event plots. | Keep. Add event plots here only if they are analysis concepts, not raw Vm concepts. |
| `protocols.views` | Protocol rows, dataframe, text, plots. | Protocol result containers. | Coherent and now backed by shared row/dataframe/unit/plotting helpers. | Keep domain-specific row construction here. |
| `inspection_views` | Inspection rendering. | `SimulationInspection`. | Good separation. | Keep as reference architecture. |
| `results.visualization` | Removed legacy spike raster plot helper and `rasterplot` alias. | No active code should import it. | It used to be inconsistent with `results.views`. | Done: spike raster plotting lives in `analysis.views.plot_spike_raster(...)`. |

## Mismatches And Redundant Paths

1. `results.visualization` was the clearest redundant plot path. It has been
   removed; spike raster plotting now lives in
   `analysis.views.plot_spike_raster(...)`.

2. Estimate rendering now follows inspection rendering. `SimulationEstimate`
   delegates text/Rich output and item/group table rows to
   `performance_views.py`.

3. VmRaster decoding now has shared result helpers. Protocols use
   `activation_values_from_vm_raster(...)` instead of private bit walking.

4. Protocol statuses and analysis statuses are separate by design. Protocol
   statuses describe search-range outcomes while analysis statuses describe row
   validity; `docs/results_recording_analysis.md` now states this explicitly.
   Protocol results can project resolved per-row metrics into `AnalysisResult`
   with explicit status mapping.

5. Plot method names are categorized by object kind. Raw Vm uses
   `plot_trace`, `plot_traces`, `plot_map`; protocol/analysis summaries use
   generic `plot`; inspection may expose `plot` plus `plot_details`. Shared
   axis creation and decoration now live in `plotting.py`:
   - raw result plots are named by signal/concept;
   - summarized result objects expose `plot()`;
   - multi-panel diagnostics may expose `plot_details()`.

6. Data export is now explicit by object kind. Summary-like objects expose
   `rows()` and `to_dataframe()` over the same row data. Raw simulation result
   does not, because raw arrays are exposed directly through `signal()` and
   per-row recordings. Simulation estimates expose `rows(section="items")` and
   `rows(section="groups")` rather than one ambiguous table.

## Proposed P3 Work Order

1. Remove the `results.visualization` island. Done.
   - Spike raster plotting moved to `analysis.views.plot_spike_raster(...)`.
   - `rasterplot` alias removed.
   - Docs/tests/public facade updated.

2. Split estimate views. Done.
   - `performance_views.py` owns text/Rich rendering.
   - `SimulationEstimate.format()` and `.print()` are thin delegators.

3. Centralize VmRaster interpretation. Done for activation booleans.
   - `VmRasterResult.definition_index(...)` and `.any_active(...)` expose raw
     named-raster queries.
   - `activation_values_from_vm_raster(...)` decodes threshold-style activation
     values with blanking.
   - Protocols use the shared helper.
   - Keep `VmRasterResult` raw; do not convert it into dense Vm or an
     `AnalysisResult`.

4. Document the status split. Done.
   - `AnalysisStatus`: row/result validity.
   - `ThresholdStatus`: protocol range/search outcome.
   - Protocol result rows may include analysis events/statuses, but should not
     pretend they are analysis results.

5. Unify summary/table display. Done.
   - Added shared `summary_views` helpers for display values, unit labels,
     dataframe conversion, and text printing.
   - `AnalysisResult`, `AnalysisReport`, protocol summaries, `VmRasterResult`,
     and `SimulationEstimate` now expose row/dataframe/text surfaces through
     one convention.
   - `ThresholdCurve.row_labels` replaces the ambiguous result field `rows`;
     `ThresholdCurve.rows()` now returns table rows.

6. Bridge protocol metrics to analysis results. Done.
   - `AnalysisResult` carries `row_labels`.
   - `ThresholdSearchResult`, `ThresholdCurve`, and `RecruitmentCurve` expose
     `to_analysis_result()`.
   - `RecruitmentCurve.threshold_like_uA` was replaced by
     `first_activation_uA`.

7. Add architecture guardrails.
   - No new `results.visualization` module or imports.
   - Estimate rendering must not import Rich/matplotlib inside `performance.py`.
   - Protocol observer paths must use shared VmRaster decoders.

8. Unify plot plumbing. Done.
   - Added `plotting.py` with shared axis creation, unit label composition,
     grid/title/legend decoration.
   - Migrated `results.views`, `analysis.views`, and `protocols.views` onto
     the shared helper.
   - Kept local scientific visualizations in examples when they are genuinely
     example-specific.
   - Guardrail prevents public view modules from reintroducing direct
     `plt.subplots` setup instead of `ensure_axis(...)`.
