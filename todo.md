# AxonScope TODO

Living checklist for AxonScope documentation, API cleanup, examples, benchmarks,
and GPU-readiness work.

Use this file as the step-by-step source of truth. At the start of each cleanup
session, read this file first. When a new mismatch is found, add it here. When a
task is done, check it only after code/docs/tests have been verified.

## Current Status Before Phase 8

- [x] Phases 0-7.5 are complete in code/tests/docs notes for the current public layer.
- [x] Latest full unit validation after Phase 6: unit suite `300 passed, 1 skipped` on 2026-06-14.
- [x] Phase 6 completed with the real public `axs.analysis` package.
- [x] Phase 6 namespace decision: use `axonscope.analysis` / `axs.analysis` as a real package, not a forwarding alias to `axs.results.analysis`.
- [x] Phase 6 scalar-result decision for now: keep scalar `simulate(...) -> SimResult`; make the new analyses work on `SimResult`, `AxonResultView`, and `AxonSimulationResult`.
- [x] Phase 6.3 moved old post-hoc algorithms out of `results/analysis.py` / `results/activation.py` into the new analysis layer without permanent forwarding aliases.
- [x] Phase 6.4 added structured missing-input requirements with recording hints and required result fields.
- [x] Phase 6.5 added online Vm observer definitions for activation and peak voltage, with online/post-hoc cross-validation tests.
- [x] Phase 7 added simulation memory estimates, typed runtime/device/precision planning values, memory metadata in hotpath manifests, and the `footprint_reuse_sweep` workload.
- [x] Latest full unit validation after Phase 7: unit suite `306 passed, 1 skipped` on 2026-06-14.
- [x] Phase 7.5 added solver-side observers for `PeakVoltage` and `Activation`, including scalar kernels, homogeneous single-cable batch observer-only runs, and trace-free `Recording.none()` results.
- [x] Latest full unit validation after Phase 7.5: unit suite `308 passed, 1 skipped` on 2026-06-15.
- [x] Phase 7.6 evidence gate is complete: realistic mixed-population, hotpath-matrix, and long observer-only CPU/GPU traces have been run and analyzed.
- [x] First observer-only `results.split_batch` cleanup is implemented locally:
  batched solver-side observations now stay in one compact dispatch cohort
  instead of materializing one internal dispatch result per axon row.
- [x] Latest full unit validation after the compact dispatch cohort cleanup:
  unit suite `314 passed, 1 skipped` on 2026-06-15.
- [x] Compact dispatch cohort Colab validation is complete:
  `colab_cpu_gpu_kernel_observer_long_20260615_122541` reduced GPU
  `results.split_batch` at `n=1000` from `140.6 ms` to `0.76 ms`.
- [x] Pulse-vectorized sparse current-clamp Colab validation is complete:
  `colab_cpu_gpu_kernel_observer_long_20260615_132920` reduced GPU
  `inputs.intracellular` at `n=1000` from `150.2 ms` to `23.8 ms`.
- [ ] Next priority before broad Phase 8 APIs: run the new
  `double_cable_extracellular` Colab case, then inspect realistic
  extracellular-drive/footprint reuse pressure.
- [ ] Next implementation phase after the targeted hotpath cleanup: Phase 8, callable studies/reuse policies/retention policies.
- [ ] Keep current Phase 5-7.5 changes uncommitted until the user asks for a commit or the next checkpoint requires it.

## Phase 7.6 Priority Before Phase 8

This phase should improve evidence quality before adding callable studies and
reuse policies. The goal is to know which execution paths still stall on
realistic populations, not just clean homogeneous smoke workloads.

- [x] Add the first realistic population benchmark workload.
  - Added `realistic_mixed_population` with mixed HH/Rattay-Aberham models,
    mixed diameters, mixed compartment counts, intracellular clamps, and some
    analytical point-source extracellular rows.
  - Local smoke run on 2026-06-15:
    `benchmark/results/hotpaths/phase7_6_realistic_smoke/manifest.json`.
- [x] Add the first compact hotpath matrix workload.
  - Added `hotpath_matrix` with homogeneous center recording, homogeneous
    probes recording, observer-only `Recording.none()`, point-source
    extracellular center recording, and realistic mixed-population center
    recording.
  - Local smoke run on 2026-06-15:
    `benchmark/results/hotpaths/phase7_6_hotpath_matrix_smoke/manifest.json`.
- [x] Keep manifests explicit about model mix, diameter distribution,
  compartment distribution, stimulation coverage, recording policy, observers,
  result retention, and per-simulation memory estimates.
- [x] Analyze first Colab GPU scale trace for Phase 7.6.
  - Run analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_gpu_20260615_094639/`.
  - Colab reported `jax_default_backend="gpu"` and device `cuda:0`.
  - `kernel.wait` is not the bottleneck at `n=500`: `intracellular_only`
    `0.082 ms`, `point_source_extracellular` `0.098 ms`,
    `observer_only` `0.016 ms`, `realistic_mixed_population` `0.444 ms`,
    and `hotpath_matrix` `0.753 ms`.
  - Homogeneous `n=500` workloads are in the `108-160 ms` range. The realistic
    mixed population is `11.5 s`, and `hotpath_matrix` is `11.9 s` because it
    includes the same mixed-population path.
  - The stall is concentrated in `runtime.prepare`: `10.8 s` for
    `realistic_mixed_population_n500` and `10.5 s` for
    `hotpath_matrix_n500`.
  - The realistic workload splits into six `parameter-single-cable` groups with
    `geometry_shared=False`, `Nx` values `11/13/15`, and group sizes around
    `83-84`; each group spends roughly `1.7-2.2 s` in preparation.
  - Decision: do not expand to MRG/double-cable until this simpler
    `parameter-single-cable` preparation stall is understood and reduced.
- [x] Add and analyze Colab CPU/GPU comparison traces for the same committed revision.
  - Run GPU and forced-CPU JAX processes in the same Colab notebook so hardware
    and dependency drift are minimized.
  - Keep both results under one downloaded folder, with a generated comparison
    summary for `simulation.pool.total`, `runtime.prepare`, input
    materialization, kernel enqueue/wait, split, and public packaging.
  - Workflow implemented in `benchmark/hotpaths/colab_gpu_hotpaths.ipynb` and
    documented in `benchmark/hotpaths/COLAB.md`.
  - Run analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_20260615_095754/`.
  - Short `Nt=6` workloads are CPU-faster than GPU because setup dominates
    execution. At `realistic_mixed_population_n500`, GPU total is `11364.8 ms`
    versus CPU `3698.0 ms`; GPU `runtime.prepare` alone is `10480.7 ms`.
  - `kernel.wait` stays small on GPU (`0.56 ms` for
    `realistic_mixed_population_n500`), so this is still a host/preparation
    stall, not a device compute bottleneck.
- [x] Attack the first `runtime.prepare` bottleneck for heterogeneous
  `parameter-single-cable` groups before adding bigger workloads.
  - Profiled what is repeated per row/group during preparation.
  - Replaced row-by-row JAX cable runtime stacking with host NumPy array
    preparation followed by one batched JAX transfer.
  - Skipped scalar stimulation-callable compilation in batch preparation when
    batch kernels already receive materialized intracellular/extracellular
    inputs.
  - Cached compiled membrane descriptors and ICM backends for repeated
    model/options/Nx signatures.
  - Updated `realistic_mixed_population` to reuse identical axon templates for
    repeated model/diameter/Nx combinations, so the dispatcher cache measures
    realistic template reuse instead of artificial per-row re-instantiation.
  - Local `realistic_mixed_population_n500` after the first optimization pass:
    total `380.4 ms`, `dispatch.build_plan 16.7 ms`,
    `runtime.prepare 129.0 ms`, `kernel.enqueue 134.7 ms`.
  - Previous local post-cache/pre-template run was total `1459.8 ms`,
    `dispatch.build_plan 646.7 ms`, and `runtime.prepare 338.1 ms`.
- [x] Re-run Colab CPU/GPU comparison after the Phase 7.6 optimization pass.
  - Use the same CPU/GPU notebook flow and compare against
    `colab_cpu_gpu_20260615_095754`.
  - Run analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_20260615_102221/`.
  - The `realistic_mixed_population_n500` GPU path improved from
    `11364.8 ms` to `556.3 ms` (`20.4x` faster). CPU improved from
    `3698.0 ms` to `238.4 ms` (`15.5x` faster).
  - The optimized `hotpath_matrix_n500` GPU path improved from `11655.0 ms`
    to `1545.6 ms` (`7.5x` faster).
  - Current `realistic_mixed_population_n500` GPU profile:
    `runtime.prepare 220.3 ms`, `kernel.enqueue 246.6 ms`,
    `dispatch.build_plan 10.7 ms`, `kernel.wait 0.44 ms`.
  - Interpretation: the first preparation wall is gone, but the current
    `duration=0.30 ms`, `dt=0.05 ms` traces only have `Nt=6`; they are still
    dominated by setup/enqueue/packaging rather than useful solver work.
- [x] Add longer Colab CPU/GPU cases to the notebook protocol.
  - `setup_scale` keeps the current all-workload short trace for preparation
    regressions.
  - `kernel_observer_long` runs `observer_only` with `Recording.none()`,
    sizes `500/1000`, `duration=10 ms`, `dt=0.01 ms`, and `51` compartments
    to emphasize GPU kernel scaling while limiting retained output.
  - `kernel_realistic_long` runs `realistic_mixed_population`, size `500`,
    `duration=5 ms`, `dt=0.01 ms`, and `51` compartments to test realistic
    heterogeneity after the observer-only kernel probe.
- [x] Run and analyze the long Colab CPU/GPU cases before Phase 8.
  - Start with `CASE = "kernel_observer_long"` in
    `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`.
  - Then run `CASE = "kernel_realistic_long"` if the observer-only trace looks
    healthy.
  - `kernel_realistic_long` run analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_realistic_long_20260615_103306/`.
  - This is the first trace where GPU wins on the realistic mixed workload:
    `744.6 ms` GPU versus `1253.8 ms` CPU (`1.68x` total speedup).
  - The useful compute gap is much larger than the total speedup suggests:
    GPU `kernel.wait 33.0 ms` versus CPU `955.2 ms` (`29x`), while GPU still
    pays larger fixed/setup costs: `runtime.prepare 237.1 ms`,
    `kernel.enqueue 330.9 ms`, `inputs.intracellular 64.3 ms`, and
    `inputs.extracellular 38.2 ms`.
  - The dense input estimate is now significant: about `52.5 MiB` for
    `Iinj[B,Nt,Nx]` and `52.5 MiB` for `Vstim[B,Nt,Nx]` at `n=500`,
    `Nt=500`, and `Nx<=55`; drive materialization is now a real Phase 8/7.6
    design pressure, not just a theoretical memory note.
  - Investigated collapsing the six single-cable `Nx=51/53/55` groups into
    two padded model groups. Do not implement this shortcut yet: current
    padded kernels need row-specific recording selectors and observer masks to
    avoid forcing `Recording.full()` or selecting the wrong row center/probes.
  - Next required trace is still `kernel_observer_long`, because
    `Recording.none()`/observers should isolate solver-side observer scaling
    from dense retained-output behavior.
  - `kernel_observer_long` run analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_104356/`.
  - Observer-only GPU total speedups are stable and meaningful:
    `337.4 ms` GPU versus `1988.6 ms` CPU at `n=500` (`5.89x`), and
    `673.4 ms` GPU versus `3884.2 ms` CPU at `n=1000` (`5.77x`).
  - Observer-only retained Vm is correctly eliminated: `vm_shapes=[]`,
    observation names are `activation`/`peak_voltage`, and
    `retained_mib=0.0` for both `n=500` and `n=1000`.
  - The dominant remaining GPU costs at `n=1000` are
    `inputs.intracellular 296.3 ms`, `results.split_batch 170.6 ms`, and
    `kernel.enqueue 146.8 ms`; `kernel.wait` is effectively zero for the
    compact observer output path.
  - The memory estimate confirms the next pressure point:
    `Iinj[B,Nt,Nx]` alone is `194.6 MiB` at `n=1000`, `Nt=1000`, `Nx=51`,
    even though no Vm trace is retained.
  - Sparse current-clamp rerun analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_114221/`.
    - The sparse path is a real Colab GPU win, not only a local memory cleanup:
      at `n=1000`, GPU total improves from `673.4 ms` to `378.0 ms`, and CPU
      total improves from `3884.2 ms` to `3140.4 ms`.
    - `inputs.intracellular` drops from `296.3 ms` to `77.8 ms` on GPU and
      from `455.4 ms` to `75.4 ms` on CPU at `n=1000`.
    - CPU/GPU total speedup rises to `8.31x` at `n=1000` and `9.46x` at
      `n=500`.
    - The new dominant GPU costs at `n=1000` are `results.split_batch
      140.5 ms`, `kernel.enqueue 109.2 ms`, and `inputs.intracellular
      77.8 ms`.
    - Important follow-up found in the same events: even with
      `context_count=0`, the runner still materialized dense zero
      `Vstim[B,Nt,Nx]` (`204 MB` at `n=1000`). This is now a targeted cleanup
      item, distinct from real extracellular-drive compression.
  - Zero-field sparse rerun analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_120457/`.
    - The `zero_no_context` patch is confirmed on both GPU and forced CPU:
      `inputs.extracellular` is `0.028 ms` on GPU and `0.039 ms` on CPU at
      `n=1000`, with no `vstim_mid` metadata and skipped dense shape
      `[1000, 1000, 51]` (`204 MB`).
    - Relative to the sparse-but-still-zero-`Vstim` run, GPU total is essentially
      unchanged (`378.0 ms` -> `373.8 ms` at `n=1000`) because the dense
      zero-field build was mainly a memory problem on GPU.
    - CPU total improves more clearly (`3140.4 ms` -> `2630.4 ms` at `n=1000`).
    - End-to-end improvement versus the original dense observer run is now
      `673.4 ms` -> `373.8 ms` on GPU and `3884.2 ms` -> `2630.4 ms` on CPU
      at `n=1000`.
    - New dominant GPU costs at `n=1000`: `results.split_batch 140.6 ms`,
      `kernel.enqueue 101.9 ms`, `inputs.intracellular 80.9 ms`.
  - Compact dispatch cohort rerun analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_122541/`.
    - The compact dispatch result path closes the `results.split_batch`
      bottleneck: at `n=1000`, GPU `results.split_batch` drops from
      `140.6 ms` to `0.76 ms`, and CPU drops from `138.6 ms` to `0.43 ms`.
    - Public packaging also drops: GPU `results.to_public` at `n=1000` is
      `2.11 ms` instead of `10.26 ms`.
    - End-to-end GPU improves more modestly because another cost moved up:
      total `373.8 ms` -> `341.8 ms` at `n=1000`, while
      `inputs.intracellular` varied upward (`80.9 ms` -> `150.2 ms`) and
      `kernel.enqueue` rose (`101.9 ms` -> `115.5 ms`).
    - At `n=500`, the total gain is cleaner: GPU `193.8 ms` -> `157.8 ms`,
      with `results.split_batch` reduced from `38.4 ms` to `0.26 ms`.
    - New observer-only GPU bottleneck map at `n=1000`: `inputs.intracellular
      150.2 ms`, `kernel.enqueue 115.5 ms`, `runtime.prepare 49.2 ms`;
      `results.split_batch` is no longer the target.
  - Pulse-vectorized sparse current-clamp rerun analyzed on 2026-06-15:
    `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_132920/`.
    - The one-pulse sparse input fast path is a strong Colab GPU win:
      at `n=1000`, `inputs.intracellular` drops from `150.2 ms` to
      `23.8 ms`, and total GPU drops from `341.8 ms` to `176.8 ms`.
    - At `n=500`, total GPU drops from `157.8 ms` to `133.7 ms`, with
      `inputs.intracellular 16.4 ms`.
    - CPU/GPU total speedup is now `14.7x` at `n=1000`; the remaining GPU
      observer-only costs are mainly `kernel.enqueue 102.8 ms`,
      `runtime.prepare 28.9 ms`, `inputs.intracellular 23.8 ms`, and
      `dispatch.build_plan 14.9 ms`.

- [ ] Phase 7.6 targeted cleanup before broad Phase 8 APIs.
  - [x] Add a compact/factorized intracellular-drive path for simple current clamps
    so observer-only and study runs do not have to materialize dense
    `Iinj[B,Nt,Nx]` when the input is sparse in space and structured in time.
    - Implemented first for homogeneous single-cable batch observer-only runs:
      point current clamps lower to `sparse_current_clamp` arrays
      `(B, Nt, K)` plus `(B, K)` indices/masks and are scattered inside the
      solver observer scan.
    - Local validation benchmark on 2026-06-15:
      `benchmark/results/hotpaths/phase7_6_after_sparse_iinj_n1000/`.
      At `n=1000`, `Nt=1000`, `Nx=51`, estimated `Iinj` drops from the previous
      dense `194.6 MiB` shape to `3.8 MiB` for `density_mid` plus tiny
      indices/mask arrays; `retained_mib=0.0` remains true.
    - Local CPU timing note: `inputs.intracellular` improved to `98.2 ms`
      compared with the previous dense local trace around `239 ms`, while the
      sparse scatter shifts some cost into `kernel.enqueue`; verify on Colab GPU
      before treating this as a speed win rather than a memory win.
    - Targeted validation on 2026-06-15: compileall passed for
      `src/axonscope`, `examples/advanced`, and `tests/unit`; mypy passed on
      `batch_inputs.py`, `input_batches.py`, `batch_kernels.py`, and
      `performance.py`; targeted unit tests passed (`22 passed`), with an
      earlier broader batch/dispatcher/analysis/hotpath run at `47 passed`.
  - [x] Vectorize the common one-pulse sparse current-clamp builder.
    - Implemented a conservative fast path for rows that each contain exactly
      one point current clamp with a three-point hold/pulse stimulus.
    - The fast path evaluates the whole `(B, Nt, 1)` pulse-density tensor with
      NumPy broadcasting and caches repeated solver-axon geometry instead of
      calling `Stimulus.evaluate(...)` row by row.
    - Local smoke benchmark on 2026-06-15:
      `benchmark/results/hotpaths/local_observer_pulse_fastpath/`.
      `observer_only_n100` reports `inputs.intracellular 4.46 ms`; the closest
      prior local zero-field smoke was `9.02 ms`.
    - Targeted validation on 2026-06-15: compileall passed for the touched
      files; mypy passed for `input_batches.py` and the modified dispatch/
      result modules; sparse current-clamp tests passed (`2 passed`) and the
      targeted dispatcher/public observer/performance run passed (`24 passed`).
    - Evidence gate completed on 2026-06-15:
      `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_132920/`
      confirmed the input-materialization gain on GPU.
  - [x] Avoid dense zero `Vstim[B,Nt,Nx]` materialization for homogeneous
    single-cable observer-only runs with no extracellular contexts.
    - The JAX runner now records `input_format="zero_no_context"` and passes no
      `vstim_mid` array into the sparse observer kernel for this case.
    - Local smoke benchmark on 2026-06-15:
      `benchmark/results/hotpaths/phase7_6_after_zero_vstim_smoke/`.
      `inputs.extracellular` reports `0.036 ms` for `n=100`, and the event
      records the skipped dense shape `[100, 100, 51]`.
    - Targeted validation on 2026-06-15: mypy passed for `batch_kernels.py`,
      `group_runner.py`, and `performance.py`; targeted unit tests passed
      (`13 passed`).
    - Colab validation on 2026-06-15:
      `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_120457/`
      confirmed `input_format="zero_no_context"` and eliminated dense
      zero-field `vstim_mid` materialization.
  - [x] Reduce observer-only `results.split_batch` cost by keeping compact
    population observations batched from the solver backend to the public
    result layer.
    - First layer: solver-side observations stay attached as batched cohort
      observations instead of being eagerly sliced/re-merged into one
      `AnalysisResult` per axon row.
    - Second layer: homogeneous batch observer-only runs now return one compact
      internal dispatch cohort instead of one `DispatchResult` per axon row.
    - Local smoke benchmark on 2026-06-15:
      `benchmark/results/hotpaths/local_observer_cohort_dispatch/`.
      `observer_only_n5` reports `results.split_batch 0.096 ms` and
      `results.to_public 0.094 ms`.
    - Targeted validation on 2026-06-15: compileall passed for `src/axonscope`
      plus the touched tests; mypy passed for the modified dispatcher/backend/
      result/simulation modules; targeted observer-only tests passed (`3
      passed`), related dispatcher/public API tests passed (`52 passed`), and
      hotpath catalog/performance tests passed (`14 passed`).
    - Evidence gate completed by
      `benchmark/results/hotpaths/colab_cpu_gpu_kernel_observer_long_20260615_122541/`.
  - Before declaring this cleanup done, re-check the same bottleneck map on
    double-cable/MRG-like runs; this is a long-term priority path, not an
    optional edge case.
    - Added `double_cable_extracellular` to `benchmark/hotpaths/`, using a
      homogeneous MRG double-cable population with analytical point-source
      extracellular stimulation.
    - Added Colab notebook case `kernel_double_cable_extracellular_long`
      with sizes `100/300`, `duration=5 ms`, `dt=0.01 ms`, and target
      `51` compartments.
    - Local smoke on 2026-06-15:
      `benchmark/results/hotpaths/local_double_cable_extracellular_warm/`.
      The run uses one `strict-double-cable` group with `geometry_shared=true`
      and `has_padding=false`; warm local `n=5`, `Nt=4`, `Nx=23` profile is
      total `38.1 ms`, `runtime.prepare 23.0 ms`, `kernel.enqueue 10.5 ms`,
      `inputs.extracellular 1.93 ms`.
    - Local fallback probes while Colab is unavailable:
      `benchmark/results/hotpaths/local_double_cable_extracellular_probe_n20/`
      and `benchmark/results/hotpaths/local_double_cable_extracellular_probe_n50/`.
      With `duration=1 ms`, `dt=0.02 ms`, target `51` compartments
      (`MRG Nx=45`), warm local CPU profiles are:
      `n=20` total `71.0 ms`, `runtime.prepare 33.5 ms`,
      `kernel.enqueue 17.5 ms`, `kernel.wait 12.4 ms`,
      `inputs.extracellular 3.25 ms`; `n=50` total `108.8 ms`,
      `runtime.prepare 45.3 ms`, `kernel.enqueue 25.0 ms`,
      `kernel.wait 24.4 ms`, `inputs.extracellular 6.23 ms`.
    - Local interpretation: the first double-cable extracellular bottleneck is
      runtime preparation plus actual double-cable kernel work. Dense
      `Vstim[B,Nt,Nx]` is visible and should remain tracked, but it is not yet
      the dominant local CPU cost at these sizes.
    - Local optimization while Colab is unavailable: cache
      `MembraneRuntime`, `CableRuntime`, and `ExtracellularRuntime` by static
      axon/model/options/geometry signatures. The membrane key includes `Nx`,
      dtype, `v_init`, and temperature; cable/extracellular keys hash the
      solver-side geometry and periaxonal arrays.
    - Local validation on 2026-06-15:
      `benchmark/results/hotpaths/local_double_cable_extracellular_after_runtime_cache_n50/`.
      On the same warm `n=50`, `duration=1 ms`, `dt=0.02 ms`, target
      `51` compartments (`MRG Nx=45`) probe, `runtime.prepare` dropped from
      `45.3 ms` to `2.55 ms`, and total wall time dropped from `108.8 ms` to
      `55.6 ms`.
    - Updated interpretation: double-cable extracellular preparation is no
      longer the dominant local cost. The next visible costs are the
      double-cable kernel boundary/work itself (`kernel.enqueue 25.0 ms`,
      `kernel.wait 16.0 ms`) plus dense extracellular input materialization
      (`inputs.extracellular 5.93 ms`). Do not over-optimize local CPU noise;
      re-check on Colab GPU as soon as possible.
    - Added a conservative analytical point-source fast path for shared
      homogeneous extracellular contexts, vectorizing footprint evaluation
      across batch rows/transverse axon offsets, then combined double-cable
      midpoint and initial-previous `Vstim` building so footprints are prepared
      once for both arrays.
    - Local validation on 2026-06-15:
      `benchmark/results/hotpaths/local_double_cable_extracellular_after_combined_vstim_builder_n50/`.
      On the same warm `n=50` probe, `inputs.extracellular` dropped from
      `5.93 ms` to `2.81 ms`; total wall time is now `54.4 ms`. The remaining
      local costs are dominated by `kernel.enqueue 23.4 ms` and
      `kernel.wait 19.3 ms`, so the next meaningful decision needs a Colab GPU
      trace rather than another CPU-only micro-optimization.
  - Before declaring this cleanup done, re-check extracellular stimulation
    runs with realistic `ExtracellularFootprint`/`ExtracellularDrive` usage,
    because dense `Vstim[B,Nt,Nx]` is expected to matter as much as dense
    `Iinj[B,Nt,Nx]` for the real product workload.
  - Keep padded single-cable group collapsing as a later backend task until
    row-specific recording selectors and observer masks are supported inside
    padded kernels.
- [ ] Extend realistic population benchmark workloads if the first traces show
  missing coverage.
  - Candidate additions: factorized `ExtracellularFootprint`/`ExtracellularDrive`
    population runs, MRG/double-cable paths, and larger realistic diameter/model
    distributions.
- [ ] Explore the remaining hotpath matrix to find remaining stalls.
  - Cover scalar fallback, homogeneous batch, parameter batch, padded groups,
    single-cable, double-cable, recording full/center/probes/none, observers
    on/off, extracellular on/off, and warm/cold runs.
  - Record whether the stall is in dispatch planning, preparation, input
    materialization, kernel enqueue, kernel wait, result splitting, public
    result packaging, or host transfer.
  - Keep `benchmark/hotpaths/` as the lightweight probe location until the
    larger benchmark-agent rewrite is justified by evidence.
- [ ] Keep footprint compression as a later design topic.
  - Candidate directions: sparse/low-rank footprints, shared footprint banks,
    quantized/static footprint storage, stimulus-only reuse, or lazy materialized
    dense `Vstim`.
  - Do not design this before the realistic hotpath matrix shows where dense
    footprint memory is actually limiting.
- [ ] Do a cleanup pass over examples before treating the public learning path
  as stable.
  - Homogenize style, naming, plotting conventions, and run headers.
  - Re-check examples against `agent.md`: didactic, line-by-line, not hidden
    behind helper scaffolding unless helpers are truly pedagogical.
  - Make examples verbose enough to guide a user through the workflow, with
    comments that explain the purpose of each important step.
  - Add more plots where they teach the signal/metric relationship: Vm traces,
    activation/peak markers, recruitment curves, velocity metrics, and
    recording/observer comparisons.
  - Keep examples runnable and lightweight; benchmark-heavy evidence stays under
    `benchmark/`.
- [ ] Later: define logging policy.
  - Decide what belongs in Python logging, benchmark traces, warnings, result
    diagnostics, and user-facing summaries.
- [ ] Later: define print/Rich/progress policy.
  - Decide what AxonScope should print by default, what is opt-in, what belongs
    to `progress=...`, and how `rich` output should degrade in notebooks/CI.
- [ ] Later: docstring and file-header responsibility pass.
  - Verify module docstrings, public class/function docstrings, file
    responsibility headers, and internal boundary descriptions.
  - Do this after Phase 7.6/8 clarifies which modules are stable enough to
    document as ownership boundaries.

## Remaining Guidelines Gaps To Resolve During Phases 7-9

These came from `GUIDELINES.md` passes during Phases 5-6. They are not all
blockers, but they must stay visible while the architecture converges.

- [x] Decide the final public analysis namespace before adding the Phase 6 package.
  - Previous guardrails rejected top-level `axs.analysis` while it only existed as an old forwarding alias.
  - `GUIDELINES.md` targets `axs.analysis.*` as a real analysis namespace.
  - Decision: introduce a real `axonscope.analysis` package and update guardrails so only the old forwarding-alias pattern stays forbidden.
- [ ] Decide whether scalar public runs eventually return `AxonSimulationResult`.
  - Current behavior: `simulate(...)` returns `SimResult`; `simulate_pool(...)` returns `AxonSimulationResult`.
  - Guideline target: one public result type for one axon and populations.
  - Phase 6.0-6.2 decision: defer the final result-unification decision, but require new analyses to accept both scalar and pool result surfaces.
- [ ] Remove `Recording` -> solver option coupling from the descriptive layer.
  - Current behavior: `Recording.to_batch_options()` imports and returns solver `BatchOptions`/`BatchRecording`.
  - Guideline target: `Recording -> RecordingPlan -> validation -> backend lowering`.
  - Best timing: during Phase 7/9 recording planning, backend lowering, and serialization cleanup.
- [ ] Add backend-neutral axon structure descriptors and cable capability descriptors.
  - Needed for analysis applicability, myelinated node-aware analyses, periaxonal signals, and semantic recording selectors.
- [ ] Extend built-in semantic signals beyond Vm/gates/currents/conductances/states.
  - Future target includes intracellular potential, periaxonal potential, ionic current, and cable/role-aware signal availability.
- [ ] Preserve the product boundary around external geometry packages.
  - AxonScope should consume footprints from geometry tools, not own CAD/electrode placement geometry.
- [x] Wire the first solver-side observers after Phase 7 memory evidence, before Phase 8 studies.
  - Done for `axs.analysis.PeakVoltage` and `axs.analysis.Activation`.
  - Scalar kernels and homogeneous single-cable batch kernels update compact observer state at every `dt`.
  - Observer-only runs use `Recording.none()` and return `result.observations` without retaining `Vm`.
  - Local hotpath evidence exists under `benchmark/results/hotpaths/phase7_5_observer_only_final/`.
  - Double-cable batch observer-only execution remains a future kernel specialization; it is not a Phase 7.5 blocker because scalar double-cable observer-only execution is correct and public.

## Documentation Audit Gate

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

Standing rules:

- Keep the current pre-release policy: no backward-compatibility aliases for prototype APIs unless they are strictly temporary inside the repo.
- Do not run or require NRV validation for cleanup steps unless explicitly requested; keep using the fast unit suite for non-NRV work.
- Treat Sphinx setup as paused until the current API and architecture direction are stable enough to document.
- Every advanced concept or non-trivial workflow needs a didactic `examples/advanced/` demo, written line-by-line with comments rather than hidden helper scaffolding.
- Examples are user-guidance artifacts: make them verbose/commented enough to
  demonstrate possibilities, and add plots when they help connect signals,
  metrics, dispatch, memory, or observer behavior.
- Update affected examples and `CHANGELOG.md` when changing public API or workflows.

### Phase 0 Guardrails Before Big Changes

- [x] Add architecture guardrail tests that prevent old and new public concepts from coexisting as permanent aliases.
  - [x] Guard root `GUIDELINES.md` as the project philosophy reference used by `agent.md` and `todo.md`.
  - [x] Guard against reintroducing removed top-level compatibility aliases such as `axs.visualization` and `axs.run_batch`; `axs.analysis` is allowed only as the real Phase 6 package.
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

### Roadmap Implementation Log And Active Issues

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
- [x] Phase 3 issue: split planning and preparation, add signatures and reusable prepared cohorts.
  - [x] Phase 2.5 before the larger split: add opt-in hotpath instrumentation to measure where CPU/GPU time is really spent.
  - [x] Phase 2.5 should instrument stage boundaries only: `dispatch.build_plan`, `dispatch.group.total`, `runtime.prepare`, `inputs.positions`, `inputs.intracellular`, `inputs.extracellular`, `kernel.enqueue`, `kernel.wait`, `results.split_batch`, and `results.to_public`.
  - [x] Phase 2.5 should record shapes, dtypes, estimated bytes, backend/device metadata, group size, `Nt`, `Nx`, recording mode, and whether geometry is shared.
  - [x] Phase 2.5 should avoid the full benchmark-framework rewrite for now; use the first traces to decide which Phase 3 preparation boundaries matter most.
  - [x] After Phase 2.5, run diagnostic workloads for intracellular-only, point-source extracellular, and population sizes around 5/50/500 before committing to a GPU refactor direction.
    - [x] Add `benchmark/hotpaths/` as the cataloged location for Phase 2.5 workload scripts.
    - [x] Add `benchmark/hotpaths/README.md` and `benchmark/hotpaths/run.py --list` as the human/CLI registry for workload scripts.
    - [x] Add `intracellular_only` and `point_source_extracellular` workloads with `smoke` and `scale` size presets.
    - [x] Smoke-run both hotpath workloads with size 2 on 2026-06-14: `python benchmark/hotpaths/run.py --workload all --sizes 2 --duration 0.10 --dt 0.05 --compartments 5 --prefix smoke_test --no-print-summary`.
    - [x] Probe-run the scale preset on the current environment on 2026-06-14: `python benchmark/hotpaths/run.py --workload all --preset scale --prefix scale_probe --no-print-summary`.
    - [x] Document the manual Google Colab GPU protocol in `benchmark/hotpaths/COLAB.md`; local GPU execution is not assumed.
    - [x] Manually run the first Google Colab GPU trace and bring it back under `benchmark/results/hotpaths/colab_gpu_YYYYMMDD/`.
    - [x] Compare the first CPU/GPU traces against `ideas/AXONSCOPE_CPU_GPU_BOTTLENECK_ANALYSIS.md`.
      - Evidence from `n=500`: GPU `kernel.wait` is negligible (`0.13 ms` intracellular-only, `0.116 ms` point-source), so the first bottleneck is not device execution wait.
      - Evidence from `n=500`: `dispatch.build_plan` is a major host-side cost (`~7.0-7.5 s` on Colab GPU, `~10.7-16.9 s` on the local CPU run).
      - Evidence from `n=500`: input materialization dominates after planning (`inputs.intracellular` up to `9.5 s`, `inputs.extracellular` up to `6.76 s` on Colab GPU).
      - The `n=5` traces are first-call/compilation polluted; use `n=50` and `n=500`, preferably with `--warmups 1`, for decisions.
    - [x] Add the repeatable Colab publishing workflow before the next GPU trace.
      - Added `make bench-colab-push`, which pushes the current clean commit to the moving `bench-colab` branch without switching local branches.
      - Added `benchmark/hotpaths/colab_gpu_hotpaths.ipynb` and updated `benchmark/hotpaths/COLAB.md` so Colab clones `bench-colab`, installs `.[examples,benchmark]`, verifies the JAX GPU backend, runs the warm scale probe, writes outputs under `benchmark/results/hotpaths/`, zips the run folder, and downloads it directly without Google Drive.
    - [x] Re-run a cleaner Colab GPU trace with warmup: `python benchmark/hotpaths/run.py --workload all --preset scale --warmups 1 --prefix colab_gpu_warm_YYYYMMDD --no-print-summary`.
      - Completed via Colab prefix `colab_gpu_20260614_190321`; Colab reported `jax` GPU backend on `cuda:0`.
      - Warm scale `n=500` after Phase 3: `intracellular_only` total `145.2 ms`, `point_source_extracellular` total `117.4 ms`.
    - [x] Keep or create a matching CPU reference prefix, then compare both result folders under `benchmark/results/hotpaths/`.
      - Compared against local warm prefix `phase3_after_split_numpy`: local `n=500` totals were `117.0 ms` intracellular-only and `86.4 ms` point-source.
      - Decision: Phase 3 is done because the seconds-scale host planning/input bottlenecks are gone; tiny current workloads are not yet GPU-favorable, and remaining public-result packaging belongs to Phase 5.
  - [x] Phase 3 PR 3.1: add deterministic preparation signatures for arrays, stimuli, extracellular footprints, drives, and stimulation collections.
  - [x] Phase 3 PR 3.1: add `examples/advanced/example_15_preparation_signatures.py` as the required didactic demo for preparation signatures.
  - Fresh unit run on 2026-06-14 after Phase 2.5 hotpath instrumentation, workload catalog, and Phase 3.1 preparation signatures: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit --tb=short` (`277 passed, 1 skipped`).
  - [x] Phase 3.2 bottleneck attack plan: attack now, but treat it as planning/preparation stabilization rather than GPU/kernel optimization.
    - Decision: do this before Phase 4 backend isolation and before the full Phase 7 benchmark rewrite, because the measured bottlenecks are host-side planning and input materialization.
    - [x] PR 3.2A: reduce `dispatch.build_plan` cost before touching kernels.
      - Compute each dispatch signature once per item and reuse it through grouping.
      - Avoid repeated O(N^2)-style compatibility/signature recomputation for homogeneous pools.
      - Cache or share `SolverAxon`/layout signatures when many `AxonInstance` rows wrap the same descriptive `Axon`.
      - Acceptance gate: `dispatch.build_plan` should drop by at least one order of magnitude on `*_n500`, and should no longer dominate simple homogeneous pools.
    - [x] PR 3.2B: add a zero-extracellular fast path for batches with no contexts.
      - If every context row is empty, build `Vstim[B, Nt, Nx]` directly as zeros instead of looping over rows.
      - Acceptance gate: `intracellular_only_n500` should not spend material time in `inputs.extracellular`.
    - [x] PR 3.2C: introduce `PreparedCohort` as the first reusable preparation object.
      - Hold grouped axons, solver rows, positions, context rows, and prepared input metadata.
      - Keep deterministic signature computation in the dispatch/preparation boundary until the public backend split is clearer.
      - Keep public behavior unchanged; route current dispatcher through the prepared object internally.
      - Add `examples/advanced/` didactic demo only if a public concept is exposed.
    - [x] PR 3.2D: lower analytical point-source stimulation through footprints/prepared drives.
      - Use `ExtracellularFootprint`/`ExtracellularDrive` as the intended prepared representation.
      - Avoid per-row analytical context compilation when a static footprint can be reused or vectorized.
      - Acceptance gate: `point_source_extracellular_n500` should show a clear drop in `inputs.extracellular`.
    - [x] PR 3.2E: reduce `results.split_batch` overhead only after planning/input preparation are no longer dominant.
      - Keep this secondary; current traces show it matters, but not enough to lead the refactor.
    - [x] Re-run `benchmark/hotpaths/run.py --workload all --preset scale --warmups 1` after each PR 3.2 step and record the before/after in this TODO.
      - Local final warm run on 2026-06-14: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python benchmark/hotpaths/run.py --workload all --preset scale --warmups 1 --prefix phase3_after_split_numpy --no-print-summary`.
      - Local `intracellular_only_n500` final warm trace: total `117.0 ms`, `dispatch.build_plan 4.25 ms`, `inputs.positions 1.59 ms`, `inputs.intracellular 39.6 ms`, `inputs.extracellular 0.67 ms`, `kernel.enqueue 2.95 ms`, `kernel.wait 5.16 ms`, `results.split_batch 48.0 ms`.
      - Local `point_source_extracellular_n500` final warm trace: total `86.4 ms`, `dispatch.build_plan 3.89 ms`, `inputs.positions 1.25 ms`, `inputs.intracellular 0.98 ms`, `inputs.extracellular 12.1 ms`, `kernel.enqueue 3.03 ms`, `kernel.wait 4.48 ms`, `results.split_batch 46.9 ms`.
      - Compared with the pre-Phase-3 local `scale_probe` no-warmup `n=500`, `dispatch.build_plan` dropped from seconds to milliseconds, point-source `inputs.extracellular` dropped from seconds to tens of milliseconds/no-warmup and ~12 ms warm, and zero-context `inputs.extracellular` is no longer material.
      - Residual: `results.split_batch` still costs ~47-48 ms at `n=500` due per-row public result objects; keep the canonical cohort-backed result model in Phase 5 rather than overfitting Phase 3.
      - Fresh unit run on 2026-06-14 after Phase 3.2 local stabilization: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit --tb=short` (`281 passed, 1 skipped`).
  - [x] Phase 3 next after PR 3.2: move solver lowering behind reusable prepared cohorts without changing the public API.
    - Added internal `PreparedCohort` and routed current batch input preparation through it.
    - Kept `PreparedCohort` out of the public `axs.preparation` facade, so no new advanced didactic example is required yet.
- [x] Phase 4 issue: isolate JAX runtime under backend modules and delete old dispatcher/solver paths once empty.
  - [x] PR 4.1: inventory current JAX-owned files/functions and define the smallest backend boundary.
    - Keep public API unchanged.
    - Keep `PreparedCohort` as the input boundary from dispatcher/preparation into backend lowering.
    - Start with file/module moves and explicit interfaces, not kernel rewrites.
    - Added `ideas/AXONSCOPE_PHASE4_BACKEND_BOUNDARY.md` with the current JAX-owned surface, smallest useful boundary, PR sequence, guardrails, non-goals, and acceptance criteria.
  - [x] PR 4.2: add a `jax` backend package for solver lowering and kernel invocation.
    - Added `src/axonscope/backends/jax/group_runner.py` as the first real backend-owned execution boundary.
    - Moved JAX-specific batch runtime preparation, runtime stacking/padding helpers, batch input materialization, kernel invocation, synchronization, and batch result splitting out of `dispatcher/execution.py`.
    - Moved `DispatchResult` into `dispatcher/results.py` so dispatcher orchestration and backend execution share a small neutral result contract without import cycles.
    - Kept descriptive axon, stimulation, recording, and public result objects unchanged.
  - [x] PR 4.3: add guardrails preventing new backend-specific imports from leaking into public/descriptive layers.
    - Added architecture tests that reject direct JAX imports from `axons/`, `membranes/`, `recording.py`, `population.py`, `results/`, and `preparation/signatures.py`.
    - Added a guardrail that keeps concrete batch kernel/runtime imports out of `dispatcher/execution.py`; dispatcher may call the backend runner but must not own the JAX kernels again.
  - [x] PR 4.4: split JAX input lowering out of `dispatcher/runtime_batches.py`.
    - Moved JAX batch input builders into `src/axonscope/backends/jax/input_batches.py`.
    - Reduced `dispatcher/runtime_batches.py` to host-side row helpers used by preparation and benchmark utilities.
    - Updated dispatcher backend execution, tests, and runtime benchmark scripts to import JAX tensor builders from the backend module.
    - Added a guardrail that keeps `dispatcher/runtime_batches.py` host-side only.
  - Fresh validation on 2026-06-14 after completing Phase 4: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m compileall -q src tests/unit benchmark/runtime` passed; targeted guardrail/solver/dispatcher/batch run passed (`70 passed`); full unit run passed (`286 passed, 1 skipped`); full NRV run passed (`116 passed, 516 warnings`); hotpath smoke passed with prefix `benchmark/results/hotpaths/phase4_final_smoke/`.
  - [x] PR 4.5: move scalar Crank-Nicholson execution behind the JAX backend boundary.
    - Added `src/axonscope/backends/jax/scalar_runner.py` for scalar runtime preparation, kernel selection, kernel invocation, and scalar backend output collection.
    - Reduced `src/axonscope/solvers/crank_nicholson.py` to a public solver facade that resolves inputs, delegates to the backend, and wraps the backend output in `SimResult`.
    - Removed direct JAX usage from `simulation.py` by using NumPy for scalar-fallback recording filters.
    - Added guardrails that keep direct JAX imports out of `simulation.py`, keep `dispatcher/runtime_batches.py` host-only, and prevent `CrankNicholson` from importing concrete runtime/kernel modules again.
    - Decision: keep low-level kernel/runtime modules under `solvers/` for this phase because they still own tested numerical kernels and are not duplicate execution paths; the public execution entry points now cross the JAX backend boundary first.
- [x] Phase 5 issue: replace list-based pool results with canonical cohort-backed results and per-axon views.
  - [x] PR 5.1: add public `CohortResult`, `AxonSimulationResult`, and `AxonResultView`.
  - [x] PR 5.1: make `simulate_pool(...)` return `AxonSimulationResult` instead of `list[SimResult]`.
  - [x] PR 5.1: keep ergonomic per-axon access through `len(result)`, `result[index]`, iteration, `.axon(index)`, and `.single`.
  - [x] PR 5.1: add `examples/advanced/example_16_canonical_pool_results.py` as the required didactic demo for canonical pool results.
  - Fresh unit run on 2026-06-14 after Phase 5.1 canonical pool results: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit --tb=short` (`288 passed, 1 skipped`).
  - [x] PR 5.2: add richer signal descriptors and recording manifests beyond Vm-only pool output.
    - Replaced the closed `Signal` enum with extensible `Signal` descriptors plus typed `SignalId`.
    - Added `RecordingManifest` and `RecordedSignal` on `AxonSimulationResult` so pool results expose requested signals, available signals, and per-cohort shape/dtype metadata.
    - Updated `examples/advanced/example_16_canonical_pool_results.py` to show manifest inspection.
    - Targeted validation on 2026-06-14: public API/facade/units/examples tests passed (`56 passed, 1 skipped`); targeted mypy on signal/recording/result files passed.
    - Fresh unit run on 2026-06-14 after Phase 5.2 descriptors/manifests: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit --tb=short` (`289 passed, 1 skipped`).
  - [x] PR 5.3: move remaining analysis/result-status assumptions into the Phase 6 provenance layer instead of expanding `SimResult` compatibility.
    - Added guardrails that keep `simulate_pool(...)` typed as `AxonSimulationResult`, reject the old `list[SimResult]` return annotation, and keep public signals as extensible descriptors rather than a closed enum.
    - Decision: do not expand `SimResult` compatibility or analysis status fields in Phase 5; Phase 6 owns analysis requirements, applicability, statuses, and provenance.
    - Targeted guardrail validation on 2026-06-14: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit/test_architecture_guardrails.py --tb=short` (`18 passed`).
  - Fresh final unit run on 2026-06-14 after completing Phase 5: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/bin/mamba run -n Axonscope-env python -m pytest -q tests/unit --tb=short` (`291 passed, 1 skipped`).
- [x] Phase 6 issue: move scientific analyses into a dedicated requirements/status/provenance layer.
  - [x] PR 6.0: resolve the public analysis namespace decision (`axs.analysis` real package vs `axs.results.analysis`).
  - [x] PR 6.1: add analysis definition objects for activation, conduction velocity, latency/block, spike count, and peak voltage.
  - [x] PR 6.1: each analysis must declare required semantic signals, supported myelination/formulation, required compartment roles, positions, and algorithm version.
  - [x] PR 6.2: add structured per-axon statuses: `VALID`, `NOT_APPLICABLE`, `MISSING_INPUT`, `NUMERICAL_FAILURE`, and `UNDETERMINED`.
  - [x] PR 6.2: add analysis result containers with population denominators (`n_total`, `n_applicable`, `n_valid`, `n_failed`) and no silent NaN completion.
    - Added `AnalysisRequirements`, `AnalysisStatus`, `AnalysisPopulation`, `AnalysisResult`, `AnalysisReport`, and `result.analyze(...)` / `result.report(...)`.
    - Added `examples/advanced/example_17_analysis_layer.py` as the required didactic demo for the structured analysis layer.
    - Fresh unit run on 2026-06-14 after Phase 6.0-6.2 structured analysis layer: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit --tb=short` (`298 passed, 1 skipped`).
  - [x] PR 6.3: move current post-hoc activation and velocity algorithms out of `results/analysis.py` into the chosen analysis layer.
  - [x] PR 6.3: keep compatibility only as internal imports during the same PR if needed; do not leave permanent forwarding aliases.
    - Moved `rasterize`, `conduction_velocity`, `average_velocity`, `peak_voltage`, and `recorded_positions_um` to `axonscope.analysis.posthoc`.
    - Moved `ActivationCriterion`, `ActivationEvent`, and `detect_activation` to `axonscope.analysis.activation`.
    - Removed public `axs.results.analysis`, `axs.results.ActivationCriterion`, and the old `results/analysis.py` / `results/activation.py` modules.
    - Fresh unit run on 2026-06-14 after Phase 6.3 post-hoc helper migration: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit --tb=short` (`298 passed, 1 skipped`).
  - [x] PR 6.4: add missing-input errors that describe the required recording instead of rerunning simulations silently.
    - Added `AnalysisInputRequirement` and per-row `AnalysisResult.input_requirements` / `.missing_input_requirements`.
    - Analysis requirements now carry required result fields, capability tags, and recording hints.
  - [x] PR 6.5: add online observer definitions only after post-hoc analysis objects and statuses are stable.
    - Added `ActivationObserver` and `PeakVoltageObserver` for streamed Vm chunks.
  - [x] PR 6.5: cross-validate online and post-hoc results on the same representative traces.
    - Added unit cross-validation for activation and peak-voltage observers.
  - [x] Add a didactic advanced example for the new analysis layer when the public concept lands.
    - Updated `examples/advanced/example_17_analysis_layer.py` with online/post-hoc observer comparison.
  - Fresh final unit run on 2026-06-14 after completing Phase 6: `MPLBACKEND=Agg /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit --tb=short` (`300 passed, 1 skipped`).
- [x] Phase 7 issue: finalize benchmark/performance story, footprint reuse, and memory estimates.
  - [x] Reconcile `benchmark/hotpaths/` with `ideas/AXONSCOPE_BENCHMARKING_AGENT_SPEC.md` before building a larger benchmark framework.
    - Decision: keep `benchmark/hotpaths/` as the lightweight evidence loop, add memory estimates to manifests, and defer the larger benchmark-agent rewrite until after solver-side observers/studies clarify the steady API.
  - [x] Keep hotpath traces as the evidence loop for CPU/GPU bottlenecks; do not overfit one Colab run.
    - `benchmark/hotpaths/run.py` now records `simulation.estimate().to_dict()` in each run manifest.
  - [x] Verify whether extracellular execution still materializes avoidable dense `Vext`/`Vstim`.
    - Phase 7 estimate reports the current dense `Vstim[B,Nt,Nx]` materialization risk and compares it with factorized footprint/stimulus sizes; removing that memory pressure is Phase 7.5.
  - [x] Add footprint reuse benchmarks for stimulus-only sweeps.
    - Added `footprint_reuse_sweep` for repeated fixed-geometry point-source pool runs with changing stimulus amplitude.
  - [x] Add memory estimates and warning thresholds for dense fallback paths.
    - Added `axs.performance`, `SimulationEstimate`, `MemoryEstimateItem`, `axs.estimate_simulation(...)`, and `AxonSimulation.estimate()`.
  - [x] Add typed runtime/device/precision policy objects or decide where they belong before serialization.
    - Added public `Runtime`, `Device`, and `PrecisionPolicy` planning values. They are estimate/planning objects for now, not execution selectors yet.
  - [x] Separate correctness validation, profiling, and product-level performance reports in docs/scripts.
    - README and `benchmark/hotpaths/README.md` now distinguish tests/NRV validation, hotpath diagnostics, memory estimates, and future benchmark-framework work.
  - Targeted validation on 2026-06-14: performance/hotpath/public API tests passed (`12 passed`); `mypy src/axonscope/performance.py benchmark/hotpaths/run.py` passed; `example_14_hotpath_benchmarking.py` ran; `footprint_reuse_sweep` smoke run wrote a manifest with memory estimates.
- [x] Phase 7.5 issue: wire observers directly into solver kernels to reduce GPU memory pressure.
  - [x] Define a backend-neutral observer lowering contract: public analysis/observer spec -> compact static observer state -> final observation output.
  - [x] Define the per-step kernel contract: every solver step calls observer `update(dt_index, t, Vm/current state, observer_state)` inside the scan/loop.
  - [x] Start with `PeakVoltageObserver`, because it is a simple max reduction and validates the state/update/finalize plumbing.
  - [x] Add `ActivationObserver` after peak voltage, preserving blanking, target positions, first-crossing time, and compact activation output.
  - [x] Support observer-only execution where possible: users can request compact observations without retaining full `Vm[t, x]`.
  - [x] Keep fixed-shape observer state vectorized over batch rows so JAX can compile it and GPU memory stays predictable.
  - [x] Add hotpath/memory traces proving observer-only runs avoid full Vm retention and device-to-host transfer.
    - Added `observer_only` to `benchmark/hotpaths/`.
    - Final local smoke trace on 2026-06-15: `benchmark/results/hotpaths/phase7_5_observer_only_final/manifest.json`.
    - Evidence: `vm_shapes=[]`, `observation_names=["activation", "peak_voltage"]`, `recording_policy="none"`, `recording_width_max=0`, `outputs.recorded_vm` shape `[5, 6, 0]`, `retained_mib=0.0`.
  - [x] Cross-validate solver-side observer outputs against Phase 6 post-hoc/streamed Vm observers on scalar and pool traces.
  - [x] Add a didactic advanced example for the public observer-only workflow.
    - Added `examples/advanced/example_18_solver_side_observers.py` as the dedicated line-by-line observer-only demo.
    - Updated `examples/advanced/example_14_hotpath_benchmarking.py` to show `Recording.none()` with solver-side `PeakVoltage` and `Activation` observers in the hotpath/memory context.
  - [x] Scope decision: double-cable batch observer-only execution can remain on scalar fallback for Phase 7.5; moving it onto the double-cable batch kernel is a later backend-specialization task, not part of the first public observer-only workflow.
- [ ] Phase 8 issue: add callable studies, reuse policies, retention policies, and study results.
  - [ ] Define the callable update contract for sweeps, thresholds, and recruitment.
  - [ ] Add reuse policies for prepared cohorts, compiled kernels, footprints, and stimulus-only updates.
  - [ ] Add retention policies so threshold/recruitment studies do not retain every trace by default.
  - [ ] Add study result containers with compact per-row/per-amplitude outputs and optional retained traces.
  - [ ] Document callable reproducibility limits; do not claim arbitrary lambdas are serializable.
- [ ] Phase 9 issue: finalize serialization schemas and add NumPy reference backend validation.
  - [ ] Define final schemas only after object/result/analysis/study models settle.
  - [ ] Serialize typed values, identifiers, recording manifests, analysis definitions, backend/device/precision, and environment metadata.
  - [ ] Do not add readers for prototype formats.
  - [ ] Add NumPy reference backend validation for small deterministic cases before treating serialization as stable.
  - [ ] Add final docs only after schemas and reference validation are stable.

### Completed Contract Preparation Archive

- [x] Design `ExtracellularFootprint` as a static spatial transfer object with intrinsic axon positions and units, not electrode CAD/world geometry.
- [x] Design `ExtracellularDrive` as footprint plus temporal `Stimulus`.
- [x] Design `ExtracellularStimulation` as the aggregate of one or more drives.
- [x] Plan migration of analytical point-source helpers into footprint builders outside the core solver dependency path.
- [x] Sketch typed selectors/signals/enums before replacing remaining string-based public domains.

### Later Target Architecture Checklist

- [x] Split planning, preparation, execution, and backend lowering so JAX-specific code is isolated under backend runtime modules.
- [x] Replace eager `list[SimResult]` pool results with a canonical cohort-backed result model and per-axon views.
- [x] Move activation/velocity analysis into a dedicated analysis layer with applicability/status/provenance metadata.
- [x] Wire the first solver-side observers directly into scalar and single-cable batch kernels as per-`dt` compact reductions before callable studies.
- [ ] Add callable studies, reuse policies, retention policies, and final serialization only after the object model/result model settle.

## Backlog: Documentation Audit Snapshot

Started from `agent.md` and a code/docs grep on 2026-06-13. This is a first
pass, not a completed audit.

| Page | Status | Notes / Next Action |
| --- | --- | --- |
| `docs/axon_model_organization.md` | partially current | Mostly matches the descriptive layer and unit-boundary direction. Re-check examples against `src/axonscope/axons/` before treating as Sphinx-ready. |
| `docs/solver_organization.md` | likely current | File list and time-grid behavior match current solver modules and `simulation_step_count`; keep as a candidate for Sphinx with light verification. |
| `docs/membranes.md` | mostly current | Built-in membrane namespace and unit normalization match `src/axonscope/membranes/`; still verify `Composite`, `SectionLayout`, and examples against tests. |
| `docs/stimulation.md` | mostly current | Known `HodgkinHuxley(length_um=..., diameter_um=...)` snippet was updated to `length=...` and `diameter=...`; still do a final full-page pass before Sphinx. |
| `docs/pool_dispatch.md` | mostly current | Public `simulate_pool`, dispatch diagnostics, and `build/print/plot_dispatch_plan` exist. Review for overlap with README and for any advanced batch API drift. |
| `docs/results_recording_analysis.md` | partially stale | Good conceptual split for `Recording`, `SimResult`, analysis, visualization, and online Vm observers, but now needs a Phase 7.5 refresh for `Recording.none()` solver-side observations. |
| `docs/recorders_observers_activation_strategy.md` | proposal | CPU/post-hoc activation, protocol sweeps, and lightweight Vm observers now exist; Phase 7.5 added the first solver-side observer-only execution path. |
| `docs/api_public_draft.md` | proposal-only | Clear proposal/roadmap warning added at the top. Later split implemented API from proposal if this document remains user-facing. |
| `docs/validation.md` | mostly current | Removed default GitHub Actions and stale NRV pass-count claims. No fresh NRV result is recorded yet. |

## Backlog: Completed Documentation Mismatches

- [x] `README.md` lists `examples/basic/example_06_velocity_vs_diameter_batch.py`, but the current file is `examples/basic/example_06_velocity_vs_diameter.py`.
- [x] `tests/unit/test_examples.py` imports `examples.basic.example_06_velocity_vs_diameter_batch`; update to `examples.basic.example_06_velocity_vs_diameter`.
- [x] `examples/basic/example_06_velocity_vs_diameter.py` still shows the removed `_batch.py` filename in its module docstring run command.
- [x] `README.md` should list the new basic examples after the rename/additions: `example_06_velocity_vs_diameter.py`, `example_07_threshold_vs_diameter.py`, and `example_08_recruitment_curve_population.py` if they are intended to be part of the public learning path.
- [x] `docs/stimulation.md` should replace `length_um`/`diameter_um` axon constructor snippets with `length`/`diameter` quantity-based calls.
- [x] `docs/api_public_draft.md` still uses target snippets with `length_um`, `diameter_um`, `Recording.none()` as runnable behavior, and solver-side observers. Label these as proposal or move them to roadmap sections.
- [x] `docs/recorders_observers_activation_strategy.md` mentions observer-only runs, amplitude-batched GPU sweeps, and `thresholds_for_pool`; current code has post-hoc `ActivationCriterion`, lightweight `ActivationObserver`/`PeakVoltageObserver`, `detect_activation`, `find_activation_threshold`, `find_activation_threshold_curve`, `pool_sweep`, and `recruitment_sweep`.
- [x] `docs/results_recording_analysis.md` had future observer examples under the old results-analysis namespace; update to `axs.analysis` with current lightweight Vm observers and mark solver-side observer execution as Phase 7.5.
- [x] `docs/validation.md` says the default GitHub Actions workflow runs checks, but this checkout has no `.github/` directory.
- [x] `docs/validation.md` hard-codes `116 passed` for NRV validation; replace with a dated validation note only after rerunning in an NRV-ready environment.
- [x] `CHANGELOG.md` references absent paths/features including `axonscope.stimulation.evaluation`, `axonscope.solvers.stimulus_runtime`, `euler.py`, and a GitHub Actions workflow.

## Backlog: Documentation Cleanup

- [x] Fix the example 06 rename everywhere: `README.md`, `examples/basic/example_06_velocity_vs_diameter.py`, and `tests/unit/test_examples.py`.
- [x] Update the README package map so analysis helpers point to the real `axs.analysis` package and plotting remains under `axs.results.visualization`.
- [x] Normalize constructor examples in docs to the implemented public names: `length`, `diameter`, `position`, `positions`, `sample_dt`, `duration`/`dt` for public wrappers, and `tsim`/`dt` for direct solver calls. Current docs are aligned outside the explicitly proposal-only `docs/api_public_draft.md`.
- [x] Add a clear warning at the top of `docs/api_public_draft.md` so stale target snippets are not mistaken for current runnable API.
- [ ] Later split `docs/api_public_draft.md` into implemented API versus proposal if it remains part of the user-facing docs.
- [x] Refresh `docs/recorders_observers_activation_strategy.md` implementation status with the current protocol functions and mark observer-only/GPU observer work as Phase 7.5.
- [x] Audit `CHANGELOG.md` against files that actually exist in this checkout; remove or reword absent module names and CI claims.
- [x] Re-run `python -m pytest -q tests/unit` after doc/example/API cleanup fixes and record only fresh results. Fresh run on 2026-06-14 after typed recording signals: `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/axonscope-mpl /Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python -m pytest -q tests/unit` (`254 passed, 1 skipped`).
- [ ] Re-run NRV validation only in an NRV-ready environment; record dated validation notes after a fresh run.
- [x] Remove duplicated narrative between README, `docs/pool_dispatch.md`, and `docs/results_recording_analysis.md` by making README a short entry point and keeping detailed contracts in `docs/`.

## Backlog: Public API And Units

- [ ] Stabilize the public API before adding Sphinx.
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
| `axs.analysis` / `axs.visualization` | Top-level aliases duplicated `axs.results.analysis` and `axs.results.visualization`. | Updated: old forwarding aliases were removed; `axs.analysis` is now a real Phase 6 package, while `axs.visualization` remains absent and plotting stays under `axs.results.visualization`. | `src/axonscope/__init__.py`, `tests/unit/test_public_api.py`, `src/axonscope/analysis/` |
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
| `SimResult.Vm` | Convenience property/field used broadly in examples and tests. | Keep as a stable notebook-friendly convenience, not a temporary compatibility shim. Errors are now explicit when observer-only runs do not carry Vm. | `src/axonscope/results/single.py`, recording/observer docs |
| `clear_extracellular_contexts(...)` | Plural name while the current instance stores one extracellular context internally. | Done: renamed to singular `clear_extracellular_context(...)`. The runtime tuple property remains `extracellular_contexts` for lower-level batch helpers. | `src/axonscope/axon_instance.py`, `tests/unit/test_public_api_facade.py` |

## Backlog: Tutorials And Examples

- [ ] Write tutorials after the docs/code audit so they match the real API.
- [ ] Add realistic examples with NRV context/validation where appropriate.
- [ ] Keep basic examples didactic and compact.
- [ ] Keep advanced examples realistic enough to show actual stimulation studies and pool workflows.

## Backlog: Recording, Observables, And Analysis

- [ ] Implement solver-side observables/observers in Phase 7.5 after Phase 7 memory/performance evidence, before Phase 8 studies.
- [x] Decide and document whether `Recording` options are handled directly by solvers, translated before solver entry, or both.
- [ ] Verify single-axon and pool recording behavior for Vm, gates, currents, conductances, and state variables. Current docs state that pool observable groups are future work; keep this open until behavior and tests are complete across supported groups.
  - [x] Lock the current public `Recording` contract with tests: scalar runs require `Vm` and may include observable groups, pool runs support `Vm` spatial modes only, and unsupported position/temporal/pool-observable filters raise explicit errors.
- [ ] Keep post-hoc `ActivationCriterion` semantics aligned with future solver-side observers.
- [x] Move analysis applicability/status/provenance tasks from this backlog into Phase 6 PRs as they become concrete.

## Backlog: Pool And Batch UX

- [x] Document and surface existing dispatch inspection helpers: `axs.dispatcher.build_dispatch_plan`, `print_dispatch_plan`, and `plot_dispatch_plan`.
- [ ] Add plotting helpers for batch groups and retained recording layouts.
- [x] Make batch diagnostics discoverable from per-axon public result views.

## Backlog: Benchmarks, CPU/GPU, And Bottlenecks

- [ ] Rework benchmark strategy; current benchmark story is not convincing enough --> see `ideas/AXONSCOPE_BENCHMARKING_AGENT_SPEC.md`.
- [x] Use a Phase 2.5 diagnostic pass before rebuilding the full benchmark suite: add opt-in hotpath spans, run a few representative traces, then use the evidence to steer Phase 3.
  - [x] Added `axs.enable_benchmark(...)`, `axs.disable_benchmark(...)`, `axs.benchmark_report(...)`, `axs.reset_benchmark()`, and `with axs.benchmark(...)`.
  - [x] Added raw `events.jsonl`, aggregate `summary.csv`, and `metadata.json` outputs for hotpath sessions.
  - [x] Added `examples/advanced/example_14_hotpath_benchmarking.py` as the didactic diagnostic demo.
  - [x] Added `benchmark/hotpaths/` as the registered location for Phase 2.5 workload scripts.
  - [x] Added `benchmark/hotpaths/run.py --list` and `benchmark/hotpaths/README.md` to catalog available hotpath workloads.
- [ ] Find a robust way to benchmark CPU versus GPU for representative workloads.
- [ ] Identify current bottlenecks and where the GPU path will likely hit memory, compilation, transfer, or batching limits. --> see `ideas/AXONSCOPE_CPU_GPU_BOTTLENECK_ANALYSIS.md`.
  - [x] First hotpath traces confirm the analysis direction: the immediate bottlenecks are `dispatch.build_plan`, `inputs.intracellular`, and `inputs.extracellular`, not GPU `kernel.wait`.
  - [x] Decision on timing: attack these as Phase 3.2 planning/preparation fixes now; do not wait for Phase 7 or Phase 4 backend isolation.
  - [x] Phase 3.2 used `benchmark/hotpaths/` as the evidence loop and closed the host-side planning/input bottlenecks.
  - [ ] Keep `benchmark/hotpaths/` as the Phase 7 evidence loop for memory estimates, footprint reuse, and CPU/GPU comparisons.
- [ ] Separate correctness validation from performance benchmarking in docs and scripts.
- [ ] Record environment/device metadata for benchmark runs.
- [ ] Defer the larger benchmark agent/spec implementation until after planning/preparation/cohort boundaries are clearer.

## Backlog: Documentation Platform

- [x] Rewrite `README.md` from scratch as a short, current entry point.
- [ ] Provide extensive docstrings on public classes/functions before generating API docs.
- [ ] Set up Sphinx documentation after the `/docs` audit.
- [ ] Decide what belongs in Sphinx pages versus README versus examples.
- [x] Keep proposal/roadmap docs clearly labeled so users do not run future API snippets as current API.

## Backlog: Cleanup And Sync

- [ ] Do a general cleanup pass after docs, examples, recordings, observers, and benchmarks are aligned.
- [ ] Remove stale aliases, removed file references, duplicate docs, and dead benchmark/example paths.
- [ ] Keep `agent.md` and this `todo.md` synchronized after each cleanup step.
