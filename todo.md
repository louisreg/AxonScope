# AxonScope TODO

Living operational roadmap for AxonScope documentation, API cleanup, examples,
benchmarks, solver/backend work, and Phase 8+ study APIs.

Read this file at the start of cleanup/API/performance work. Add newly
discovered mismatches here, and check items only after code, docs, examples,
and relevant tests have been verified.

## How To Use This File

- `GUIDELINES.md` is the master architecture and product-boundary reference.
  Update it only when the target philosophy, product boundary, or architecture
  changes.
- `agent.md` captures project working rules for future agents. Keep it aligned
  when the workflow or standing policy changes.
- `todo.md` is the chronological execution plan. Keep it actionable, mostly
  flat, and free of long benchmark prose.
- Source, tests, and runnable examples remain the truth for current behavior.
- AxonScope is pre-release: prefer clean breaking changes over compatibility
  shims, and delete superseded paths once replacements are in use.

## Current Snapshot

Updated on 2026-06-16 after re-reading `agent.md`, checking `GUIDELINES.md`,
and auditing the current solver/pseudo-double code paths.

| Area | Status | Notes |
| --- | --- | --- |
| Phases 0-7.5 | Done | Guardrails, object model, typed contracts, preparation, JAX boundary, canonical pool results, analysis layer, performance estimates, and solver-side observers are implemented for the current public layer. |
| Phase 7.6.1 | Done | Benchmark evidence matrix exists under `benchmark/hotpaths/`. |
| Phase 7.6.2 | Done | Memory-transfer and long-run cleanup has landed for current hotpaths. |
| Phase 7.6.3 | Active | Exact double-cable GPU solver optimization. See `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`. |
| Phase 7.6.4 | Standby | Pseudo-double/pseudo-MRG remains validation-only under `benchmark/pseudo_double/`; not a public solver path and not part of `auto`. |
| Phase 7.7 | Next | Stimulation and placement API cleanup against `GUIDELINES.md`. |
| Phase 7.8 | Next | Examples learning-path cleanup after API cleanup and solver-option docs. |
| Phase 8 | Not started | Callable studies, reuse policies, retention policies, and study results. |
| Phase 9 | Not started | Final serialization schemas and NumPy/reference backend validation. |

Latest verified non-NRV unit run:

- [x] 2026-06-15 after compact dispatch cohort cleanup:
  `314 passed, 1 skipped`.

Current code audit:

- `BatchOptions.double_cable_block_solver` accepts exactly `auto`, `thomas`,
  `pcr`, `pcr_soa`, and `pcr_adaptive`.
- `auto` resolves in `resolve_double_cable_block_solver(...)`: CPU/default
  backends use `thomas`; GPU-like backends use `pcr_adaptive`.
- `pcr_adaptive` is resolved inside batch kernels: `B <= 4096` uses `pcr_soa`,
  larger batches use `pcr`. The array-output `pcr_soa` path currently uses the
  batch-native scan for larger batches and keeps the previous per-fiber route
  for smaller batches until more device evidence exists.
- Pseudo-double modes are benchmark harness candidates only:
  `mrg_single_cable_surrogate`, `pseudo_double_effective`,
  `pseudo_double_single_myelinated_chain`, `pseudo_double_series`,
  `pseudo_double_split`, `pseudo_double_schur_local`, and planned
  `pseudo_double_modal`.

## Immediate Queue

Work should start here unless the user asks otherwise.

- [ ] Phase 7.6.3: finish the exact double-cable GPU solver optimization pass.
- [ ] During Phase 7.6.3, prioritize substantive solver implementations
  from the exact-GPU roadmap over small heuristic retuning. Record heuristic
  thresholds as benchmark-backed follow-up calibration, not as the main work.
- [ ] Phase 7.6.3 current Pallas retry: run the bounded-SMEM
  `pallas_thomas_4` Kaggle P100 focus once committed. `pallas_thomas_16`
  failed during Mosaic GPU lowering with `smem_bytes=60424 >
  max_smem_bytes=49152` before timing, so keep `16` and `128` in standby.
  After this retry, return to Pallas PCR/hybrid or exact `pcr_soa`
  optimization rather than more Thomas block-size tuning unless the timing is
  surprisingly strong.
- [ ] Phase 7.6.3 next implementation target: optimize the existing
  batch-native `pcr_soa` stage body. The 2026-06-17 P100 trace shows SoA cuts
  matrix-PCR device kernel events from `31-48` to `7-13`; remaining hot kernels
  are `loop_select_subtract_fusion_*`, so prioritize reducing per-stage
  `where`/boundary-mask/gather work over more heuristic threshold tuning.
- [x] Run Kaggle P100 `linear_pcr_soa_nomask_focus` to validate the
  benchmark-only `pcr_soa_nomask` and `pcr_soa_shift` candidates against
  `pcr_soa` on GPU. Result: `pcr_soa_nomask` was effectively neutral
  (`2/4` wins, geomean `1.001x` runtime vs `pcr_soa`), while `pcr_soa_shift`
  was slower in all cases (`1.786x` geomean runtime vs `pcr_soa`). Do not
  route either candidate through `auto`; keep `shift` closed/standby.
- [x] Re-run the exact Thomas-family/associative sweep on Kaggle P100 after
  the JAX update. Result: `assoc_backward` still beats `thomas_batched`
  cleanly, but only beats `pcr_soa` in `1/9` cases under JAX `0.10.2`; keep it
  benchmark-only/standby rather than routing it through `auto`.
- [x] Phase 1.5 split iterative solver: validate `split_gs_3` in an
  end-to-end/physiology harness, with `split_gs_4` as the stricter residual
  fallback, before considering any routing change. Result: fixed-K
  `split_gs_3`/`split_gs_4` failed local E2E trace agreement and are in
  abandoned/closed status.
- [x] Abandon split iterative double-cable solver approaches for the current
  optimization pass. Do not spend more Kaggle runs or implementation time on
  split Jacobi/Gauss-Seidel/Richardson variants unless the user explicitly
  reopens the line.
- [x] Keep pseudo-double/pseudo-MRG on standby until exact-solver work exposes a
  clear need for approximate screening again.
- [ ] Phase 7.7: clean stimulation and placement APIs before Phase 8.
- [ ] Phase 7.8: clean examples after API cleanup and add a solver-options
  example only once the public contract is stable.
- [ ] Phase 8: add callable studies, reuse policies, retention policies, and
  study result containers.
- [ ] Phase 9: finalize serialization schemas and add reference-backend
  validation.
- [ ] Keep current Phase 5-7.6 changes uncommitted until the user asks for a
  commit or an explicit checkpoint requires it.

## Guidelines Comparison

The active roadmap should keep matching these `GUIDELINES.md` constraints:

- AxonScope owns 1D axon dynamics, stimulation along axons, execution,
  recording, analyses, validation, and performance.
- External field/geometry packages should provide spatial extracellular
  footprints; AxonScope combines footprints with temporal stimuli.
- Solver options are execution knobs, not new public biological models.
- Exact double-cable remains the reference for MRG-like behavior.
- Pseudo-double work must not create a parallel public architecture, silently
  reinterpret `double`, or become part of `auto`.
- Avoid legacy shims: once an API is replaced, update tests/docs/examples and
  remove the superseded path.

## Phase 7.6 Completed Evidence

Detailed evidence lives in `benchmark/results/`, hotpath manifests, tests, and
git history. Keep this section compact.

- [x] Phase 7.6.1: added path comparison, typed footprint/drive, solver-only,
  recording-policy, observer-policy, cold/warm, and JAX compile/profiler
  benchmark coverage.
- [x] Phase 7.6.2: reduced retained-output pressure with sparse current clamp,
  zero-field and zero-Iinj paths, compact observer results, runtime caches,
  shared double-cable coefficients, chunked runs, and prepared-input timing.
- [x] Double-cable observer-only batch runs now return compact observations for
  homogeneous MRG-like groups with `Recording.none()`.
- [x] PCR and SoA PCR variants match Thomas in current unit tests and are
  available behind explicit solver options.

## Phase 7.6.3 Exact Double-Cable GPU Solver Optimization

Status: active.

Goal: improve exact double-cable GPU performance without replacing the model
with pseudo-single-cable approximations.

Primary roadmap:

- `ideas/axonscope_double_cable_exact_gpu_solver_roadmap.md`

Current solver option contract:

| Option | Current role | Intended use |
| --- | --- | --- |
| `auto` | Backend-aware default. CPU/default -> `thomas`; GPU -> `pcr_adaptive`. | Normal benchmark and production-oriented default for current code. |
| `thomas` | Exact block-Thomas scan. | CPU/default fallback and correctness reference. |
| `pcr` | Exact matrix-layout PCR variant. | GPU diagnostic and larger-batch fallback inside adaptive PCR. |
| `pcr_soa` | Exact struct-of-arrays PCR variant. | GPU diagnostic for small/medium batches. |
| `pcr_adaptive` | Current GPU policy: SoA up to `B=4096`, matrix-layout PCR above. | Explicit reproduction of current GPU `auto` behavior. |

Near-term tasks:

- [x] Confirm the current code only exposes the five solver choices above.
- [x] Keep planned names from the GPU roadmap out of user-facing docs until
  they are implemented and tested.
- [ ] Re-run Colab `kernel_double_cable_observer_auto_long` after
  `pcr_adaptive` to confirm the combined target: keep the SoA gains through
  `n=600` without the `n=2000` regression.
- [x] Add a solver-only exact double-cable benchmark focused on linear-solver
  throughput, separate from dispatch/input/result packaging.
- [x] Add a small JAX trace script or hotpath preset for exact double-cable
  linear solvers, with Thomas, `pcr`, `pcr_soa`, and `pcr_adaptive`.
- [x] Add Kaggle P100 `linear_pcr_soa_trace` preset for focused GPU
  `jax.profiler` traces of `pcr`, `pcr_soa`, and `pcr_adaptive`.
- [x] Run Kaggle P100 `linear_pcr_soa_trace` and inspect GPU trace output.
  Result: `pcr_soa` is `1.09x-1.38x` faster than matrix-layout `pcr` on the
  focused `B=2048/4096`, `Nx=51/96`, `float32` cases; trace evidence says the
  useful optimization target is inside PCR_SOA stage fusion/masking.
- [x] Add benchmark-only `pcr_soa_nomask` and `pcr_soa_shift` candidates for
  PCR_SOA stage optimization. `pcr_soa_nomask` removes explicit boundary
  `where` masks; `pcr_soa_shift` also replaces clamped neighbor gathers with
  static slice/concat shifts. Local targeted tests passed; P100 timing still
  needs validation. Local HLO smoke at `B=8`, `Nx=13` reduced
  `pcr_soa_shift` gather/select counts from `104/105` to `0/0`, replacing
  them with static slices/concats.
- [x] Validate `pcr_soa_nomask` and `pcr_soa_shift` on Kaggle P100. Decision:
  local HLO simplification did not translate to GPU speed. `pcr_soa_nomask`
  is too neutral to justify production routing; `pcr_soa_shift` should not get
  more time unless a future trace shows concat/slice fusion has changed.
- [x] Add an end-to-end exact double-cable batch-kernel benchmark for
  recording/Iinj pressure before GPU reruns.
- [ ] Decide whether to keep the current Literal-based solver option or promote
  it to a typed enum after the option set stabilizes.
- [x] Add batch-native PCR_SOA and route it through array-output
  `DoubleCableBatchKernel` chunks where current evidence supports it.
- [ ] Add and benchmark `Nx` padding buckets as a real solver candidate before
  further threshold/heuristic tuning.
  - [x] Add exact identity-row padding helpers and a benchmark-only
    `pcr_soa_padded` candidate.
  - [x] Local smoke passed on 2026-06-16 for `B=2`, `Nx=45/89`, `float32`;
    `pcr_soa_padded` matched the Thomas64 reference with max absolute error
    about `7.8e-08`. Local CPU timing was slower at this tiny batch and is not
    used as GPU performance evidence.
  - [x] Kaggle P100 `20260616_220653_linear_NvidiaTeslaP100`: padded matched
    Thomas64 within `~1.4e-07` max abs error but was not a general speed win
    versus unpadded `pcr_soa` (`6/20` wins, geomean `1.086x` slower). Keep it
    benchmark-only/standby; do not route it into `auto`.
- [ ] Add and benchmark batch-native exact Thomas as a real solver candidate.
  - [x] Add `solve_block_tridiagonal_2x2_scalar_batched(...)` and
    benchmark-only `thomas_batched`.
  - [x] Local smoke passed on 2026-06-16 for `B=2`, `Nx=45/89`, `float32`;
    `thomas_batched` matched the Thomas64 reference with max absolute error
    about `4.6e-08`. Local CPU timing was slightly faster but is not GPU
    performance evidence.
  - [x] Kaggle P100 `20260616_222231_linear_NvidiaTeslaP100`: numerically
    matched Thomas64 within `~1.4e-07` max abs error, but was not a steady-state
    GPU win versus current vmapped `thomas` (`8/20` wins, geomean `1.009x`
    slower). Compile time improved (`0.885x` geomean), but runtime does not
    justify routing it into `auto`; keep benchmark-only/standby.
- [ ] Add and benchmark transposed-layout exact PCR_SOA as a real solver
  candidate.
  - [x] Add `solve_block_tridiagonal_2x2_pcr_soa_batched_transposed(...)` and
    benchmark-only `pcr_soa_transposed`.
  - [x] Local smoke passed on 2026-06-16 for `B=2`, `Nx=45/89`, `float32`;
    `pcr_soa_transposed` matched the Thomas64 reference with max absolute error
    about `7.8e-08`. Local CPU timing was faster than batch-first `pcr_soa`,
    but the routing decision needs P100 evidence.
  - [x] Kaggle P100 recovered output under `benchmark/results/kaggle/linear`
    from the `20260616_223754_linear_NvidiaTeslaP100` run: transposed matched
    Thomas64 within `~1.4e-07` max abs error but was not a general speed win
    versus batch-first `pcr_soa` (`8/20` wins, geomean `1.047x` slower).
    Keep it benchmark-only/standby; do not route it into `auto`.
- [x] Phase 1E: add and benchmark exact hybrid PCR/Thomas candidates.
  - [x] Add `solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(...)` and
    benchmark-only `pcr_soa_hybrid_4`, `pcr_soa_hybrid_8`, and
    `pcr_soa_hybrid_16`.
  - [x] Local smoke passed on 2026-06-16 for `B=2`, `Nx=45/89`, `float32`;
    all hybrid variants matched the Thomas64 reference with max absolute error
    about `7.8e-08`. CPU/local timing was mixed (`hybrid_4` faster at `Nx=89`,
    slower at `Nx=45`); use P100 evidence for go/no-go.
  - [x] Kaggle P100 `20260616_225915_linear_NvidiaTeslaP100`: all hybrid
    variants matched Thomas64 within `~1.4e-07` max abs error but were much
    slower than batch-native `pcr_soa`. `hybrid_4` won `0/20` cases and was
    `3.405x` slower geomean; `hybrid_8` was `3.828x` slower; `hybrid_16` was
    `4.274x` slower. Decision: keep all hybrid variants benchmark-only/standby;
    do not route them into `auto`.
- [ ] Evaluate optimized Thomas, PCR hybrid, associative scans, split iterative,
  and Pallas only in the sequence described by the exact-GPU roadmap.
  - [x] Add Phase 1.5 split-system helpers, scalar batched tridiagonal solve,
    fixed-K Jacobi/Gauss-Seidel/Richardson candidates, and residual metric.
  - [x] Keep split candidates benchmark-only as `split_jacobi_4`,
    `split_jacobi_8`, `split_gs_4`, `split_gs_8`, and `split_richardson_4`.
  - [x] Local split smoke passed on 2026-06-16 for `B=2`, `Nx=45/89`,
    `float32`. `split_jacobi_8`, `split_gs_4`, and `split_gs_8` reached
    residual/error near the exact `pcr_soa` baseline (`~1e-7`); `split_jacobi_4`
    stayed around `~6e-6` residual; `split_richardson_4` was not acceptable
    locally (`~1e-3` residual). Treat local timing as CPU-only smoke, not GPU
    evidence.
  - [x] Kaggle P100 `20260616_233228_linear_NvidiaTeslaP100`: `split_gs_4`
    matched exact-solver residual/error levels (`max_residual ~2.1e-7`), won
    `13/20` cases versus `pcr_soa`, and was `0.893x` geomean runtime overall
    (`0.687x` for `B>=2048`, winning `8/8`). Decision: keep `split_gs_4` as
    the only clean Phase 1.5 performance candidate for E2E/physiology
    validation.
  - [x] Kaggle P100 `20260616_233228_linear_NvidiaTeslaP100`: `split_jacobi_4`
    was faster (`16/20` wins, `0.818x` geomean runtime vs `pcr_soa`) but
    remained approximate (`max_residual ~7.1e-6`, max_abs error `~2.5e-6`).
    Keep it benchmark-only as a possible approximate/physiology experiment,
    not an exact-solver routing candidate.
  - [x] Kaggle P100 `20260616_233228_linear_NvidiaTeslaP100`: `split_jacobi_8`
    and `split_gs_8` matched exact residual/error levels but were slower than
    `pcr_soa` geomean; `split_richardson_4` was faster in some cases but
    failed the residual/error bar (`max_residual ~1.1e-3`). Keep these in
    standby.
  - [x] Add benchmark-only follow-up candidates `split_gs_2`, `split_gs_3`,
    and `split_jacobi4_gs1` to test lower-K Gauss-Seidel and a Jacobi-plus-GS
    cleanup pass before moving to E2E.
  - [x] Local split follow-up smoke passed on 2026-06-16 for `B=2`,
    `Nx=45/89`, `float32`. `split_gs_3` and `split_jacobi4_gs1` reached
    residual/error near the exact baseline (`~4e-7`) while running faster than
    local `split_gs_4`; `split_gs_2` was too approximate locally
    (`~5e-5` residual). Treat local timing as smoke only.
  - [x] Kaggle P100 `20260616_235328_linear_split_focus_NvidiaTeslaP100`:
    `split_gs_3` was the best clean follow-up candidate. It kept
    `max_residual ~6.5e-7` and `max_abs_error ~3.1e-7`, won `11/12` cases
    versus `pcr_soa`, and was `0.648x` geomean runtime versus `pcr_soa`.
    It also beat `split_gs_4` in `12/12` cases (`0.818x` geomean runtime).
  - [x] Kaggle P100 `20260616_235328_linear_split_focus_NvidiaTeslaP100`:
    `split_jacobi4_gs1` had the same residual/error level as `split_gs_3` but
    was slower (`0.839x` geomean runtime vs `pcr_soa`, `1.060x` vs
    `split_gs_4`). `split_gs_2` and `split_jacobi_4` were faster but too
    approximate (`~6e-5` and `~7e-6` max residual). Keep them benchmark-only
    and out of exact-routing decisions.
  - [x] Add a benchmark-only E2E path for `split_gs_3`, with `split_gs_4` as
    the stricter residual fallback. This is wired through an internal kernel
    override and does not expose split solvers through `BatchOptions` or `auto`.
  - [x] Local E2E smoke passed on 2026-06-17 for `B=2`, actual `Nx=45`,
    `Nt=3`, `recording=center`, `Iinj=none`, and solvers `pcr_adaptive`,
    `split_gs_3`, `split_gs_4`. Local CPU/JIT timing is not performance
    evidence.
  - [x] Kaggle P100 `20260617_105250_e2e_split_focus_NvidiaTeslaP100`:
    `split_gs_3` kept the solver-only signal in the real array-output E2E
    kernel. Versus `pcr_adaptive`, it won `5/6` cases on median kernel time
    (`1.39x` geomean speedup), `4/4` for `B>=2048` (`1.66x`), and `2/2` for
    `B>=2048` with actual `Nx=89` (`1.94x`). `split_gs_3` also beat
    `split_gs_4` in `6/6` kernel cases (`1.26x` geomean speedup).
  - [x] Kaggle P100 `20260617_105250_e2e_split_focus_NvidiaTeslaP100`:
    full total-with-input gains were much smaller for large batches because
    dense `Vext` materialization dominated. For `B>=2048`, `split_gs_3`
    improved `total_with_inputs_ms` only `~1.05x` geomean overall, and
    `~1.09x` for actual `Nx=89`. Decision at this point: the solver timing
    win was real, but it still needed output/physiology agreement proof before
    any routing change.
  - [x] Add Phase 2A benchmark-only exact `assoc_backward`: same Thomas forward
    elimination as `thomas_batched`, with associative affine scan for backward
    substitution. Keep it out of `BatchOptions` and `auto` until P100 evidence
    exists.
  - [x] Local Phase 2A smoke passed on 2026-06-17 for `B=2`, `Nx=45/89`,
    `float32`, and solvers `thomas`, `thomas_batched`, `assoc_backward`,
    `pcr_soa`. `assoc_backward` matched Thomas64 with max absolute error
    about `5.0e-08` and max residual about `9.3e-08`. Local CPU timing is not
    GPU performance evidence.
  - [x] Add Phase 2B dense-transfer associative prototype as
    `assoc_transfer_dense` for diagnostics only. It matches Thomas on
    well-conditioned artificial systems, but benchmark-like float32 systems are
    numerically unstable due transfer-matrix amplification. Keep it out of the
    Kaggle focus and standby unless a stabilized formulation is derived.
  - [x] Kaggle P100 `20260617_112515_linear_assoc_focus_NvidiaTeslaP100`:
    `assoc_backward` matched Thomas64 cleanly (`max_abs_error ~1.0e-7`,
    `max_residual ~1.3e-7`) and beat `thomas` in `9/9` cases (`1.42x`
    geomean speedup), so the associative backward pass works as a Thomas
    optimization. It did not beat current exact `pcr_soa` overall (`3/9` wins,
    `1.313x` geomean runtime vs `pcr_soa`), although it won all `B=4096` cases
    (`1.26x` geomean speedup vs `pcr_soa`). Decision: keep benchmark-only/
    standby; do not route into `auto` unless future workloads specifically
    need an exact large-batch Thomas-family fallback.
  - [x] Kaggle P100 `20260618_182820_linear_assoc_focus_NvidiaTeslaP100`
    retested the same exact candidates after the JAX upgrade. Kaggle installed
    `jax==0.10.2`/`jaxlib==0.10.2`. `assoc_backward` still beat
    `thomas_batched` in `9/9` cases (`1.385x` geomean speedup), but only beat
    `pcr_soa` in `1/9` cases (`B=4096`, `Nx=96`, by about `1.005x`) and was
    `1.570x` geomean runtime versus `pcr_soa`. Decision confirmed: no routing
    change; PCR_SOA remains the best exact JAX backend for these P100 cases.
  - [x] Add Phase 3A Pallas spike `pallas_thomas_128` as a benchmark-only
    exact Thomas-family candidate. It runs one Pallas program per `128` fibers
    over the full `Nx`, requires `B` divisible by `128`, and stays out of
    `BatchOptions`/`auto`.
  - [x] Add `pallas_thomas_16` as a bounded-SMEM Phase 3A retry. Local
    `interpret=True` smoke passed for `B=16`, `Nx=8`, `float32`, but the P100
    run `20260618_183720_linear_pallas_focus_NvidiaTeslaP100` failed during
    Mosaic GPU lowering at `B=1024`, `Nx=51` with `smem_bytes=60424 >
    max_smem_bytes=49152`. Decision: keep `pallas_thomas_16` in standby.
  - [ ] Run Kaggle P100 `linear_pallas_focus` with `pallas_thomas_4`.
    `pallas_thomas_8` is available as an intermediate probe, but it is likely
    too close to the P100 SMEM ceiling at `Nx=96`; use it only if `4` is
    correct but too slow to be informative. Local smoke
    `local_pallas_blocks_smoke` passed for `pallas_thomas_4/8/16` at `B=16`,
    `Nx=8`, with max error `4.575e-08` vs Thomas64. First P100 attempt
    `20260618_184529_linear_pallas_focus_NvidiaTeslaP100` passed the SMEM
    limit but failed on Mosaic's 128-byte gmem-to-smem transfer alignment
    requirement (`4 * 51 * 4 = 816` bytes). Internal Pallas block specs now pad
    storage lengths to multiples of 8 and slice outputs back to real `Nx`;
    local padded smoke `local_pallas_padded_smoke` matched Thomas64 for
    `pallas_thomas_4/8/16` at `B=16`, `Nx=51` with max error `6.493e-08`.
    Second P100 attempt `20260618_185101_linear_pallas_focus_NvidiaTeslaP100`
    then failed on Mosaic layout inference for the artificial
    `jnp.arange`/iota used as batch indices. The kernel now uses explicit
    `pl.ds(row, 1)` / `pl.ds(component, 1)` slices for scratch/output
    load-store helpers; local `local_pallas_ds_smoke` still matches Thomas64.
  - [x] Local Pallas smoke passed on 2026-06-17 for `B=128`, `Nx=16`,
    `float32`, and solvers `thomas`, `thomas_batched`, `assoc_backward`,
    `pallas_thomas_128`, `pcr_soa`. `pallas_thomas_128` matched Thomas64 with
    max absolute error about `5.9e-08` and max residual about `1.2e-07`.
    Local execution used Pallas `interpret=True` on CPU, so timing is not GPU
    performance evidence.
  - [x] Run Kaggle P100 `linear_pallas_focus` and decide whether Pallas Thomas
    justifies any Phase 3B PCR/hybrid work.
    - [x] First Kaggle P100 attempt
      `20260617_114922_linear_pallas_focus_NvidiaTeslaP100` failed before
      measuring Pallas because Kaggle uses `jax 0.7.2`, where
      `jax.experimental.pallas.MemoryRef` is no longer public. Add a small
      compatibility shim that falls back to `jax._src.pallas.core.MemoryRef`;
      rerun required.
    - [x] Second Kaggle P100 attempt
      `20260617_115323_linear_pallas_focus_NvidiaTeslaP100` reached the
      fallback `MemoryRef`, but JAX `0.7.2` uses a two-argument
      `(shape, dtype)` signature instead of the older three-argument
      `(shape, dtype, memory_space)` form. Make the shim accept both
      signatures; rerun required.
    - [x] Third Kaggle P100 attempt
      `20260617_115814_linear_pallas_focus_NvidiaTeslaP100` reached scratch
      creation, then failed because Kaggle/JAX `0.7.2` exposes
      `jax.experimental.pallas.triton.CompilerParams` rather than the older
      `TritonCompilerParams`. Remove explicit Triton compiler params for this
      spike and let Pallas choose defaults; rerun required.
    - [x] Fourth Kaggle P100 attempt
      `20260617_120625_linear_pallas_focus_NvidiaTeslaP100` reached Pallas
      kernel tracing, then failed on direct scratch writes
      `scratch_ref[:, row, k] = ...` with a `swap.abstract_eval` error in
      JAX `0.7.2`. Replace scratch/output direct indexing with explicit
      `pl.store`/`pl.load`; rerun required.
    - [x] Fifth Kaggle P100 attempt
      `20260617_121201_linear_pallas_focus_NvidiaTeslaP100` still failed
      during Pallas tracing because scratch refs in JAX `0.7.2` do not expose
      `.shape`. Stop deriving vector indices from `scratch_ref.shape`; compute
      batch indices once from the input block shape and reuse them for
      `pl.store`/`pl.load`. This is the last compatibility patch before
      putting the Pallas spike in standby if Kaggle still fails.
    - [x] Sixth Kaggle P100 attempt
      `20260617_121708_linear_pallas_focus_NvidiaTeslaP100` still failed before
      measuring Pallas. It reached explicit `pl.store`, then failed inside
      Pallas `swap` abstract evaluation with `IndexError: tuple index out of
      range`. Decision: put Phase 3A `pallas_thomas_128` in standby and do not
      spend more Kaggle runs on Pallas until the kernel is rewritten against
      the current JAX/Pallas indexing API in a controlled environment.
    - [x] Local JAX upgrade retest on 2026-06-17: environment is Python
      `3.12.13`, `jax==0.10.1`, `jaxlib==0.10.1`, CPU backend. Updated the
      Pallas shim for the current API: `pl.load`/`pl.store` are no longer
      public exports, and `MemoryRef` now needs a shaped abstract value through
      `MemorySpace.ANY(shape, dtype)`/`ShapedArray`. Local `interpret=True`
      Pallas smoke passes against Thomas with max error about `6e-08`; the
      small solver benchmark passes for `B=128`, `Nx=16`, `float32`.
      Non-interpreted Pallas still requires a real GPU backend; CPU reports
      `Only interpret mode is supported on CPU backend`.
    - [x] Seventh Kaggle P100 attempt
      `20260617_211635_linear_pallas_focus_NvidiaTeslaP100` reached a real
      P100 backend with Python 3.12 and `jax/jaxlib==0.10.1`, then failed
      before benchmarking because Kaggle kept a preinstalled
      `jax_cuda12_plugin==0.7.2` incompatible with `jaxlib==0.10.1`
      (`PJRT_FFI_UserData_Add_Args size` mismatch). Fix the Kaggle wrapper to
      install the matching JAX CUDA 12 extra after project install:
      `jax[cuda12]==<installed jax version>`.
    - [x] Eighth Kaggle P100 attempt
      `20260617_212151_linear_pallas_focus_NvidiaTeslaP100` confirmed the CUDA
      plugin fix: Kaggle installed `jax-cuda12-plugin==0.10.1`, selected the
      P100 GPU backend, and measured `thomas`, `thomas_batched`, and
      `assoc_backward` for the first case. It then failed during Pallas
      lowering with `Unsupported memory space: any`. Fix the Pallas scratch ref
      helper to prefer `MemorySpace.DEFAULT` and keep `ANY` only as fallback.
    - [x] Ninth Kaggle P100 attempt
      `20260617_212605_linear_pallas_focus_NvidiaTeslaP100` showed
      `MemorySpace.DEFAULT` lowers to `gmem`, which Mosaic GPU also rejects for
      scratch allocation. Fix the GPU path to allocate scratch with
      `jax.experimental.pallas.mosaic_gpu.SMEM(...)`, while keeping the generic
      Pallas memory-ref fallback for local `interpret=True` and older APIs.
    - [x] Tenth Kaggle P100 attempt
      `20260617_213002_linear_pallas_focus_NvidiaTeslaP100` confirmed SMEM is
      the right Mosaic GPU scratch memory space, but the current
      `pallas_thomas_128` design exceeds P100 shared memory
      (`smem_bytes=419848 > max_smem_bytes=49152`) before any Pallas timing is
      recorded. Decision: keep `pallas_thomas_128` benchmark-only/standby; do
      not spend more Kaggle runs on this full-`Nx` scratch design. A future
      Pallas attempt needs a different design, e.g. much smaller block size,
      streaming/recomputed backward coefficients, or a PCR-style kernel with
      bounded scratch.
  - [x] Add output-agreement/physiology validation for `split_gs_3` against
    `pcr_adaptive`/Thomas on held-out double-cable workloads before any public
    solver-option exposure or `auto` routing.
  - [x] Local E2E agreement validation on 2026-06-17 failed fixed-K
    `split_gs_3` and `split_gs_4`: `B=2`, target `Nx=51`, actual `Nx=45`,
    `Nt=3`, `dt=0.05 ms`, `recording=center`, `Iinj=none` diverged from
    `pcr_adaptive` by about `77 mV` and produced false activations at
    `-20 mV`. Exact controls in the same harness (`pcr_soa`/`pcr_adaptive`
    versus `thomas`) were close at about `0.0014 mV` max absolute error.
    Decision: split iterative approaches are abandoned/closed for this
    optimization pass; keep the existing code benchmark-only for historical
    reproducibility until a later cleanup removes failed candidates.
- [ ] Update `auto` only from benchmark evidence; keep resolved choices recorded
  in manifests.
- [ ] Add a didactic advanced solver-options example after the API is stable.
  It should show how to use `auto` and how to force variants for diagnostics,
  not turn examples into timing stress tests.

Example diagnostic command:

```bash
python benchmark/hotpaths/run.py \
  --workload double_cable_observer \
  --sizes 100 300 600 2000 \
  --duration 10.0 \
  --dt 0.01 \
  --compartments 51 \
  --warmups 1 \
  --double-cable-block-solver auto
```

Solver-only diagnostic command:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 128 512 1024 \
  --nx 32 51 64 \
  --solvers thomas pcr pcr_soa pcr_adaptive split_jacobi_4 split_jacobi_8 split_gs_4 split_gs_8 split_richardson_4 \
  --dtypes float32 \
  --warmups 1 \
  --repeats 5
```

Local smoke on 2026-06-16 passed with `B=2`, `Nx=5`, `float32`, `thomas` and
`pcr_soa`; the profiler wrapper produced JAX trace files under
`jax_traces/.../plugins/profile/...`.

Kaggle P100 E2E evidence on 2026-06-16:

- `20260616_205416_e2e_NvidiaTeslaP100`: baseline bounded E2E matrix.
- `20260616_214351_e2e_NvidiaTeslaP100`: batch-native PCR_SOA array-output
  scan. Direct `pcr_adaptive` kernel improved at `B=2048` (`~1.46x` median for
  `Nx=51`, `~1.09x` for `Nx=96`) but regressed at `B=512`; notebook wall time
  improved from about `377s` to `346s`. E2E total remained dominated by
  Vext/setup materialization.

## Phase 7.6.4 Pseudo-Double / Pseudo-MRG Standby

Status: standby.

Decision: the first pseudo-double pass was useful as a validation harness, but
current candidates are not accepted as double-cable replacements. Exact
double-cable optimization now has priority.

Keep:

- [x] `benchmark/pseudo_double/` as an experimental validation harness.
- [x] Exact double-cable reference runs in every pseudo validation workflow.
- [x] JSON/CSV summaries and optional plots for future comparison.
- [x] Unit tests that protect mode parsing, experimental status, output writing,
  plotting, reductions, and validation-only runners.

Do not do now:

- [x] Do not add pseudo-double to `BatchOptions.double_cable_block_solver`.
- [x] Do not make pseudo-double part of `auto`.
- [x] Do not add a public pseudo-double example yet.
- [x] Do not optimize pseudo modes for GPU until a held-out physiology set
  shows credible threshold, activation, propagation, and recruitment behavior.

Resume only if one of these becomes true:

- exact double-cable GPU optimization is insufficient for a real target study;
- pseudo modes are explicitly needed as a high-recall pre-filter before exact
  refinement;
- a new reduction produces substantially better physiology on held-out
  workloads, not only calibrated smoke cases.

When resumed:

- [ ] Start from `pseudo_double_series` and `pseudo_double_schur_local`, because
  they are coefficient-derived rather than pure stimulus-scale probes.
- [ ] Use small deterministic cases plus held-out MRG-like workloads before any
  performance claim.
- [ ] Track activation boolean, threshold amplitude, activation time/location,
  conduction velocity, recruitment ordering, peak Vm, RMS/probe trace error,
  and subthreshold response.
- [ ] Near-threshold or ambiguous pseudo results must be rerun with exact
  double-cable.

## Phase 7.7 Stimulation And Placement API Cleanup

Goal: make the public API match the product boundary before Phase 8 studies.

- [ ] Re-read `GUIDELINES.md` before implementing Phase 7.7 and extract the
  concrete target boundary for stimulation, placement, populations, and study
  inputs.
- [ ] Compare `GUIDELINES.md` against current source, tests, examples, and docs
  before editing public APIs; write the rename/delete checklist here or in a
  short implementation note.
- [ ] If the intended implementation differs from `GUIDELINES.md`, update
  `GUIDELINES.md` first, then align `todo.md` and `agent.md`.
- [ ] Remove remaining public `y` / `z` placement parameters from axon model
  constructors. An `Axon` describes cable, membrane, length, diameter, and
  layout only.
- [ ] Move physical placement to instance, population, or study layers where it
  is still needed.
- [ ] Remove public `intracellular_context` and `extracellular_context`
  terminology from user-facing APIs and examples.
- [ ] Replace generic context methods with explicit domain commands: current
  clamps, point-source electrodes, extracellular drives, footprints,
  stimulation collections, and study inputs.
- [ ] Keep PointSource/electrode/footprint concepts, but make them first-class
  stimulation objects rather than hidden context plumbing.
- [ ] Decide which lower-level internal objects may keep `context` as an
  implementation detail; keep them out of the public facade and examples.
- [ ] Update tests, docs, examples, benchmark workload builders, public API
  tests, `tests/unit/test_examples.py`, and `CHANGELOG.md` in the same pass.
- [ ] Add or update guardrails so removed names do not return as compatibility
  aliases.

## Phase 7.8 Examples Learning Path

Goal: keep examples didactic, runnable, plot-rich where helpful, and aligned
with the cleaned public API.

- [ ] Update examples after the stimulation/placement API cleanup.
- [ ] Add an advanced solver-options example after Phase 7.6.3 stabilizes the
  public contract.
- [ ] Add a pseudo-double example only if Phase 7.6.4 leaves standby with a
  validated physiology harness.
- [ ] Keep examples verbose, line-by-line, and commented near the code being
  taught.
- [ ] Add useful plots for signals, metrics, activation, recruitment, velocity,
  observer outputs, dispatch layouts, or memory comparisons.
- [ ] Avoid turning examples into stress tests; benchmark-heavy evidence stays
  under `benchmark/`.
- [ ] Re-run `tests/unit/test_examples.py` after example edits.

## Phase 8 Studies

Goal: add the public workflow for sweeps, thresholds, recruitment, reuse, and
compact study outputs.

- [ ] Define callable update contract:
  `update(base_simulation: AxonSimulation, condition: Condition) -> AxonSimulation`.
- [ ] Require updates to avoid mutating the base simulation and to make the
  condition explicit.
- [ ] Add sweep API.
- [ ] Add threshold-search API.
- [ ] Add recruitment-sweep API.
- [ ] Add reuse policies for prepared cohorts, compiled kernels, footprints,
  and stimulus-only updates: `AUTO`, `REQUIRE`, `NONE`.
- [ ] Add retention policies so threshold/recruitment studies do not retain
  every trace by default.
- [ ] Add study result containers with compact per-row/per-condition outputs
  and optional retained traces.
- [ ] Document callable reproducibility limits; do not claim arbitrary lambdas
  are serializable.
- [ ] Add didactic advanced examples for each new public study concept.

## Phase 9 Serialization And Reference Backend

Goal: stabilize schemas only after object/result/analysis/study models settle.

- [ ] Define final schemas for simulations, results, and study results.
- [ ] Serialize typed values, identifiers, recording manifests, analysis
  definitions, backend/device/precision, and environment metadata.
- [ ] Do not add readers for prototype formats.
- [ ] Add NumPy reference backend validation for small deterministic cases.
- [ ] Add cross-backend validation before treating serialization as stable.
- [ ] Add final docs only after schemas and reference validation are stable.

## Open Architecture Decisions

- [ ] Decide whether scalar public `simulate(...)` eventually returns
  `AxonSimulationResult` instead of `SimResult`.
- [ ] Replace direct `Recording.to_batch_options()` solver coupling with:
  `Recording -> RecordingPlan -> validation -> backend lowering`.
- [ ] Add backend-neutral axon structure descriptors and cable capability
  descriptors.
- [ ] Extend semantic signals beyond Vm/gates/currents/conductances/states:
  intracellular potential, periaxonal potential, ionic current, and
  cable/role-aware signal availability.
- [ ] Decide whether latency/block-style analyses become direct solver
  observers or thin views over activation observer state.
- [ ] Decide logging policy: Python logging, benchmark traces, warnings,
  result diagnostics, and user-facing summaries.
- [ ] Decide print/Rich/progress policy: defaults, opt-in progress, notebook
  behavior, and CI degradation.
- [x] Decide when Sphinx docs are stable enough to generate after the docs/code
  audit: only after Phase 7.7 API cleanup, `/docs` audit, and public docstring
  coverage are done.

## Documentation Audit

- [ ] Audit `/docs` against current code before writing Sphinx docs.
- [x] Refresh `docs/solver_organization.md` whenever solver options change.
- [x] Split `docs/api_public_draft.md` into implemented API versus proposal if
  it remains part of user-facing docs.
- [x] Refresh `docs/results_recording_analysis.md` for solver-side observer
  execution and trace-free `Recording.none()` results.
- [ ] Re-run NRV validation only in an NRV-ready environment; record dated
  validation notes after a fresh run.
- [x] Keep proposal/roadmap docs clearly labelled so users do not run future
  API snippets as current API.
- [ ] Provide extensive public docstrings before generating API docs.
- [x] Decide what belongs in Sphinx pages versus README versus examples.

Documentation placement decision:

- `README.md`: installation, public API shape, quickstarts, links to canonical
  learning paths, and pre-release status.
- `examples/basic/` and `examples/advanced/`: runnable user workflows and
  didactic concepts; examples stay the executable docs for new public behavior.
- `docs/`: design explanations, current architecture notes, validation notes,
  and proposal/roadmap documents that are clearly labelled.
- Future Sphinx pages: generated API reference and curated stable user guides
  after Phase 7.7 API cleanup, `/docs` audit, and public docstrings.

Current page snapshot:

| Page | Status | Next action |
| --- | --- | --- |
| `docs/axon_model_organization.md` | Partially current | Re-check examples against `src/axonscope/axons/`. |
| `docs/solver_organization.md` | Current for solver options after 2026-06-16 cleanup | Re-check after any new solver mode lands. |
| `docs/membranes.md` | Mostly current | Verify `Composite`, `SectionLayout`, and examples. |
| `docs/stimulation.md` | Mostly current | Re-check after Phase 7.7 API cleanup. |
| `docs/pool_dispatch.md` | Mostly current | Review for overlap with README and API drift. |
| `docs/results_recording_analysis.md` | Current for Phase 7.5+ observer-only behavior | Re-check after new observer kinds or recording lowering changes. |
| `docs/recorders_observers_activation_strategy.md` | Proposal plus current status note | Re-check after Phase 8 study APIs or new observer kinds. |
| `docs/api_public_draft.md` | Proposal with current API snapshot | Re-check after Phase 7.7 API cleanup. |
| `docs/validation.md` | Mostly current | Add dated NRV result only after a fresh NRV-ready run. |

## Benchmark Workstream

Canonical lightweight evidence loop:

- `benchmark/hotpaths/`
- `benchmark/hotpaths/run.py --list`
- `benchmark/hotpaths/README.md`
- `benchmark/hotpaths/COLAB.md`
- `benchmark/hotpaths/colab_gpu_hotpaths.ipynb`

Open benchmark-agent tasks:

- [x] Implement opt-in JAX profiler traces for hotpath runs; traces are
  recorded under `jax_traces/<workload>_n<size>/` and linked from manifests.
- [x] Add explicit cold-start/first-call signature labels.
- [ ] Audit scalar `simulate(...)` instrumentation against pool
  instrumentation and ensure both expose consistent root spans.
- [x] Decide whether `level="minimal"` and `level="detailed"` are worth
  implementing, or keep only `level="hotpaths"` and document that choice.
- [ ] Improve benchmark summaries with percentages of root time, median/p95
  columns, parent names, and enough dimensions to compare runs without
  reopening every `events.jsonl`.
- [x] Add or refresh docs for asynchronous GPU timing, `kernel.enqueue`,
  `kernel.wait`, first-call classification, output files, and JAX trace
  limitations.
- [ ] Add skipped GPU integration tests that verify device metadata and
  `kernel.wait` behavior when a GPU is available.
- [x] Separate correctness validation from performance benchmarking in docs and
  scripts.
- [ ] Clean legacy benchmark assets only after benchmark-agent and bottleneck
  leftovers are closed.

Legacy benchmark areas to classify later:

- `benchmark/runtime/`
- `benchmark/results/runtime/`
- `benchmark/reports/runtime/`
- old standalone runtime scripts
- generated caches
- dated exploratory output files

## API And Units Backlog

- [ ] Stabilize the public API before adding Sphinx.
- [ ] Prefer clean user-facing interfaces over backward compatibility shims.
- [ ] Remove temporary compatibility aliases and old argument names once
  examples/tests/docs use the clean API.
- [ ] When removing or renaming public API, update affected examples in the
  same change and keep `tests/unit/test_examples.py` aligned.
- [ ] Pass units everywhere at public boundaries, including membranes, axons,
  stimulation, protocols, recording, and solver wrappers.
- [ ] Audit membrane constructors for plain-number ambiguity and decide where
  units should be required versus normalized.
- [x] Preserve `SimResult.Vm` as a stable notebook-friendly convenience, with
  explicit errors when observer-only runs do not carry Vm.

Locked compatibility decisions:

- [x] `axs.analysis` is a real package; old forwarding aliases are removed.
- [x] `axs.visualization` remains absent; plotting stays under
  `axs.results.visualization`.
- [x] `axs.run_batch(...)` is removed; use `simulate_pool(...)`.
- [x] Public wrappers use `duration` / `dt` with Pint quantities.
- [x] Direct solvers use solver-level `tsim` / `dt` only.
- [x] Public recording uses `sample_dt` and `positions` with units.
- [x] Public point-source electrodes use quantity-oriented `x`, `y`, `z`, and
  `min_distance`.
- [x] Activation protocol currents require current units.
- [x] Explicit `Stimulus` time inputs require units.
- [x] `clear_extracellular_context(...)` is singular.

## Recording, Observables, And Analysis Backlog

- [ ] Verify single-axon and pool recording behavior for Vm, gates, currents,
  conductances, and state variables across supported groups.
- [x] Keep post-hoc `ActivationCriterion` and `Activation` semantics aligned
  with current solver-side activation observers.
- [ ] Add plotting helpers for batch groups and retained recording layouts.
- [x] Lock current public `Recording` contract with tests: scalar runs require
  Vm and may include observable groups; pool runs support Vm spatial modes only;
  unsupported position/temporal/pool-observable filters raise explicit errors.
- [x] Make batch diagnostics discoverable from per-axon public result views.

## Evidence Ledger

Use this section for compact dated evidence that affects current decisions.
Keep long narrative in benchmark artifacts, not here.

| Date | Evidence | Result / Decision |
| --- | --- | --- |
| 2026-06-14 | Phase 2 final unit + NRV | Unit `263 passed, 1 skipped`; NRV `116 passed, 516 warnings`. |
| 2026-06-14 | Phase 4 final validation | Compileall passed; targeted backend/guardrail run `70 passed`; full unit `286 passed, 1 skipped`; full NRV `116 passed, 516 warnings`; hotpath smoke passed. |
| 2026-06-15 | Compact dispatch cohort cleanup | Full unit `314 passed, 1 skipped`. |
| 2026-06-15 | Single-cable observer hotpath | GPU observer-only `n=1000`, `duration=10 ms`, `dt=0.01 ms`, `Nx=51` improved from `673.4 ms` to `176.8 ms` after sparse current-clamp, zero-field, compact-cohort, and pulse-vectorization cleanups. |
| 2026-06-15 | Double-cable baseline | Matching single-cable GPU scaling was much better than double-cable, showing the bottleneck is kernel-specific rather than generic extracellular preprocessing. |
| 2026-06-15 | Double-cable PCR | `pcr` produced strong GPU wins but severe CPU regressions, so `auto` keeps Thomas on CPU/default and PCR-family solvers on GPU. |
| 2026-06-15 | Double-cable observer-only | Homogeneous double-cable observer-only batch runs returned compact observations with `Vm=None` and greatly reduced result packaging. |
| 2026-06-15 | Double-cable PCR SoA/adaptive | `pcr_soa` improved smaller GPU batches but regressed at `n=2000`; `pcr_adaptive` now selects SoA up to `B=1024` and matrix-layout PCR beyond that. |
| 2026-06-15 | Pseudo-double validation | Harness and candidate modes exist, but candidates remain rough screening probes with trace errors too large for production acceptance. Pseudo-double is standby. |
| 2026-06-16 | Local CPU solver-only baseline | `benchmark/solvers/bench_double_cable_linear_solvers.py` local CPU matrix at `B=8/128/512`, `Nx=32/51/64`, `float32` confirms Thomas is the CPU/default path; PCR variants are slower for production CPU use, with only tiny `B=8,Nx=32` noise favoring matrix PCR. |
| 2026-06-16 | End-to-end double-cable benchmark smoke | `benchmark/solvers/bench_double_cable_end_to_end.py` local smokes passed for center/no-Iinj, observer-only/dense-zero-Iinj, and full/nonzero-Iinj at `B=2`, target `Nx=51`, `Nt=3`; Colab notebook is ready for GPU runs. |
| 2026-06-17 | Split E2E agreement validation | Added `benchmark/solvers/validate_double_cable_solver_agreement.py`; local held-out smoke failed `split_gs_3`/`split_gs_4` with `~77 mV` center-trace error and false activations versus `pcr_adaptive`, while exact PCR controls stayed close to Thomas. Split iterative approaches are abandoned/closed for this optimization pass despite timing wins. |
| 2026-06-17 | JAX 0.10.1 local validation | Environment now uses Python `3.12.13`, `jax==0.10.1`, `jaxlib==0.10.1`; full unit `424 passed, 1 skipped`. Updated Pallas compatibility for JAX 0.10.1 and local `interpret=True` Pallas smoke passes, but GPU lowering still needs Kaggle/Colab validation. |
| 2026-06-17 | Kaggle JAX 0.10.1 Pallas retry setup | P100 run `20260617_211635_linear_pallas_focus_NvidiaTeslaP100` reached GPU backend, then failed before benchmark due stale Kaggle `jax_cuda12_plugin==0.7.2` against `jaxlib==0.10.1`. Kaggle wrapper now installs matching `jax[cuda12]==<installed jax version>` for P100 runs. |
| 2026-06-17 | Kaggle Pallas scratch memory retry | P100 run `20260617_212151_linear_pallas_focus_NvidiaTeslaP100` reached GPU benchmark execution and measured the non-Pallas first case, then Pallas lowering failed on scratch `MemorySpace.ANY`. Pallas scratch refs now prefer `MemorySpace.DEFAULT`; local Pallas smoke still passes (`11 passed`). |
| 2026-06-17 | Kaggle Pallas SMEM scratch retry | P100 run `20260617_212605_linear_pallas_focus_NvidiaTeslaP100` showed `MemorySpace.DEFAULT` becomes unsupported `gmem` scratch under Mosaic GPU. GPU Pallas scratch now uses `mosaic_gpu.SMEM(...)`; local Pallas smoke still passes (`11 passed`). |
| 2026-06-17 | Kaggle PCR_SOA JAX trace | P100 run `20260617_214032_linear_pcr_soa_trace_NvidiaTeslaP100` completed. `pcr_soa` beat matrix-layout `pcr` by `1.09x-1.38x` steady median on focused `B=2048/4096`, `Nx=51/96`, `float32` cases, reducing device fusion events from `31-48` to `7-13`. Remaining hot spots are `loop_select_subtract_fusion_*`, so next work should optimize PCR_SOA stage masking/gather behavior. |
| 2026-06-17 | Kaggle PCR_SOA stage candidates | P100 run `20260617_220929_linear_pcr_soa_nomask_focus_NvidiaTeslaP100` completed. `pcr_soa_nomask` was neutral (`2/4` wins, geomean `1.001x` runtime vs `pcr_soa`); `pcr_soa_shift` was slower in all focused cases (`1.786x` geomean runtime). Do not route these candidates; close `shift` despite the local HLO gather/select reduction. |
| 2026-06-17 | Pallas Thomas 128 standby decision | P100 run `20260617_213002_linear_pallas_focus_NvidiaTeslaP100` reached Mosaic GPU SMEM lowering but exceeded P100 shared memory (`419848 > 49152` bytes). `pallas_thomas_128` remains benchmark-only/standby; future Pallas work needs bounded-scratch redesign rather than more compatibility patches. |
| 2026-06-18 | Kaggle exact assoc retest | P100 run `20260618_182820_linear_assoc_focus_NvidiaTeslaP100` installed JAX `0.10.2` and completed. `assoc_backward` remains a good Thomas-family optimization (`1.385x` geomean speedup vs `thomas_batched`) but not a better general backend than `pcr_soa` (`1/9` wins, `1.570x` geomean runtime vs `pcr_soa`). No `auto` routing change. |
| 2026-06-18 | Pallas Thomas bounded-SMEM retry | P100 run `20260618_183720_linear_pallas_focus_NvidiaTeslaP100` reached Mosaic GPU lowering with JAX `0.10.2` but `pallas_thomas_16` still exceeded shared memory (`60424 > 49152` bytes) before timing. Added benchmark-only `pallas_thomas_4/8`; local smoke `local_pallas_blocks_smoke` matched Thomas64 for `4/8/16`, and the next Kaggle focus uses `pallas_thomas_4`. |
| 2026-06-18 | Pallas Thomas transfer-alignment retry | P100 run `20260618_184529_linear_pallas_focus_NvidiaTeslaP100` showed `pallas_thomas_4` clears the previous SMEM limit but fails Mosaic's gmem-to-smem copy alignment (`816` bytes not divisible by `128`) at `B=1024`, `Nx=51`. Pallas internal block specs now pad main/edge/rhs/output storage lengths to multiples of 8 while preserving real-`Nx` Thomas loops; local padded smoke matched Thomas64 for `pallas_thomas_4/8/16`. |
| 2026-06-18 | Pallas Thomas iota-layout retry | P100 run `20260618_185101_linear_pallas_focus_NvidiaTeslaP100` passed the transfer-alignment fix but failed Mosaic layout inference for the `jnp.arange` batch-index iota. Scratch/output helpers now use `pl.ds(row, 1)` / `pl.ds(component, 1)` slices instead of vectorized batch indices; local `local_pallas_ds_smoke` still matches Thomas64 for `pallas_thomas_4/8/16`. |

## Completed Roadmap Archive

Keep this as a compact map of what has landed. Detailed history lives in git,
tests, examples, and benchmark result folders.

- [x] Phase 0: guardrails, public API cleanup checks, import-boundary checks,
  obsolete benchmark inventory, and non-NRV baseline.
- [x] Phase 1: `AxonInstance`, root `AxonSimulation`, `AxonPopulation`, and
  one/population lifecycle.
- [x] Phase 2: typed public contracts, opaque identifiers, extracellular
  footprints/drives/stimulation, and analytical footprint builders.
- [x] Phase 2.5: opt-in benchmark spans, hotpath workload catalog, and Colab
  GPU workflow.
- [x] Phase 3: preparation signatures and reusable prepared cohorts.
- [x] Phase 4: JAX execution enters through `axonscope.backends.jax`.
- [x] Phase 5: canonical pool results and recording manifests.
- [x] Phase 6: public analysis layer, reports, statuses, and online observers.
- [x] Phase 7: performance estimates and hotpath memory metadata.
- [x] Phase 7.5: solver-side observers for current scalar/batch workflows.

## Cleanup And Sync

- [ ] Do a general cleanup pass after docs, examples, recordings, observers,
  and benchmarks are aligned.
- [ ] Remove stale aliases, removed file references, duplicate docs, and dead
  benchmark/example paths.
- [x] Keep `agent.md` and `todo.md` synchronized after each cleanup step.
- [x] Keep this TODO flat: when a section starts accumulating long narrative,
  move details into docs, benchmark manifests, or a compact evidence-ledger row.
