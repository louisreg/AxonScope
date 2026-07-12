# P11C Solver Path Inventory And Decision Map

Date: 2026-07-09

This is a working inventory for the solver cleanup decision after the P11B and
P11C benchmark gates. It separates:

- production/runtime paths that can be reached through the public API;
- backend-private paths that are only reachable through benchmark overrides;
- benchmark-only diagnostic candidates;
- archived or rejected solver spikes.

The goal is to make the next solver flattening decision explicit before
promoting, renaming, deleting, or archiving any implementation.

## Decision Snapshot

Current recommended reading, updated after the 2026-07-11 policy-cleanup
decision note:

- CPU double-cable production is Thomas only. `auto` on CPU resolves to
  Thomas; CPU PCR, PCR-SoA, tiled-Thomas, and Triton are not production choices.
- GPU double-cable keeps explicit typed solver choices while the full policy
  matrix is finished.
- The looped jax-triton XB route is the preferred large-population GPU
  promotion candidate, but it should become the default only after the policy
  matrix covers `Naxons`, `Nx`, dtype, recording modes, CPU/GPU comparison,
  cold/warm cache, memory, dependency failure modes, and corrected physical
  curve workflows.
- Do not promote PCR micro-variants as-is. They are useful diagnostic evidence,
  not production candidates.
- Do not add any membrane-model-specific solver/runtime path. MRG is a
  realistic benchmark workload, not a runtime branch.
- Cleanup decision ledger:
  `docs/architecture/p11_solver_policy_cleanup_decisions_2026_07_11.md`.

## Boundary Rules

Production solver routes are reachable through:

```text
AxonSimulation(...).run()
    -> axonscope.runtime.execution
    -> axonscope.runtime.jax.group_runner
    -> axonscope.runtime.jax.batch_kernels
```

Benchmark-only solver routes may be used by:

```text
benchmark/analysis/*.py
benchmark/run.py --double-cable-block-solver tiled_thomas ...
benchmark/kaggle/run_kernel.py ...
```

They must not be treated as public API until explicitly promoted.

## Public And Production Surface

### Public solver facade

| Surface | Path | Status | Notes |
| --- | --- | --- | --- |
| `CrankNicholson` | `src/axonscope/solvers/crank_nicholson.py` | removed in P12B | Superseded by `AxonSimulation(...).run()` through the batch route, including `B=1`. |
| `SolverOptions` | `src/axonscope/solvers/options.py` | production | Reserved numerical preparation options. |
| `BatchOptions` | `src/axonscope/solvers/options.py` | production | Batch recording and time chunking only. |
| `BatchRecording` | `src/axonscope/solvers/options.py` | production | Solver-side retained Vm policy: full, center, probes, indices, none. |
| `ExecutionPolicy.solvers` | `src/axonscope/performance.py` | production | Typed per-cable solver policy surface. |

Current typed public double-cable solver vocabulary:

```text
axs.runtime.jax.SingleCableSolver.auto()
axs.runtime.jax.SingleCableSolver.jax_tridiagonal()
axs.runtime.jax.DoubleCableSolver.auto()
axs.runtime.jax.cpu.DoubleCableSolver.thomas()
axs.runtime.jax.gpu.DoubleCableSolver.pcr()
axs.runtime.jax.gpu.DoubleCableSolver.pcr_soa()
axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(...)
```

Current internal artifact label:

```text
jax_triton_loop_xb
```

This label may appear in benchmark metadata, inspection records, and internal
runtime tests. Active curve benchmarks should select the route through the
typed/public-facing `tiled_thomas` solver policy instead of accepting this
implementation label as a CLI choice.

### Public runtime routing

| Route | Main code path | Solver family | Status |
| --- | --- | --- | --- |
| Scalar single-cable | `runtime/jax/execution/scalar_runner.py` -> `runtime/jax/kernels.py` | JAX `lax.linalg.tridiagonal_solve` | removed in P12B |
| Scalar double-cable | `runtime/jax/execution/scalar_runner.py` -> `runtime/jax/kernels.py` | specialized block Thomas | removed in P12B |
| Batch single-cable | `runtime/jax/group_runner.py` -> `runtime/jax/batch_kernels.py` | JAX `lax.linalg.tridiagonal_solve` over batch rows | production |
| Batch double-cable, dense/probe Vm | `group_runner.py` -> `DoubleCableBatchKernel.run(...)` -> `_run_double_cable_batch_array_chunks(...)` | Thomas/PCR/PCR-SoA policy | production |
| Batch double-cable, observer-only VmRaster | `group_runner.py` -> `DoubleCableBatchKernel.run(...)` -> `_run_double_cable_batch_observer_chunks(...)` | Thomas/PCR/PCR-SoA policy | production |
| Batch double-cable, tiled Thomas GPU policy | same as above, with typed `DoubleCableSolver.tiled_thomas(...)` policy | looped jax-triton XB block Thomas | explicit GPU candidate |

Important implementation detail:

- `axonscope.runtime.execution.execution_context(...)` resolves typed
  `ExecutionPolicy.solvers` into backend-private JAX solver-engine descriptors
  when an execution context knows the platform.
- `runtime/jax/group_runner.py` carries that route as one `JaxSolverEngine`
  value into `DoubleCableBatchKernel.run(...)`; the batch kernel no longer
  accepts parallel raw solver-policy arguments.
- Host-side inspection/reporting reads one runtime-level solver-route summary
  from the same policy resolution instead of resolving single- and double-cable
  labels independently.
- `batch_kernels._resolve_double_cable_kernel_block_solver(...)` then resolves
  `pcr_adaptive` to `pcr_soa` for `B <= 4096`, otherwise to `pcr`.
- Batch-native integrated paths are used for `pcr_soa` at sufficient batch size
  and for the internal tiled-Thomas kernel label.
- Other double-cable solvers use a row-wise/vmap style kernel body.

## Active Production Solvers

### Single-cable JAX tridiagonal solve

| Item | Value |
| --- | --- |
| Source | `src/axonscope/runtime/jax/kernels.py`, `src/axonscope/runtime/jax/batch_kernels.py` |
| Core operation | `jax.lax.linalg.tridiagonal_solve` |
| Applies to | Single-cable scalar and batch runs, including imposed extracellular forcing |
| Public selector | none separate from cable model |
| Status | production |

This is not part of the current Triton/PCR decision. No active Triton
single-cable implementation exists. If single-cable custom GPU solving is
opened later, it should be a separate evidence-backed route.

### Double-cable `thomas`

| Item | Value |
| --- | --- |
| Source | `solve_block_tridiagonal_2x2_scalar(...)` in `src/axonscope/runtime/jax/common.py` |
| Algorithm | Exact scalarized 2x2 block Thomas sweep |
| Layout | one axon/system at a time |
| CPU status | only supported production double-cable route; `auto` resolves here |
| GPU status | production selectable, but not preferred by current policy |
| Benchmark reading | Best CPU family; poor GPU at small/medium batch sizes |

Evidence:

- CPU `Naxons=512`, actual `Nx=89`, observer-only, fp32: Thomas is the only
  sensible CPU family in the P11B real-stage evidence.
- GPU `Naxons=512`, actual `Nx=89`: Thomas-family solves are around
  `2.1-2.2 ms`, while PCR/SoA is around `0.44-0.46 ms`.

### Double-cable `pcr`

| Item | Value |
| --- | --- |
| Source | `solve_block_tridiagonal_2x2_pcr(...)` in `src/axonscope/runtime/jax/common.py` |
| Algorithm | Exact matrix-layout parallel cyclic reduction |
| Layout | per-row/vmap style in production batch kernels |
| Runtime role | Explicit GPU choice and large-batch target of `pcr_adaptive` |
| Status | GPU production/benchmark route; not a supported CPU production route |

This remains part of the current production vocabulary, mainly as the
large-batch side of the existing GPU adaptive policy.

### Double-cable `pcr_soa`

| Item | Value |
| --- | --- |
| Source | `solve_block_tridiagonal_2x2_pcr_soa(...)` and `solve_block_tridiagonal_2x2_pcr_soa_batched(...)` in `src/axonscope/runtime/jax/common.py` |
| Algorithm | Exact PCR using struct-of-arrays 2x2 block coefficients |
| Layout | per-row/vmap or batch-first `[B, Nx]` |
| Runtime role | Main current GPU-oriented exact JAX route for small/medium batches |
| Status | GPU production/benchmark route; not a supported CPU production route |

Evidence:

- P11B lowering showed `pcr_soa_batched` and `pcr_soa_vmap` compile to very
  similar optimized HLO. The batch-native route is slightly smaller and is the
  retained active path.
- At `Naxons=512`, actual `Nx=89`, P100 fp32 observer-only, `pcr_soa_batched`
  is around `0.44-0.46 ms` for block solve and close to the fused one-step
  time, making the GPU path solver-sensitive.

### Double-cable `pcr_adaptive`

| Item | Value |
| --- | --- |
| Source | `src/axonscope/runtime/jax/batch_kernels.py` |
| Algorithm | Policy alias, not a solver body |
| Resolution | `pcr_soa` for `B <= 4096`, otherwise `pcr` |
| Runtime role | Current GPU `auto` policy target |
| Status | GPU production policy; never a CPU policy |

This should be treated as policy glue. If Triton is promoted, the policy should
be redesigned from fresh P11C-F evidence instead of patched ad hoc.

## Backend-Private Integrated Candidate

### `jax_triton_loop_xb`

| Item | Value |
| --- | --- |
| Runtime override name | `jax_triton_loop_xb` |
| Source | `src/axonscope/runtime/jax/jax_triton_double_cable.py` and `src/axonscope/runtime/jax/common.py` |
| Runtime hook | `solve_double_cable_linear_system_jax_triton_loop_xb(...)` |
| Kernel path | `solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb(...)` |
| Layout | node-first `[Nx, B]`, one Triton program per axon tile |
| Algorithm | Exact tiled 2x2 block Thomas, looped `tl.range` recurrence |
| Supported target | GPU, fp32, optional `jax-triton`/`triton` dependency |
| Public API status | not public |
| Benchmark status | integrated benchmark-only override |

Why it exists:

- The static `tl.static_range` Triton kernel had very strong warm timing but
  catastrophic first lowering on Kaggle P100: roughly `40-41 min` for real
  `Nx=89`.
- The looped `tl.range` version removed the cold-start cliff: the cold audit at
  `Nx=89`, `B=128` dropped lowering from about `2340 s` to about `3.72 s`.
- Large-population real-stage gates showed strong warm wins:
  - `Naxons=8192`: looped Triton block solve `0.619 ms` vs PCR `3.791 ms`;
    one-step `1.579 ms` vs PCR `4.456 ms`.
  - `Naxons=16384`: looped Triton block solve `0.992 ms` vs PCR `7.172 ms`;
    one-step `2.221 ms` vs PCR `8.223 ms`.
- Integrated workflow gates at `Naxons=8192` showed large timing wins, but the
  first threshold run had an amplitude/geometry issue and was timing-only.
- Corrected threshold smoke at commit `3f60820` validated physical bracketing
  on Kaggle P100:
  - base PCR: `64/64` thresholds ok;
  - Triton override: `64/64` thresholds ok;
  - PCR/Triton threshold max absolute difference: `0.0 uA`.

Current decision:

```text
Promising, but not yet policy.
```

Promotion requires the full P11C-F policy matrix and a clean API decision.

## Benchmark-Only Diagnostic Solvers

These routes are useful for comparison but should not be exposed through
`BatchOptions` as-is.

| Candidate | Source | Status | Main evidence |
| --- | --- | --- | --- |
| `thomas_batched_scan` | `solve_block_tridiagonal_2x2_scalar_batched(...)` in `common.py` | diagnostic | Poor GPU at `Naxons=512`, but becomes useful at very large `Naxons`; kept as reference for large-population gates. |
| `large_population_exact_double_cable_jax` | `benchmark/analysis/large_population_solver.py` | diagnostic/prototype | Validated layout/bucketing ideas, but mixed timing and not enough for production. CPU route should not move here. Moved out of `runtime/jax` during P12B cleanup. |
| `jax_triton_tiled_thomas` | `jax_triton_double_cable.py` | rejected as static-unrolled route | Strong warm timing but unacceptable static-range lowering time at realistic `Nx`. Superseded by looped version. |
| `jax_triton_tiled_thomas_loop` | `jax_triton_double_cable.py` | diagnostic plus integrated candidate source | Strong warm large-population GPU signal; looped XB form is the current integrated benchmark candidate. |
| `pcr_soa_vmap` | benchmark analysis wrappers | diagnostic | Optimized HLO essentially equivalent to batch-native PCR/SoA; no separate runtime route needed. |
| `pcr_soa_symmetric_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | diagnostic | Reduced HLO state and estimated fusion output, but hot solve improved only about `2.6%` at P100 `B=512`. Not enough. |
| `pcr_soa_nomask_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | rejected diagnostic | Lower select count, slower hot runtime. |
| `pcr_soa_shift_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | diagnostic reference | Removed gathers/selects and improved first-run behavior, but hot runtime did not improve. |
| `pcr_soa_transposed_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | rejected diagnostic | Slower hot runtime. |
| `pcr_soa_padded_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | diagnostic reference | Similar hot runtime to baseline within noise; not a clear win. |
| `pcr_soa_hybrid_batched` | `benchmark/analysis/double_cable_solver_candidates.py` | rejected diagnostic | Much larger HLO/fusion surface and much slower. |
| reciprocal inverse rewrite | reverted code, documented in `p11b_reciprocal_inverse_gate_2026_07_07.md` | rejected | Reduced divide count but regressed hot block solve and one-step timing. |

## Archived Or Historical Solver Families

The following directories are historical evidence, not active runtime paths:

```text
benchmark/legacy/pre_p11/solvers/
benchmark/legacy/pre_p11/triton_solver/
benchmark/legacy/pre_p11/jax_triton_solver/
benchmark/legacy/pre_p11/cuda_ffi_solver/
tests/archive/solver_spikes/
```

Archived families include:

- old Triton/Pallas/CUDA FFI spikes;
- split/approximate routes;
- old PCR layout probes and benchmark wrappers;
- validation-failed or toolchain-blocked custom kernels.

They can inform future design, but they should not reappear in active runtime
or public docs without a new benchmark-backed hypothesis.

## Evidence Matrix

### CPU double-cable

| Shape/evidence | Reading |
| --- | --- |
| P11B real-stage `Naxons=512`, actual `Nx=89`, fp32, observer-only | Thomas is the only sensible CPU family. PCR/SoA is orders of magnitude slower on CPU in the recorded diagnostics. |
| P11C CPU synthetic cross-check | Do not move CPU toward the P11C tiled/padded route. It only gives modest wins for short `Nx=47` and loses for larger buckets. |

Decision:

```text
Keep CPU double-cable Thomas-only: auto -> thomas, with explicit thomas allowed.
Treat non-Thomas CPU double-cable routes as unsupported/invalid.
```

### GPU small/medium double-cable

| Shape/evidence | Reading |
| --- | --- |
| P11B real-stage `Naxons=512`, actual `Nx=89` | PCR/SoA is clearly better than Thomas-family JAX scans. |
| PCR/SoA lowering audit | HLO has large PCR fusion tuples, but simple micro-variants do not produce enough hot-time gain. |
| Reciprocal inverse rewrite | Instruction-count cleanup alone is not a useful optimization target. |

Decision:

```text
Keep current JAX PCR policy for production.
Do not promote PCR micro-variants.
```

### GPU large-population double-cable

| Shape/evidence | Reading |
| --- | --- |
| Synthetic `Naxons=8192/16384` | JAX `thomas_batched_scan` starts beating current PCR/SoA in solver-only regimes. |
| Real-stage `Naxons=8192/16384` | The large-Naxons Thomas signal survives realistic prepared double-cable inputs. |
| Static jax-triton | Best warm timing, unacceptable cold lowering for static unrolled `Nx`. |
| Looped jax-triton | Keeps warm speedup and removes cold-start cliff. |
| Integrated workflow `Naxons=8192` | Strong curve-level timing improvement, but policy still needs a broader matrix. |
| Corrected threshold smoke `Naxons=64` | Physical threshold setup now brackets, and PCR/Triton thresholds match exactly on the smoke. |

Decision:

```text
Continue P11C-F. The candidate is worth full policy benchmarking.
Do not make it public/default yet.
```

## Source Path Map

### Runtime/public path

```text
src/axonscope/simulation.py
src/axonscope/runtime/execution.py
src/axonscope/runtime/jax/group_runner.py
src/axonscope/runtime/jax/batch_kernels.py
```

### Production solver implementation path

```text
src/axonscope/solvers/options.py
src/axonscope/runtime/jax/common.py
src/axonscope/runtime/jax/batch_kernels.py
```

### Backend-private P11C implementation path

```text
src/axonscope/runtime/jax/jax_triton_double_cable.py
benchmark/analysis/large_population_solver.py
```

### Benchmark entry points

```text
benchmark/run.py
benchmark/curves/threshold_curves.py
benchmark/curves/recruitment_curves.py
benchmark/workloads/curve_options.py
benchmark/workloads/curve_runtime.py
benchmark/kaggle/run_kernel.py
```

### Low-level analysis tools

```text
benchmark/analysis/double_cable_real_stage_profile.py
benchmark/analysis/large_population_double_cable_solver_profile.py
benchmark/analysis/double_cable_solver_stage_profile.py
benchmark/analysis/double_cable_solver_lowering_audit.py
benchmark/analysis/pcr_soa_stage_state_audit.py
benchmark/analysis/jax_triton_cold_start_audit.py
benchmark/analysis/hlo_fusion_summary.py
```

## Flattening Questions For The Next Decision

Before changing public policy:

1. Should the production double-cable solver implementation be split into a
   clearer module namespace, for example:

   ```text
   runtime/jax/solvers/single_cable.py
   runtime/jax/solvers/double_cable_thomas.py
   runtime/jax/solvers/double_cable_pcr.py
   runtime/jax/solvers/double_cable_triton.py
   ```

2. Should benchmark-only PCR probes move out of `common.py` into
   `benchmark/analysis` or a clearly marked runtime-private reference/probe
   area?

3. If Triton is promoted, what is the clean public name?

   Candidate names should describe the route without exposing too much
   implementation detail:

   ```text
   triton_thomas
   triton_xb
   tiled_thomas
   gpu_tiled_thomas
   ```

   The current benchmark name `jax_triton_loop_xb` is useful as an internal
   artifact label but too implementation-specific for a stable public option.

4. Should public selection be shape-specific policy only, explicit user option
   only, or both?

   A conservative path is:

   ```text
   public selectable option first
   -> full benchmark matrix
   -> auto policy only for proven shapes
   ```

5. What are the minimum promotion gates?

   Suggested gates:

   - exactness against current route on real assembled systems;
   - corrected threshold curves and recruitment curves;
   - `Naxons` sweep from small to large;
   - `Nx` sweep across bucket boundaries;
   - fp32 primary and fp64 spot checks where supported;
   - observer-only, probe Vm, and full Vm where feasible;
   - cold and warm compile/cache behavior;
   - RSS/device memory;
   - no silent fallback when optional Triton dependencies or GPU support are
     unavailable.

## Suggested Next Cleanup Order

1. Finish P11C-F as a policy benchmark, not as another implementation pass.
2. Keep `jax_triton_loop_xb` as an internal artifact/runtime label while
   active curve benchmarks use the typed `tiled_thomas` policy.
3. Move or label benchmark-only solver probes so production code is easier to
   read.
4. If P11C-F supports promotion, add a named backend-private production route
   and only then expose a public selectable option.
5. Decide `auto` policy last.
