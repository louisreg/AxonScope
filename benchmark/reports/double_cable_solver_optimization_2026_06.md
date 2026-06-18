# Double-Cable Solver Optimization Summary, June 2026

This report closes the June 2026 double-cable solver optimization pass. The
short version: keep the public solver surface unchanged, archive the custom
kernel spikes, and move the next performance work to `Vext` materialization and
workflow-level benchmarks.

![Solver speedup summary](double_cable_solver_optimization_2026_06_speedups.svg)

## Final Decision

Retained active solver routes:

- `auto`
- `thomas`
- `pcr`
- `pcr_soa`
- `pcr_adaptive`

No new GPU full-batch double-cable solver is routed by default. The best isolated
custom-kernel signals were real, especially Triton block Thomas, but none cleared
the integration and validation gates required for production AxonScope use.

## Evidence Table

| Candidate | Main evidence | Performance result | Validation / risk | Decision |
|---|---:|---:|---|---|
| `pcr_soa` / `pcr_adaptive` | baseline retained route | `1.00x` reference | current public route | Keep |
| `pcr_soa_nomask` | P100 `20260617_220929_linear_pcr_soa_nomask_focus_NvidiaTeslaP100` | `1.001x` runtime vs `pcr_soa`, `2/4` wins | algebra unchanged, no useful gain | Archive |
| `pcr_soa_shift` | same P100 run | `1.786x` runtime vs `pcr_soa`, `0/4` wins | slower despite simpler HLO | Archive |
| `pcr_soa_layout_auto` | P100 `20260618_202917_linear_pcr_soa_layout_focus_NvidiaTeslaP100` | `1.021x` runtime vs `pcr_soa`, `2/6` wins | layouts identical to baseline | Archive |
| `pcr_soa_ref` | same P100 run | `1.033x` runtime vs `pcr_soa`, `0/6` wins | internal refs did not help | Archive |
| `assoc_backward` | P100 `20260618_182820_linear_assoc_focus_NvidiaTeslaP100` | `1.385x` faster than `thomas_batched`, but `1.570x` runtime vs `pcr_soa` | exact candidate, not competitive | Archive |
| split iterative `split_gs_*` | P100 linear/E2E split focus + local agreement | fast in solver-only cases | failed trace/physiology agreement | Abandon |
| Pallas Thomas / PCR spikes | P100/T4 Pallas focus runs through `20260618_200242_linear_pallas_focus_NvidiaTeslaT4` | no stable timing on current stack | Mosaic GPU limitations on P100/T4; T4 old-stack notebook only | Standby |
| standalone `triton_block_thomas` | T4 `20260618_205135_linear_triton_focus_NvidiaTeslaT4` | `2.684x` geomean speedup vs JAX `pcr_soa` | pure Triton/Torch path, not a clean JAX time-loop route | Archive for evidence |
| standalone `triton_pcr_soa` | T4 `20260618_210243_linear_triton_focus_NvidiaTeslaT4` | `1.619x` vs JAX `pcr_soa`; `1.697x` slower than Triton Thomas | no reason to pursue over Thomas | Archive |
| DLPack/Torch bridge | T4 `20260618_214520_linear_triton_focus_NvidiaTeslaT4` | only `1.060x` vs JAX, `2.522x` slower than pure Triton | host/framework bridge overhead too high | Archive |
| `jax_triton_block_thomas` | T4 `20260618_221506_linear_jax_triton_focus_NvidiaTeslaT4` | `1.991x` geomean speedup vs JAX `pcr_soa` | promising isolated solver | Archive until validation solved |
| `jax_triton_thomas` E2E | T4 `20260618_223213_e2e_jax_triton_focus_NvidiaTeslaT4` | `1.595x` geomean E2E kernel speedup, `7/8` wins | failed strict Vm agreement gates | Archive |

## Validation Notes

The JAX-Triton bridge was the closest custom-kernel route to a usable AxonScope
integration, but it did not pass the physiology agreement gate:

- `20260618_224225_validate_jax_triton_focus_NvidiaTeslaT4`: `0/16` rows passed
  strict thresholds versus `pcr_adaptive`; max absolute Vm error ranged from
  `0.041` to `102.583 mV`, with up to `2` extra activations.
- `20260618_224837_validate_jax_triton_thomas_focus_NvidiaTeslaT4`: `0/8` rows
  passed strict thresholds versus Thomas. `jax_triton_thomas` preserved
  activation counts on these cases, but still had max absolute Vm errors up to
  `95.508 mV`.

That makes the current conclusion conservative: the custom kernels are useful
evidence and future material, but not production solver routes.

## Code Organization After Cleanup

Active code remains under:

- `src/axonscope/solvers/`
- `benchmark/solvers/`
- `benchmark/kaggle/`

Archived or reproduction-only code lives under:

- `benchmark/archived_solver_spikes/`
- `benchmark/triton_solver/`
- `benchmark/jax_triton_solver/`
- `benchmark/cuda_ffi_solver/`
- `tests/archive/solver_spikes/`

The active Kaggle wrapper accepts only:

- `smoke`
- `linear`
- `linear_pcr_soa_trace`
- `e2e`
- `e2e_full`
- `both`

## Recommended Next Performance Target

The E2E runs repeatedly showed that dense `Vext` materialization and input
movement dominate many realistic cases once the solver route is reasonably fast.
The next optimization campaign should focus on `Vext`, not another solver spike.

The new workflow benchmark for that pass is:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset standard \
  --platforms cpu gpu \
  --run-counts 2 5 10 \
  --family-counts 5 25 50 \
  --repeats 3 \
  --warmups 1
```

Suggested next phases:

1. Add realistic workflow benchmarks for examples 6/7/8 to measure CPU vs GPU
   wall time, compile time, input generation, `Vext`, solve/runtime, and outputs.
2. Profile `Vext` materialization by batch size, fiber morphology, recording
   mode, and stimulation pattern.
3. Avoid dense `Vext` when possible: lazy/on-device generation, compressed
   electrode/stimulus representation, chunked batches, and reuse across runs.
4. Re-run E2E only after `Vext` changes, using `pcr_adaptive` as the retained
   GPU solver baseline.

## Result Folders

Key source folders used for this report:

- `benchmark/results/kaggle/20260617_220929_linear_pcr_soa_nomask_focus_NvidiaTeslaP100`
- `benchmark/results/kaggle/20260618_182820_linear_assoc_focus_NvidiaTeslaP100`
- `benchmark/results/kaggle/20260618_202917_linear_pcr_soa_layout_focus_NvidiaTeslaP100`
- `benchmark/results/kaggle/20260618_205135_linear_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_210243_linear_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_214520_linear_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_221506_linear_jax_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_223213_e2e_jax_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_224225_validate_jax_triton_focus_NvidiaTeslaT4`
- `benchmark/results/kaggle/20260618_224837_validate_jax_triton_thomas_focus_NvidiaTeslaT4`
