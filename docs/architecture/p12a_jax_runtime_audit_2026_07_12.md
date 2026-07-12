# P12A JAX Runtime Audit - 2026-07-12

This audit maps the current `src/axonscope/runtime/jax/` tree after the P11
solver cleanup and the first P12 runtime-contract pass. The goal is to clean and
homogenize host-side runtime orchestration without losing the current
single-cable or double-cable performance.

## Keep JAX-Specific

These modules own JAX arrays, JIT behavior, device placement, or solver kernels.
Do not move them to `axonscope.runtime`:

- `batch_kernels.py`
- `common.py`
- `solver_core.py`
- `jax_triton_double_cable.py`
- `large_population_solver.py`
- `kernels.py`
- `membrane_backend.py`
- `membrane_layout.py`
- `membrane_program.py`
- `model_ir_lowering.py`
- `rate_tables.py`
- `runtime.py`
- `runtime_caches.py`
- `execution_policy.py`
- `solver_engines/`
- `stimulation_runtime.py`

Changes in these files can affect solver performance or compilation behavior
and need benchmark evidence before any performance claim.

## Host-Side JAX Orchestration

These modules are JAX-owned today, but they contain host-side orchestration that
should converge across single-cable and double-cable execution:

- `group_runner.py`
- `runtime_preparation.py`
- `input_lowering.py`
- `input_batches.py`
- `batch_inputs.py`
- `benchmark_metadata.py`
- `recording.py`
- `recording_lowering.py`
- `output_plan.py`
- `batch_results.py`
- `observer_runtime.py`
- `benchmark.py`

P12A should clean these by small steps. Shared host orchestration can be
factored inside `runtime/jax` first; only semantic, non-JAX concepts should move
to `axonscope.runtime`.

## Runtime-Neutral Candidates

The following concepts are semantic enough to support a future NumPy/SciPy
runtime:

- cable formulation normalization;
- runtime input modes and capabilities;
- recording/output-plan semantics;
- observer-output route labels;
- planned input-lowering summaries;
- benchmark/inspection metadata vocabulary.

The first moved contract is `src/axonscope/runtime/input_contract.py`.

## Current P12A Cleanup Decisions

- `runtime_context` is the internal execution-context name. The old
  `backend_context` name is removed from active Python sources.
- Single-cable and double-cable batch execution share the same recording and
  observer/VmRaster lowering helper.
- Single-cable and double-cable batch execution share kernel wait, pending
  observer finalization, and dispatcher-result assembly helpers.
- The solver/kernel calls remain separate. This keeps cable-specific hot paths
  explicit and avoids speculative performance changes.

## Benchmark Gate

P12A cleanup is ready for a small non-regression benchmark once local tests are
green. Run the local CPU matrix first:

```bash
python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --cable single_cable \
  --recording observer_only \
  --n-axons 64 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss \
  --output benchmark/results/p12a_runtime_contract_single_cpu

python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --cable double_cable \
  --recording observer_only \
  --n-axons 64 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss \
  --output benchmark/results/p12a_runtime_contract_double_cpu
```

Then run the matching GPU smoke through Kaggle if the CPU gate is sane:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p12a-runtime-contract-single-gpu \
  --script recruitment_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --cable single_cable \
  --recording observer_only \
  --n-axons 1024 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss

python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p12a-runtime-contract-double-gpu \
  --script recruitment_curves \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --cable double_cable \
  --recording observer_only \
  --n-axons 1024 \
  --nx 89 \
  --precision fp32 \
  --double-cable-block-solver tiled_thomas \
  --tiled-thomas-block-b 64 \
  --repeats 2 \
  --warmups 1 \
  --memory-trace rss
```

These runs are not for choosing policy. They only check that P12A host-side
cleanup did not obviously regress the P11-sensitive single/double observer-only
paths.

## CPU Gate Result

The local CPU gate was run on 2026-07-12 at commit
`5266e8559018e43126ce2a360e39ce5d703b2c04` with `--memory-trace rss`,
`repeats=2`, `warmups=1`, `Naxons=64`, `Nx=89`, `fp32`, and
`observer_only` recording.

Artifacts:

- `benchmark/results/p12a_runtime_contract_single_cpu`
- `benchmark/results/p12a_runtime_contract_double_cpu`

Summary:

| Cable | curve.simulate total | runtime.prepare total | kernel.enqueue total | kernel.wait total | inputs.extracellular total | finalize_observer total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-cable | 4115.5 ms | 2367.4 ms | 1317.2 ms | 254.5 ms | 23.5 ms | 1.4 ms |
| double-cable | 3560.6 ms | 1831.9 ms | 1278.4 ms | 321.9 ms | 26.3 ms | 1.7 ms |

This is a sanity gate, not a performance-policy benchmark. On these small CPU
runs, wall time is dominated by cold preparation/JIT work. The shared
non-solver costs targeted by P12A are small: extracellular lowering is
approximately 23-26 ms total, observer finalization is approximately 1-2 ms
total, and public result conversion is approximately 1 ms total for each run.
The next gate is the matching GPU smoke before claiming that P12A has no
performance regression on the P11-sensitive GPU paths.

## GPU Gate Result

The GPU smoke gate was run on Kaggle P100 on 2026-07-12 at commit
`6e9a0f525a49ee93bb178be6c80937f162192648`.

Artifacts:

- `benchmark/results/kaggle/20260712_111722_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12a-runtime-contract-single-gpu-6e9a0f5/outputs/axonscope_benchmark_results_recruitment_curves_gpu_smoke_gpu_20260712_111724.zip`
- `benchmark/results/kaggle/20260712_112604_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12a-double-gpu-jt-6e9a0f5/outputs/axonscope_benchmark_results_recruitment_curves_gpu_smoke_gpu_20260712_112605.zip`

Summary:

| Cable | Solver | curve.simulate total | runtime.prepare total | kernel.enqueue total | kernel.wait total | inputs.extracellular total | finalize_observer total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-cable | auto | 4074.7 ms | 1843.7 ms | 1592.7 ms | 61.8 ms | 108.9 ms | 1.8 ms |
| double-cable | tiled Thomas b64 | 9098.9 ms | 2340.5 ms | 6144.6 ms | 103.8 ms | 118.6 ms | 1.9 ms |

The initial double-cable Kaggle run failed before useful timing because
`jax-triton` was not installed in the kernel. The corrected run passed with
`--pip-package jax-triton` and installed `jax-triton==0.3.1`.

This gate validates that the P12A runtime-contract cleanup still runs on the
P11-sensitive GPU observer-only paths. It does not prove that the small smoke
configuration is solver-bound: `kernel.wait` remains minor, while
`runtime.prepare` and `kernel.enqueue` dominate the short run.

Warm-only split:

| Cable | Slice | curve.simulate mean | simulation.run_pool mean | kernel.enqueue mean | kernel.dispatch_jax mean | kernel.wait mean | inputs.extracellular mean | observer.plan mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-cable | repeat phase, 6 simulations | 72.0 ms | 53.7 ms | 16.5 ms | 4.6 ms | 6.9 ms | 11.8 ms | 7.1 ms |
| double-cable | repeat phase, 6 simulations | 67.1 ms | 53.1 ms | 11.5 ms | 4.7 ms | 13.0 ms | 13.5 ms | 7.6 ms |
| single-cable | steady repeat amplitudes, 4 simulations | 36.7 ms | 31.8 ms | 14.2 ms | 4.6 ms | 6.9 ms | 2.0 ms | 0.1 ms |
| double-cable | steady repeat amplitudes, 4 simulations | 37.6 ms | 32.6 ms | 11.4 ms | 4.7 ms | 13.0 ms | 2.5 ms | 0.1 ms |

The first amplitude of each repeat still pays extra planning/input work. After
that, steady warm amplitudes are much cleaner: `runtime.prepare` is
approximately 0.06 ms, observer planning is approximately 0.1 ms, and
extracellular lowering is approximately 2-3 ms per simulation. Double-cable is
closer to being kernel-bound in this steady slice, but `kernel.wait` is still
only about 40% of `simulation.run_pool`; single-cable remains more dominated by
enqueue/orchestration than by device wait.

With `repeat_pool_policy=rebuild`, the benchmark still rebuilds the pool once
per repeat. That adds approximately 208 ms per repeat for single-cable and
251 ms per repeat for double-cable outside the `curve.simulate` timing. This is
useful for workflow timing, but should not be confused with steady simulation
hot-path timing.

## Remaining Before Benchmark Claim

- Add architecture guardrails for the runtime-context name and runtime input
  contract. [done]
- Run fast unit coverage for dispatcher, batch kernels, inspection, and
  performance views. [done]
- Run the local CPU benchmark gate above. [done]
- Run the matching GPU smoke gate. [done]
- Re-run broader P11 hotpath/realistic slices before making larger performance
  claims beyond this small P12A sanity gate.

## Do Not Do In P12A

- Do not choose a new GPU double-cable default policy.
- Do not move JAX array containers to runtime-neutral modules.
- Do not merge solver kernels.
- Do not broaden dense/factorized double-cable paths without equivalence tests
  and benchmark evidence.
