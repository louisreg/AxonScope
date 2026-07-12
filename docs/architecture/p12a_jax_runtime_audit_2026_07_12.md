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

## Remaining Before Benchmark Claim

- Add architecture guardrails for the runtime-context name and runtime input
  contract. [done]
- Run fast unit coverage for dispatcher, batch kernels, inspection, and
  performance views. [done]
- Run the local CPU benchmark gate above. [done]
- Run the matching GPU smoke gate before claiming no performance loss on GPU.

## Do Not Do In P12A

- Do not choose a new GPU double-cable default policy.
- Do not move JAX array containers to runtime-neutral modules.
- Do not merge solver kernels.
- Do not broaden dense/factorized double-cable paths without equivalence tests
  and benchmark evidence.
