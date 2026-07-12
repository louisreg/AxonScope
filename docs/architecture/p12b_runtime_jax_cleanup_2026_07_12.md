# P12B Runtime/JAX Cleanup - 2026-07-12

P12B continues after the P12A runtime-contract sanity gate. The goal is to make
the runtime boundary cleaner for a future NumPy/SciPy runtime while keeping the
current JAX hot paths intact.

## Scope

Keep cable-specific solvers, kernels, JAX arrays, JIT behavior, and device
placement in `src/axonscope/runtime/jax/`.

Move or formalize only semantic contracts that can apply to another runtime:
input-lowering modes, recording/observer-output semantics, execution policy
shape, benchmark vocabulary, and result assembly concepts.

## Cleanup Done In This Pass

- Shared single-cable and double-cable host preparation in
  `runtime/jax/group_runner.py` now goes through one helper for
  `runtime.prepare` and `inputs.positions`.
- Input planning no longer depends on observer-output planning. Recording and
  observer choices decide outputs, not whether an extracellular input is
  compact or dense.
- Double-cable input planning now predicts compact `shared_current` and
  `scaled_shared_waveform` factorized inputs consistently with the runtime
  lowering path, including probe/full-style recording.
- Dead parameters were removed from JAX input-lowering functions where the
  runtime did not use them.
- Guardrails now check that input planning does not reintroduce
  `observer_plan` coupling.

## Current Boundary

`runtime/input_contract.py` owns runtime-neutral semantic labels:

- cable formulation: `single-cable`, `double-cable`;
- intracellular modes: `zero`, `dense`, `sparse_current_clamp`;
- extracellular modes: `zero`, `shared_current`,
  `scaled_shared_waveform`, `current_table`, `dense`.

`runtime/jax/input_lowering.py` owns the current JAX implementation of those
semantics. It may use JAX-specific containers internally, but benchmark and
inspection metadata should report the runtime-neutral mode labels.

## Contract Cleanup Pass

The next P12B cleanup pass moved host-side contracts that are not inherently JAX
specific out of `runtime/jax/`:

- `OutputPlan` moved from `runtime/jax/output_plan.py` to
  `runtime/output_contract.py`. JAX batch execution still consumes it, but the
  concept describes output sinks (`vm`, `vm_raster`, `none`) and chunking, not
  JAX kernels.
- Intracellular and extracellular input-format type labels moved to
  `runtime/input_contract.py`; `runtime/jax/input_lowering.py` now owns only
  the JAX implementation of those labels.
- Observer-output labels and VmRaster observer compatibility moved to
  `runtime/output_contract.py`. Public estimate/inspection helpers can now ask
  `runtime.execution` for those labels without routing through
  `runtime.jax.benchmark`.
- Guardrails now assert that `runtime/jax/output_plan.py` stays absent and that
  input/output labels remain runtime-neutral.
- Dense-equivalent input shape and byte-size helpers moved to
  `runtime/input_contract.py`, so JAX benchmark metadata no longer imports
  those generic memory-estimate helpers from JAX input lowering.
- Dead observer-output proxy helpers were removed from `runtime/jax/benchmark.py`.
  The active facade for estimate/inspection code is now `runtime.execution`,
  backed by the runtime-neutral output contract for observer-output labels.
- Public dispatch-record assembly moved from `runtime/jax/batch_results.py` to
  `runtime/result_assembly.py`. The JAX module now keeps only JAX kernel-output
  synchronization, pending VmRaster finalization, and padded kernel-output trim.

Validation:

```bash
python -m compileall -q src/axonscope tests/unit
python -m pytest -q tests/unit/test_architecture_guardrails.py tests/unit/test_inspection.py tests/unit/test_performance.py --tb=short
python -m pytest -q tests/unit/test_dispatcher.py --tb=short
```

Results: `compileall` passed, guardrails/inspection/performance passed
`102/102`, and dispatcher passed `56/56`.

## Benchmark Gate

After this cleanup, run the local CPU sanity gate:

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
  --output benchmark/results/p12b_runtime_cleanup_single_cpu

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
  --output benchmark/results/p12b_runtime_cleanup_double_cpu
```

Only run the matching GPU smoke if the cleanup touches a GPU-sensitive path or
if the local gate shows a suspicious regression.

## CPU Gate Result

The local CPU gate was run on 2026-07-12 after the P12B cleanup.

Artifacts:

- `benchmark/results/p12b_runtime_cleanup_single_cpu`
- `benchmark/results/p12b_runtime_cleanup_double_cpu`

Comparison against the P12A CPU gate:

| Cable | Stage | P12A total | P12B total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 4115.5 ms | 3184.5 ms | -22.6% |
| single-cable | `runtime.prepare` | 2367.4 ms | 1514.1 ms | -36.0% |
| single-cable | `inputs.extracellular` | 23.5 ms | 24.2 ms | +3.0% |
| single-cable | `kernel.enqueue` | 1317.2 ms | 1264.1 ms | -4.0% |
| single-cable | `kernel.wait` | 254.5 ms | 249.1 ms | -2.1% |
| double-cable | `curve.simulate` | 3560.6 ms | 3581.1 ms | +0.6% |
| double-cable | `runtime.prepare` | 1831.9 ms | 1851.6 ms | +1.1% |
| double-cable | `inputs.extracellular` | 26.3 ms | 26.3 ms | +0.3% |
| double-cable | `kernel.enqueue` | 1278.4 ms | 1277.2 ms | -0.1% |
| double-cable | `kernel.wait` | 321.9 ms | 321.1 ms | -0.3% |

The CPU sanity gate shows no obvious regression. The double-cable path is
essentially unchanged, and the shared preparation helper preserved the original
benchmark span names.

## GPU Smoke Gate Result

Because the shared preparation helper also touches the GPU execution path, two
small Kaggle P100 smoke runs were launched at commit `deb6954`.

Artifacts:

- `benchmark/results/kaggle/20260712_115203_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-single-gpu-deb6954`
- `benchmark/results/kaggle/20260712_115217_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12b-double-gpu-jt-deb6954`

Comparison artifacts:

- P12A single-cable GPU smoke:
  `benchmark/results/kaggle/20260712_111722_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p12a-runtime-contract-single-gpu-6e9a0f5`
- P12A double-cable GPU smoke with `jax-triton`:
  `benchmark/results/kaggle/20260712_112604_recruitment_curves_gpu_smoke_gpu_NvidiaTeslaP100_axs-p12a-double-gpu-jt-6e9a0f5`

All-phase totals from `summary.csv`:

| Cable | Stage | P12A total | P12B total | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 4074.7 ms | 4476.4 ms | +9.9% |
| single-cable | `runtime.prepare` | 1843.7 ms | 1968.7 ms | +6.8% |
| single-cable | `inputs.extracellular` | 108.9 ms | 136.1 ms | +25.0% |
| single-cable | `kernel.enqueue` | 1592.7 ms | 1790.5 ms | +12.4% |
| single-cable | `kernel.wait` | 61.8 ms | 55.7 ms | -9.7% |
| double-cable | `curve.simulate` | 9098.9 ms | 9064.4 ms | -0.4% |
| double-cable | `runtime.prepare` | 2340.5 ms | 2283.8 ms | -2.4% |
| double-cable | `inputs.extracellular` | 118.6 ms | 114.9 ms | -3.1% |
| double-cable | `kernel.enqueue` | 6144.6 ms | 6167.4 ms | +0.4% |
| double-cable | `kernel.wait` | 103.8 ms | 103.0 ms | -0.8% |

Steady repeat means use only `phase=repeat` and `iteration>0` simulations:

| Cable | Stage | P12A mean | P12B mean | Delta |
| --- | --- | ---: | ---: | ---: |
| single-cable | `curve.simulate` | 36.716 ms | 39.198 ms | +6.8% |
| single-cable | `runtime.prepare` | 0.056 ms | 0.051 ms | -8.9% |
| single-cable | `inputs.extracellular` | 2.047 ms | 2.291 ms | +11.9% |
| single-cable | `kernel.enqueue` | 14.185 ms | 15.555 ms | +9.7% |
| single-cable | `kernel.wait` | 6.908 ms | 6.655 ms | -3.7% |
| double-cable | `curve.simulate` | 37.578 ms | 38.232 ms | +1.7% |
| double-cable | `runtime.prepare` | 0.060 ms | 0.048 ms | -20.9% |
| double-cable | `inputs.extracellular` | 2.520 ms | 2.606 ms | +3.4% |
| double-cable | `kernel.enqueue` | 11.363 ms | 11.870 ms | +4.5% |
| double-cable | `kernel.wait` | 13.048 ms | 13.024 ms | -0.2% |

Interpretation:

- The double-cable GPU smoke is effectively unchanged and passes the P12B
  cleanup gate.
- The single-cable GPU smoke is still runnable, but this small case shows a
  modest enqueue/dispatch-side increase. Since `kernel.wait` did not worsen,
  this is not solver degradation, but it should be watched in the broader P11
  hot-path slices before claiming no GPU performance loss.
- The current cleanup therefore remains acceptable as a runtime-boundary
  cleanup, but it does not close the broader P12 performance-loss claim.

## Remaining Cleanup

- Continue auditing `runtime/jax/` for dead or duplicated host-side code.
- Keep `runtime/jax/reference_solvers.py` private to tests/reference
  equivalence. Do not promote those dense/reference routes into public examples
  or stable runtime policy.
- Keep diagnostic solver routes out of public examples and stable docs.
- Do not choose a new solver policy in P12B.
- Do not optimize cold start until the runtime contract and hot path are stable.
