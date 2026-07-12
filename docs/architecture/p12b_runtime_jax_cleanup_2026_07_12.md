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

## Remaining Cleanup

- Continue auditing `runtime/jax/` for dead or duplicated host-side code.
- Keep `experimental.py` only while the reference solver tests require it;
  archive or rename it once public examples no longer depend on those classes.
- Keep diagnostic solver routes out of public examples and stable docs.
- Do not choose a new solver policy in P12B.
- Do not optimize cold start until the runtime contract and hot path are stable.
