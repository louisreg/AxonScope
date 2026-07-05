# Solver Benchmarks

This folder contains the active AxonScope double-cable solver benchmarks. It is
kept intentionally small after the June 2026 optimization campaign:

- retained public solver routes: `auto`, `thomas`, `pcr`, `pcr_soa`,
  `pcr_adaptive`
- active linear benchmark: public solver comparison and optional JAX trace for
  the retained PCR/SoA path
- active E2E benchmark: real double-cable time-step loop with bounded and full
  matrices
- archived solver spikes: split iterative, associative-Thomas variants, Pallas,
  Triton, JAX-Triton, and CUDA FFI

The campaign summary and decision table live in:

- `benchmark/reports/double_cable_solver_optimization_2026_06.md`

## Linear Solver Benchmark

Dry-run the default retained-solver matrix:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run
```

Run a compact local smoke:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 8 128 \
  --nx 32 51 \
  --solvers thomas pcr pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 3
```

Run the GPU-oriented retained matrix:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 128 512 1024 2048 4096 \
  --nx 32 51 64 96 \
  --solvers thomas pcr pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 5
```

Each row records compile time, first compiled run time, steady-state
min/median/p95, node-solves per second, max error versus a Thomas float64
reference unless `--skip-reference` is used, and block residual norms.

`pcr_soa` uses the batch-native
`solve_block_tridiagonal_2x2_pcr_soa_batched(...)` path. `pcr_adaptive` uses
that same SoA path through `B <= 4096`, then falls back to matrix-layout `pcr`.

## JAX Trace

Capture a focused GPU trace for retained PCR/SoA work:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 2048 4096 \
  --nx 51 96 \
  --solvers pcr pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 2 \
  --skip-reference \
  --jax-trace
```

On Kaggle, use the dedicated active preset:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_trace \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

Downloaded outputs include `jax_traces/` under the
`linear_pcr_soa_trace` result directory. Open the trace with TensorBoard's
profile viewer or a generated Perfetto trace when
`--jax-trace-create-perfetto` is used locally.

## End-To-End Benchmark

Run the bounded E2E double-cable batch-kernel matrix:

```bash
python benchmark/solvers/bench_double_cable_end_to_end.py \
  --batch-sizes 512 2048 \
  --nx 51 96 \
  --nt 500 \
  --dt 0.01 \
  --recordings none center \
  --iinj-modes none dense_zero \
  --solvers auto thomas pcr_adaptive \
  --warmups 1 \
  --repeats 2
```

Run the larger matrix:

```bash
python benchmark/solvers/bench_double_cable_end_to_end.py \
  --batch-sizes 512 1024 2048 \
  --nx 51 64 96 \
  --nt 500 1000 \
  --dt 0.01 \
  --recordings none center full \
  --iinj-modes none dense_zero \
  --solvers auto thomas pcr_adaptive \
  --warmups 1 \
  --repeats 3
```

Kaggle equivalents:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark e2e \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 120 \
  --wait-timeout 21600
```

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark e2e_full \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 120 \
  --wait-timeout 21600
```

## Solver Agreement

The validation harness remains useful for reproducing historical physiology
checks or testing future candidates:

```bash
python benchmark/solvers/validate_double_cable_solver_agreement.py \
  --batch-sizes 128 512 \
  --nx 51 96 \
  --nt 300 \
  --dt 0.01 \
  --recordings center full \
  --iinj-modes none dense_zero \
  --reference-solvers thomas \
  --candidate-solvers pcr_adaptive \
  --warmups 1
```

Split iterative candidates are abandoned for production routing and are no
longer active benchmark choices. Use the archived results/report to reproduce
that decision rather than adding them back to the standard runner.

## Archived Spikes

The code for non-retained candidates is intentionally outside the active solver
package:

- `benchmark/archived_solver_spikes/`: Pallas kernels and archived unit checks
- `benchmark/triton_solver/`: standalone Torch/Triton block-Thomas and bridge
  experiments
- `benchmark/jax_triton_solver/`: JAX-Triton block-Thomas experiment
- `benchmark/cuda_ffi_solver/`: CUDA FFI prototype
- `tests/archive/solver_spikes/`: archived tests for those experiments

These spikes are useful evidence, but they are not active `BatchOptions` solver
routes and are not accepted by the standard Kaggle wrapper.
