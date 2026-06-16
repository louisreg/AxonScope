# Solver-Focused Benchmarks

This folder isolates numerical solver kernels from AxonScope model building,
dispatch, input materialization, and result packaging.

The first benchmark targets the exact double-cable 2x2 block-tridiagonal
linear solve used inside each implicit myelinated time step.

## Double-Cable Linear Solvers

Colab notebook:

- `benchmark/solvers/colab_double_cable_linear_solvers.ipynb`
- `benchmark/solvers/colab_double_cable_end_to_end.ipynb`

Before opening it in Colab, publish the committed local revision:

```bash
git add -A
git commit -m "Benchmark double-cable linear solvers"
make bench-colab-push
```

The notebook clones `bench-colab`, installs `.[benchmark]`, runs one selectable
case (`smoke`, `gpu_matrix`, `gpu_full`, or `trace_pcr_adaptive`), and downloads
a zipped `benchmark/results/solvers/<run_id>/` folder.

Dry-run the default matrix:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run
```

Run a compact local smoke:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 8 128 \
  --nx 32 51 \
  --solvers thomas pcr pcr_soa pcr_adaptive split_gs_2 split_gs_3 split_gs_4 split_jacobi4_gs1 \
  --dtypes float32 \
  --warmups 1 \
  --repeats 3
```

Run the GPU-oriented sweep from the exact double-cable roadmap:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 1 8 128 512 1024 2048 4096 \
  --nx 16 32 51 64 96 100 128 \
  --solvers thomas pcr pcr_soa pcr_adaptive split_jacobi_4 split_jacobi4_gs1 split_gs_2 split_gs_3 split_gs_4 \
  --dtypes float32 float64 \
  --warmups 1 \
  --repeats 5
```

Each row records compile time, first compiled run time, steady-state min/median/
p95 time, `B * Nx` node-solves per second, and max error versus a Thomas
float64 reference unless `--skip-reference` is used. It also records max and
median relative block residual norms for the solved linear system.

`pcr_soa` is measured with the batch-native
`solve_block_tridiagonal_2x2_pcr_soa_batched(...)` path. `pcr_adaptive` uses
that same SoA path through `B <= 4096`, then falls back to matrix-layout `pcr`.
`thomas_batched` is a benchmark-only exact candidate that runs block Thomas as
one batch-first scan instead of an outer `vmap` over fibers.
`pcr_soa_hybrid_4`, `pcr_soa_hybrid_8`, and `pcr_soa_hybrid_16` are
benchmark-only Phase 1E candidates that run partial PCR, then exact block
Thomas on the independent residual chains.
`pcr_soa_transposed` is a benchmark-only exact candidate that keeps the public
RHS shape as `[B, Nx]` but runs PCR internally as `[Nx, B]`.
`pcr_soa_padded` is a benchmark-only Phase 1D candidate that pads `Nx` to
32/64/128 identity rows before the batch-native SoA solve; it is not a
`BatchOptions.double_cable_block_solver` value.
`split_jacobi_4`, `split_jacobi4_gs1`, `split_gs_2`, `split_gs_3`,
`split_gs_4`, `split_jacobi_8`, `split_gs_8`, and `split_richardson_4` are
benchmark-only Phase 1.5 candidates. They are fixed-iteration approximate split
solvers, not exact direct solvers, so judge them by both speed and
residual/error columns before considering any public routing.

Summarize one or more downloaded `summary.csv` files:

```bash
python benchmark/solvers/summarize_double_cable_linear_solvers.py \
  benchmark/results/solvers/<run_id>/gpu/summary.csv \
  --out benchmark/results/solvers/<run_id>/crossover_summary.csv
```

Run the end-to-end double-cable batch-kernel matrix:

```bash
python benchmark/solvers/bench_double_cable_end_to_end.py \
  --batch-sizes 512 1024 2048 \
  --nx 32 51 64 \
  --nt 500 1000 \
  --recordings none center full \
  --iinj-modes none dense_zero nonzero \
  --solvers auto thomas pcr_adaptive \
  --warmups 1 \
  --repeats 3
```

This runner builds MRG-like double-cable batches, materializes dense `Vext`,
optionally materializes dense `Iinj`, and records setup/runtime/input/kernel/
output byte metrics. It is the Phase 0.2 complement to the isolated linear
solver benchmark.

Capture a JAX profiler trace for one case:

```bash
python benchmark/solvers/profile_double_cable_linear_solvers.py \
  --solver pcr_adaptive \
  --batch-size 1024 \
  --nx 64 \
  --dtype float32
```

On some local environments JAX may print
`Can't import tensorflow.python.profiler.trace` while still writing the
`plugins/profile/...` trace files. Treat the exit code and generated files as
the source of truth.

Outputs are written under `benchmark/results/solvers/`, which is intentionally
ignored by git.
