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
  --solvers thomas pcr pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 3
```

Run the GPU-oriented sweep from the exact double-cable roadmap:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 1 8 128 512 1024 2048 4096 \
  --nx 16 32 51 64 96 100 128 \
  --solvers thomas pcr pcr_soa pcr_adaptive \
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
`pcr_soa_nomask` is a benchmark-only Phase 1C candidate that removes explicit
per-stage boundary `where` masks from the batch-native SoA PCR update, relying
on zero boundary-coupling invariants instead. It was neutral on the 2026-06-17
P100 `linear_pcr_soa_nomask_focus` run and should not be routed by `auto`.
`pcr_soa_shift` is a benchmark-only Phase 1C candidate that also replaces
clamped neighbor gathers with static slice/concat shifts at each PCR stride.
It was slower in all focused P100 cases and is standby/closed unless future
XLA lowering changes.
`thomas_batched` is a benchmark-only exact candidate that runs block Thomas as
one batch-first scan instead of an outer `vmap` over fibers.
`pcr_soa_hybrid_4`, `pcr_soa_hybrid_8`, and `pcr_soa_hybrid_16` are
benchmark-only Phase 1E candidates that run partial PCR, then exact block
Thomas on the independent residual chains.
`pcr_soa_transposed` is a benchmark-only exact candidate that keeps the public
RHS shape as `[B, Nx]` but runs PCR internally as `[Nx, B]`.
`assoc_backward` is a benchmark-only Phase 2A exact candidate that keeps the
Thomas forward elimination and replaces the reverse substitution scan with an
associative affine scan.
`assoc_transfer_dense` is a benchmark-only Phase 2B prototype that uses dense
5x5 transfer matrices and an associative prefix product; it is a
stability/performance probe, not an optimized backend. It is numerically
fragile on benchmark-like float32 systems, so do not spend Kaggle runs on it
unless a stabilized formulation is added.
`pallas_thomas_4`, `pallas_thomas_8`, `pallas_thomas_16`, and
`pallas_thomas_128` are benchmark-only Phase 3A spikes that run exact block
Thomas in a Pallas kernel. `pallas_thomas_128` is the historical full-block
spike; it exceeded P100 SMEM and remains standby. `pallas_thomas_4/8/16` are
the bounded-SMEM retries that were used to map Mosaic GPU lowering constraints.
The P100 retries ultimately closed the Thomas-Pallas line: `BLOCK_B=128`
matches Mosaic's 128-element strided-load preference but exceeds SMEM, while
small `BLOCK_B` variants fit SMEM but fail strided-load lowering. Keep these
variants out of routing and avoid more Kaggle runs until a Phase 3B PCR/hybrid
Pallas layout replaces the Thomas spike.
`pcr_soa_padded` is a benchmark-only Phase 1D candidate that pads `Nx` to
32/64/128 identity rows before the batch-native SoA solve; it is not a
`BatchOptions.double_cable_block_solver` value.
`split_jacobi_4`, `split_jacobi4_gs1`, `split_gs_2`, `split_gs_3`,
`split_gs_4`, `split_jacobi_8`, `split_gs_8`, and `split_richardson_4` are
historical benchmark-only Phase 1.5 candidates. The 2026-06-17 E2E agreement
smoke failed for the best split candidates, so split iterative approaches are
abandoned for the current optimization pass and should not be included in new
Kaggle runs except to reproduce old evidence.

Run the focused Phase 2A associative-backward comparison:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 1024 2048 4096 \
  --nx 51 64 96 \
  --solvers thomas thomas_batched assoc_backward pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 5
```

Latest P100 retest: `20260618_182820_linear_assoc_focus_NvidiaTeslaP100`
installed JAX `0.10.2`. `assoc_backward` remained faster than
`thomas_batched`, but did not beat `pcr_soa` generally (`1/9` wins), so it
stays benchmark-only/standby.

Capture a focused GPU JAX trace for PCR/SoA work:

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

On Kaggle, use the same matrix through the dedicated preset:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_trace \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

Downloaded outputs include the zipped `jax_traces/` folder under the
`linear_pcr_soa_trace` result directory. Open it with TensorBoard's profile
viewer or the generated Perfetto trace when `--jax-trace-create-perfetto` is
used locally.

Run the focused Phase 1C PCR_SOA no-mask candidate on Kaggle:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_pcr_soa_nomask_focus \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

Run the focused Phase 3A Pallas-Thomas spike:

```bash
python benchmark/solvers/bench_double_cable_linear_solvers.py \
  --batch-sizes 1024 2048 4096 \
  --nx 51 64 96 \
  --solvers thomas thomas_batched assoc_backward pallas_thomas_4 pcr_soa pcr_adaptive \
  --dtypes float32 \
  --warmups 1 \
  --repeats 5
```

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

Historical closed Phase 1.5 split E2E comparison:

```bash
python benchmark/solvers/bench_double_cable_end_to_end.py \
  --batch-sizes 1024 2048 4096 \
  --nx 51 96 \
  --nt 500 \
  --dt 0.01 \
  --recordings center \
  --iinj-modes none \
  --solvers pcr_adaptive split_gs_3 split_gs_4 \
  --warmups 1 \
  --repeats 2
```

Historical split trace validation that closed the split line:

```bash
python benchmark/solvers/validate_double_cable_solver_agreement.py \
  --batch-sizes 2 \
  --nx 51 \
  --nt 3 \
  --dt 0.05 \
  --recordings center \
  --iinj-modes none \
  --reference-solvers pcr_adaptive \
  --candidate-solvers split_gs_3 split_gs_4 \
  --warmups 0
```

The 2026-06-17 local smoke failed for `split_gs_3` and `split_gs_4`
(`~77 mV` center-trace error and false activations versus `pcr_adaptive`), so
split iterative approaches are abandoned for the current optimization pass
despite their timing wins. Keep the validation runner for future non-split
candidates; do not use the split timing benchmark alone as acceptance evidence.

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
