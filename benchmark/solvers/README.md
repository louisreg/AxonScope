# Solver-Focused Benchmarks

This folder isolates numerical solver kernels from AxonScope model building,
dispatch, input materialization, and result packaging.

The first benchmark targets the exact double-cable 2x2 block-tridiagonal
linear solve used inside each implicit myelinated time step.

## Double-Cable Linear Solvers

Colab notebook:

- `benchmark/solvers/colab_double_cable_linear_solvers.ipynb`

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
float64 reference unless `--skip-reference` is used.

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
