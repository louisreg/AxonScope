# jax-triton Double-Cable Solver Spike

This folder contains benchmark-only experiments for calling Triton kernels from
JAX through `jax-triton`.

The current candidate is:

- `jax_triton_block_thomas`: exact 2x2 block-Thomas solve, implemented as two
  `jax_triton.triton_call` custom calls inside `jax.jit`.

This is the direct integration gate after the standalone Triton benchmark
showed a useful pure-kernel speedup and the eager Torch/DLPack bridge proved too
expensive for production routing.

The target regime is `Nx=30-100` and large batches, but `Nx` is a compile-time
specialized parameter, not a public hard limit. The Kaggle focus includes
`Nx=128` as a guard.

Run locally; this skips cleanly if `jax-triton` or CUDA GPU support is absent:

```bash
python benchmark/jax_triton_solver/bench_double_cable_jax_triton.py
```

Run on Kaggle:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_jax_triton_focus \
  --machine-shape NvidiaTeslaT4 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

