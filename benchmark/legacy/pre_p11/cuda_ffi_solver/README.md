# CUDA FFI Double-Cable Solver Spike

This folder contains benchmark-only experiments for calling a custom CUDA
double-cable solver from inside JAX/XLA via `jax.ffi`.

The current candidate is:

- `cuda_ffi_block_thomas`: exact 2x2 block-Thomas solve, one CUDA thread per
  fiber, grouped into small thread blocks and using dynamic shared memory for
  the forward coefficients.

This is not public AxonScope solver routing. It is the integration gate after
the standalone Triton result showed that a custom block-Thomas kernel can beat
JAX `pcr_soa`, while the eager DLPack bridge was too expensive.

The target regime is `Nx=30-100` and large batches, but `Nx` is runtime dynamic
in this spike. The Kaggle focus includes `Nx=128` to avoid accidentally baking
in the target regime as a hard solver limit.

Run locally; this skips cleanly without CUDA/NVCC:

```bash
python benchmark/cuda_ffi_solver/bench_double_cable_cuda_ffi.py
```

Run on Kaggle:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_cuda_ffi_focus \
  --machine-shape NvidiaTeslaT4 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```

