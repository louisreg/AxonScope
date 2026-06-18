# Triton Double-Cable Solver Spike

This folder contains a standalone Triton spike for the exact double-cable
2x2 block-tridiagonal solve. It is intentionally benchmark-only and does not
participate in AxonScope solver routing.

The first candidate is a simple exact block-Thomas kernel:

- one Triton program per fiber;
- one forward kernel stores modified upper blocks and RHS values;
- one backward kernel writes `x0/x1`;
- no JAX custom call yet.

The second candidate is a quick exact PCR_SOA-style Triton scout:

- global-memory init/stage/final kernels;
- one PCR stage launch per stride;
- same compact double-cable inputs and same residual/reference checks;
- intentionally not tuned until it shows a clear signal.

The point is deliberately modest: find out whether a small hand-written Triton
kernel is even in the right performance ballpark. If it does not clearly beat
the JAX `pcr_soa` baseline on the same GPU, close the Triton line.

Run locally on a CUDA machine with PyTorch and Triton:

```bash
python benchmark/triton_solver/bench_double_cable_triton.py \
  --batch-sizes 1024 2048 4096 \
  --nx 51 96 \
  --solvers triton_block_thomas triton_pcr_soa \
  --repeats 5
```

Kaggle combined baseline + Triton run:

```bash
python benchmark/kaggle/run_kernel.py \
  --username louisregnacq \
  --benchmark linear_triton_focus \
  --machine-shape NvidiaTeslaT4 \
  --poll-interval 60 \
  --wait-timeout 7200 \
  --max-status-fetch-failures 20
```
