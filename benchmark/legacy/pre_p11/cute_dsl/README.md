# CuTe DSL JAX Smoke

This folder contains a minimal CuTe DSL / JAX integration smoke based on the
official JAX CuTe DSL guide. It is deliberately separate from the solver
benchmarks for now: current Kaggle P100 (`sm_60`) and T4 (`sm_75`) GPUs are
below CuTe DSL's documented minimum of SM 8.0.

Install on a compatible CUDA GPU runtime:

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]"
```

Run the smoke:

```bash
python benchmark/cute_dsl/run_cute_dsl_smoke.py --n 4096
```

By default the runner exits successfully with a `skipped` status when CuTe DSL
dependencies or a compatible GPU are not available. Use `--strict` when a
compatible runtime is expected and a skip should fail the command.

