# Archived Solver Spikes

This folder contains solver experiments that are no longer active AxonScope
routes after the June 2026 optimization campaign.

Archived families include:

- Pallas Thomas/PCR kernels
- split iterative double-cable probes
- custom-kernel smoke tests moved out of the active unit suite

Related experiment folders:

- `benchmark/triton_solver/`
- `benchmark/jax_triton_solver/`
- `benchmark/cuda_ffi_solver/`
- `tests/archive/solver_spikes/`

These files are retained for evidence and reproduction only. They should not be
added to `BatchOptions.double_cable_block_solver`, `auto`, the active Kaggle
wrapper, or user-facing examples unless a future campaign revalidates them.
