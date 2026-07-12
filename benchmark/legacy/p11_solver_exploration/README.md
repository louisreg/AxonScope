# P11 Solver Exploration Archive

This directory keeps historical P11B/P11C solver probes and analysis scripts.
They are retained as evidence only and are not part of the active benchmark
surface.

Current solver validation should use:

- `benchmark/run.py`
- `benchmark/campaigns/double_cable_solver_policy.py`
- `benchmark/campaigns/single_cable_solver_policy.py`

The archived scripts may reference removed runtime probes such as PCR/SoA or
static-unrolled Triton variants. Do not use them to make current performance
claims without first porting the hypothesis to the active benchmark API.
