# Benchmark Campaigns

Campaigns group the two canonical curve scripts into reproducible benchmark
matrices. They should call `benchmark/run.py`; they should not implement solver
or workload logic directly.

P11A keeps campaign code minimal until the concrete case list is reviewed. The
publication campaign should eventually cover:

- activation thresholds and block thresholds;
- recruitment curves;
- `dt`, `Nx`, `Naxons`;
- FP32 versus FP64;
- full Vm, probe Vm, observer-only outputs;
- single-cable, double-cable, and mixed populations;
- same-diameter and different-diameter cohorts;
- CPU and GPU;
- NRV comparison after the baseline adapter contract is implemented.

Every campaign must write a manifest with fixed presets, raw data paths, plot
paths, summary-table paths, git metadata, and hardware metadata.

