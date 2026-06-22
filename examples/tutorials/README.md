# Tutorials

Tutorials are notebook mini-courses. They are meant to teach concepts in order,
not to benchmark or exhaust every option. Each notebook should keep the same
shape:

1. What this lesson teaches.
2. The minimal model or population.
3. One runnable workflow.
4. A plot or structured result to inspect.
5. A short "what changed" recap.

Planned notebook sequence:

- `01_first_axon_simulation.ipynb`: units, stimuli, one axon, one recording.
- `02_extracellular_stimulation.ipynb`: point-source electrode, footprint, and
  imposed extracellular drive.
- `03_populations_thresholds_recruitment.ipynb`: populations, activation
  thresholds, and recruitment curves.
- `04_recording_analysis_vmraster.ipynb`: retained Vm, post-hoc analyses, and
  trace-free VmRaster observer-only output.
- `05_runtime_policy_and_inspection.ipynb`: execution policy, CPU/GPU intent,
  precision, memory estimates, and pipeline inspection.
- `06_with_nrv_geometry.ipynb`: NRV-owned geometry and fiber placement mapped
  into AxonScope simulations.

Rules for tutorial notebooks:

- keep all user-facing code on the public `axonscope as axs` API;
- keep each notebook runnable on CPU with small defaults;
- move benchmarking and profiling notebooks to `benchmark/notebooks/`;
- include an executable script counterpart when a tutorial introduces a new
  public concept.
