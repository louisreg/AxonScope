# Examples

Examples are the executable learning path for AxonScope. They use public
top-level APIs (`import axonscope as axs`) and Pint-style quantities such as
`500 * axs.um`. Backend, solver, benchmark, and profiling internals stay out of
public examples.

## Basic

`examples/basic/` is the first tour. Scripts are short, runnable, and focused on
one visible result.

- `01_first_intracellular_simulation.py`: simulate one Hodgkin-Huxley axon with
  a current clamp.
- `02_stimuli_and_units.py`: build temporal stimuli with physical units.
- `03_point_source_footprint.py`: sample an analytical point-source helper into
  typed extracellular objects.
- `04_extracellular_mrg_simulation.py`: stimulate one myelinated MRG axon with a
  sampled point-source footprint.
- `05_population_pool_run.py`: run a small population with `axs.AxonSimulation`.
- `06_activation_velocity.py`: compare activation/velocity trends over
  diameters.
- `07_threshold_vs_diameter.py`: estimate extracellular activation thresholds.
- `08_recruitment_curve_population.py`: compute a recruitment curve for a mixed
  population.

## Advanced

`examples/advanced/` is organized by AxonScope subsystem. These scripts are
still runnable examples, but they explain the moving parts more explicitly.

### Simulation Workflow

- `simulation_workflow/01_axon_simulation_root.py`: use `AxonSimulation` as the
  executable root.
- `simulation_workflow/02_axon_population.py`: build and run explicit
  `AxonPopulation` cohorts.
- `simulation_workflow/03_pool_results.py`: inspect canonical pool results and
  per-axon views.

### Axon Models

- `axon_models/01_layout_options.py`: compare descriptive layout construction
  patterns.
- `axon_models/02_custom_axon_from_scratch.py`: define a custom axon from
  membranes, sections, and layout elements.
- `axon_models/03_cable_formulation.py`: choose typed single-cable or
  double-cable formulations.
- `axon_models/04_non_uniform_activation_function.py`: derive a non-uniform
  unmyelinated discretization from an activation-function proxy.
- `axon_models/05_custom_membrane_authoring.py`: write custom membrane
  `Model` classes with units, gates, non-gate state, diagnostics, generated-code
  inspection, and a short simulation.

### Stimulation

- `stimulation/01_stimulation_contexts.py`: reuse sampled drives with different
  temporal stimuli.
- `stimulation/02_extracellular_footprint_drive.py`: separate static
  extracellular footprints from temporal drives.
- `stimulation/03_intracellular_plus_extracellular.py`: combine a local
  intracellular clamp with sampled extracellular stimulation on the same axon.

### Recording And Analysis

- `recording_analysis/01_recording_options.py`: compare public recording
  policies, `RecordingPlan` values, and typed `axs.signals` descriptors.
- `recording_analysis/02_position_selectors.py`: select analysis targets
  with `axs.positions`.
- `recording_analysis/03_activation_criterion.py`: evaluate post-hoc activation
  criteria from recorded Vm.
- `recording_analysis/04_analysis_layer.py`: use structured post-hoc analyses,
  statuses, reports, and population denominators.
- `recording_analysis/05_vmraster_observer_only.py`: run trace-free VmRaster
  observer-only simulations with `Recording.none()`.

### Protocols

- `protocols/01_threshold_vs_parameters.py`: estimate threshold curves over
  diameter and extracellular waveform parameters.
- `protocols/02_recruitment_waveforms.py`: compare recruitment curves for
  different extracellular waveforms.
- `protocols/03_protocol_result_views.py`: inspect protocol summaries as
  compact text, dataframes, direct view rows, and plots without running a solver.

### Runtime

- `runtime/01_runtime_policy.py`: set JAX/CPU/precision through
  `ExecutionPolicy`.
- `runtime/02_preparation_signatures.py`: inspect preparation signatures for
  reusable inputs.
- `runtime/03_pipeline_inspection.py`: inspect heterogeneous dispatch groups,
  preparation, lowering, kernel route, and result assembly without launching
  kernels.

## With NRV

`examples/with_nrv/` is for integration with NRV, where NRV owns complex nerve
geometry/fiber placement and AxonScope owns axon dynamics. Reusable handoff
logic lives in `axonscope.integrations.nrv`; examples should use the two public
bridges, `population_from_nrv(...)` then `footprints_from_nrv(...)`, instead of
re-implementing fiber-table extraction, LIFE/FEM footprint sampling, or NRV
recruitment decoding. NRV geometry, population, electrode, and FEM setup remain
explicit NRV code in the example or benchmark, not package integration code.

- `01_realistic_fascicle_geometry.py`: build one realistic NRV nerve
  from the bundled histology contour image with `cv2`, sample NRV's LIFE/FEM
  footprint into AxonScope stimulation objects, run an AxonScope recruitment
  sweep, and plot recruitment plus activated fibers on the nerve cross-section.
  NRV validation and large timing/memory comparisons live under
  `benchmark/nrv_performance/`.

## Tutorials

`examples/tutorials/` is reserved for notebook mini-courses. The intended style
is close to NRV's tutorial index: numbered lessons, one concept per notebook,
short explanations before code, and a workflow progression from first
simulation to population protocols and NRV integration.

## Benchmarks And Profiling

Timing, memory, profiling, and reproducibility scripts live under `benchmark/`,
not `examples/`:

- `benchmark/runtime/benchmark_001_simple_batching.py`: compare batched and
  simulation-by-simulation execution.
- `benchmark/hotpaths/cold_run_progress.py`: inspect, estimate, and run one
  fixed shape twice with progress and hotpath spans to compare cold and warm
  execution.
- `benchmark/runtime/hotpath_observer_only_example.py`: inspect hotpath spans
  for trace-free VmRaster observer-only execution.
- `benchmark/notebooks/`: benchmark notebooks and Colab-style runs.
