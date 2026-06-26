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
- `05_population_pool_run.py`: run a small population with `axs.simulate_pool`.
- `06_activation_velocity.py`: compare activation/velocity trends over
  diameters.
- `07_threshold_vs_diameter.py`: estimate extracellular activation thresholds.
- `08_recruitment_curve_population.py`: compute a recruitment curve for a mixed
  population.

## Advanced

`examples/advanced/` is organized by AxonScope subsystem. These scripts are
still runnable examples, but they explain the moving parts more explicitly.

### Object Model

- `object_model/01_axon_simulation_root.py`: use `AxonSimulation` as the
  executable root.
- `object_model/02_axon_population.py`: build and run explicit
  `AxonPopulation` cohorts.
- `object_model/03_pool_results.py`: inspect canonical pool results and
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

### Stimulation

- `stimulation/01_stimulation_contexts.py`: reuse sampled drives with different
  temporal stimuli.
- `stimulation/02_extracellular_footprint_drive.py`: separate static
  extracellular footprints from temporal drives.

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

### Runtime

- `runtime/01_runtime_policy.py`: set JAX/CPU/precision through
  `ExecutionPolicy`.
- `runtime/02_preparation_signatures.py`: inspect preparation signatures for
  reusable inputs.
- `runtime/03_pipeline_inspection.py`: inspect heterogeneous dispatch groups,
  preparation, lowering, kernel route, and result assembly without launching
  kernels.
- `runtime/04_cold_run_progress.py`: run the same batchable pool twice with
  human-readable progress to make cold JAX compilation and warm execution
  visible.

## With NRV

`examples/with_nrv/` is for integration with NRV, where NRV owns complex nerve
geometry/fiber placement and AxonScope owns axon dynamics.

- `01_realistic_fascicle_geometry_comparison.py`: recreate NRV's bundled
  realistic fascicle-contour workflow, convert fractional NRV `node_shift`
  values into AxonScope MRG `x_shift`, sample NRV's LIFE/FEM footprint into
  AxonScope stimulation objects, build per-fascicle AxonScope recruitment
  curves, and run one NRV validation amplitude. The script also exposes
  `ExampleConfig.geometry_mode = "synthetic_4_fascicles"` for reproducible
  benchmark geometry: one synthetic nerve, four circular fascicles, NRV-owned
  fiber placement, and the same LIFE/FEM footprint handoff.

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
- `benchmark/runtime/hotpath_observer_only_example.py`: inspect hotpath spans
  for trace-free VmRaster observer-only execution.
- `benchmark/notebooks/`: benchmark notebooks and Colab-style runs.
