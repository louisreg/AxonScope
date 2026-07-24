# Examples

Examples are the executable learning path for AxonFleet. They use public
top-level APIs (`import axonfleet as axs`) and Pint-style quantities such as
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

`examples/advanced/` is organized by AxonFleet subsystem. These scripts are
still runnable examples, but they explain the moving parts more explicitly.

### Simulation Workflow

- `simulation_workflow/01_axon_population.py`: use an explicit
  `AxonPopulation` when a workflow already owns concrete instance rows.
- `simulation_workflow/02_pool_results.py`: inspect canonical pool results and
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
  `Model` classes with units, HH gates, coupled Markov occupancies, auxiliary
  state, diagnostics, generated-code inspection, and a short simulation.
- `axon_models/06_composite_recording_names.py`: label `Composite` components
  and inspect the resulting recording names.
- `axon_models/07_gaines_motor_sensory.py`: compare the Gaines motor and
  sensory myelinated families under the same intracellular pulse.
- `axon_models/08_mrg_markov_nav.py`: replace MRG nodal sodium with a public
  Nav1.1/Nav1.6 Markov composition while retaining the canonical cable path.
- `axon_models/09_validated_unmyelinated_families.py`: compare the Sundt,
  Tigerholm, Schild94, and Schild97 families with their validated intracellular
  configurations.
- `axon_models/10_nav_isoform_catalog.py`: run Nav1.1 through Nav1.9 in one
  shared public cable composition and compare stimulus/distal traces.

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
- `recording_analysis/03_activation_detection.py`: detect compact post-hoc
  activation events from recorded Vm.
- `recording_analysis/04_analysis_layer.py`: use structured post-hoc analyses,
  statuses, reports, and population denominators.
- `recording_analysis/05_dense_recording.py`: record dense gates,
  currents, and conductances through batch-native result manifests.

### Observers

- `observers/01_vmraster_observer_only.py`: run trace-free VmRaster
  observer-only simulations with `Recording.none()`.
- `observers/02_compact_activation.py`: retain one activation boolean per axon,
  apply solver-side blanking, and compare it visually with recorded Vm.
- `observers/03_compact_latency.py`: retain one first-crossing step per axon and
  compare compact latency visually with recorded Vm.
- `observers/04_compact_spike_count.py`: count repeated rearmed crossings with
  bounded timestamp storage and explicit overflow, then compare it with dense
  Vm and its raster.
- `observers/05_downsampled_vm_raster.py`: retain sparse spatial probes in
  event-preserving temporal windows and compare them with dense Vm.

### Protocols

- `protocols/01_threshold_vs_parameters.py`: estimate threshold curves over
  diameter and extracellular waveform parameters.
- `protocols/02_recruitment_waveforms.py`: compare recruitment curves for
  different extracellular waveforms.
- `protocols/03_protocol_result_views.py`: inspect protocol summaries as
  compact text, dataframes, direct view rows, and plots without running a solver.
- `protocols/04_parameter_sweep.py`: describe a generic value sweep, estimate
  its repeated work and peak memory, then run it through one reusable runner.

### Runtime

- `runtime/01_runtime_policy.py`: set JAX/CPU/precision through
  `ExecutionPolicy`.
- `runtime/02_pipeline_inspection.py`: inspect heterogeneous dispatch groups,
  preparation, lowering, kernel route, and result assembly without launching
  kernels.
- `runtime/03_solver_policy.py`: choose typed single-cable/double-cable solver
  policies and inspect the resolved double-cable backend route.
- `runtime/04_cache_policy.py`: inspect deterministic generated/runtime cache
  sections, generate one requested membrane target, and clean known artifacts.
- `runtime/05_study_plan.py`: compose lazy simulation plans into a named study,
  estimate and inspect it, execute it through one state-reusing runner, and
  request cooperative cancellation.

## With NRV

`examples/with_nrv/` is for integration with NRV, where NRV owns complex nerve
geometry/fiber placement and AxonFleet owns axon dynamics. Reusable handoff
logic lives in `axonfleet.integrations.nrv`; examples should use the two public
bridges, `population_from_nrv(...)` then `footprints_from_nrv(...)`, instead of
re-implementing fiber-table extraction, LIFE/FEM footprint sampling, or NRV
recruitment decoding. NRV geometry, population, electrode, and FEM setup remain
explicit NRV code in the example or benchmark, not package integration code.

- `01_synthetic_fascicle_geometry.py`: build the synthetic two-fascicle NRV
  tutorial geometry, sample NRV's LIFE/FEM footprint into AxonFleet stimulation
  objects, run an AxonFleet recruitment sweep, and plot recruitment plus
  activated fibers on the nerve cross-section.
- `02_realistic_fascicle_geometry.py`: build one realistic NRV nerve from the
  bundled histology contour image with `cv2`, then run the same NRV-to-AxonFleet
  footprint handoff and recruitment workflow as the synthetic example.
  NRV validation lives under `tests/nrv`; executable performance gates belong
  in `benchmark/examples/with_nrv_examples.py`.

## Tutorials

`examples/tutorials/` is reserved for notebook mini-courses. The intended style
is close to NRV's tutorial index: numbered lessons, one concept per notebook,
short explanations before code, and a workflow progression from first
simulation to population protocols and NRV integration.

## Benchmarks And Profiling

Small API-level instrumentation examples can live under `examples/advanced/`.
Reusable timing, memory, profiling, and reproducibility campaigns live under
`benchmark/`:

- `benchmark/README.md`: supported benchmark surface and retained commands.
- `benchmark/run.py`: shared launcher for benchmark scripts and presets.
- `benchmark/curves/threshold_curves.py`: activation-threshold cases
  matrix.
- `benchmark/curves/recruitment_curves.py`: recruitment case matrix.
- `benchmark/analysis/trace_summary.py`: event/profile artifact summaries.
