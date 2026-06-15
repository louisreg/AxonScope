# Examples

Examples are organized as a learning path.

Scripts use Pint quantities through AxonScope's top-level unit aliases: inputs
are written as physical values such as `500 * axs.um`, then converted to plain
arrays only at solver/plotting boundaries.

## Basic

`examples/basic/` contains short scripts that each introduce one concept:

- `example_01_stimulus_waveforms.py`: temporal waveforms in milliseconds and
  microamperes.
- `example_02_point_source_electrode.py`: analytical point-source footprints in
  micrometers, microamperes, and millivolts.
- `example_03_intracellular_hh.py`: one axon with one current clamp.
- `example_04_extracellular_mrg.py`: one myelinated axon with one point-source
  context.
- `example_05_pool_dispatch_basic.py`: a tiny pool dispatch workflow without
  NRV.
- `example_06_velocity_vs_diameter.py`: conduction velocity trends across
  diameters with automatic batch dispatch.
- `example_07_threshold_vs_diameter.py`: extracellular activation thresholds
  versus diameter with point-source stimulation and batched binary search.
- `example_08_recruitment_curve_population.py`: recruitment curve for a mixed
  unmyelinated/myelinated population around a point-source electrode.

These examples should stay compact and avoid benchmarking or profiling. Pool
examples should focus on the public `simulate_pool` workflow rather than
low-level dispatcher internals.

## Advanced

`examples/advanced/` contains complete workflows that combine concepts:

- `example_01_pool_dispatch_nrv.py`: build a heterogeneous axon pool and run
  dispatcher execution while preserving input order.
- `example_02_layout_options.py`: compare low-level `Layout` construction
  patterns and template-generated layouts.
- `example_03_custom_axon_from_scratch.py`: define a custom axon class from
  membranes, sections, and explicit layout elements.
- `example_04_stimulation_contexts.py`: reuse electrode geometry with different
  stimuli and combine multiple electrodes in one analytical context.
- `example_05_recording_options.py`: compare public recording policies,
  single-axon observable groups, and pool Vm retention modes.
- `example_06_activation_criterion.py`: evaluate post-hoc activation criteria
  from recorded Vm traces.
- `example_07_recruitment_curve.py`: estimate one activation threshold and a
  small recruitment curve with the protocol API.
- `example_08_root_axon_simulation.py`: use the executable `AxonSimulation`
  root object for one axon and a small population.
- `example_09_axon_population.py`: build an explicit `AxonPopulation` cohort
  and run it through the root simulation object.
- `example_10_typed_recording_signals.py`: request recording outputs with
  typed `axs.signals` selectors instead of raw strings.
- `example_11_typed_position_selectors.py`: evaluate activation criteria with
  typed `axs.positions` selectors instead of raw position strings.
- `example_12_cable_formulation.py`: use `axs.axons.CableFormulation` for
  custom axon layouts instead of raw formulation strings.
- `example_13_extracellular_footprint_drive.py`: separate static
  extracellular footprints from temporal drives and summed stimulation.
- `example_14_hotpath_benchmarking.py`: estimate simulation memory and collect
  opt-in hotpath timing spans for an observer-only run that keeps compact
  solver-side observations instead of full Vm traces.
- `example_15_preparation_signatures.py`: inspect deterministic signatures for
  reusable preparation inputs.
- `example_16_canonical_pool_results.py`: work with canonical cohort-backed
  pool results and per-axon result views.
- `example_17_analysis_layer.py`: evaluate structured scientific analyses,
  missing-input/status metadata, and lightweight online Vm observers.
- `example_18_solver_side_observers.py`: compare post-hoc analyses with
  solver-side observer-only execution using `Recording.none()`.

Advanced examples may include optional NRV input helpers, but should still favor
readability over measurement.

## Benchmarks And Profiling

Timing, memory, profiling, and reproducibility scripts live under `benchmark/`,
not `examples/`:

- `benchmark/runtime/pool_batch_demo.py`: scalar-vs-batch timing/profiling for
  imposed-field pool batches.
- `benchmark/runtime/pool_memory.py`: retained-output and tensor-size scans.
- `benchmark/runtime/environment_info_demo.py`: environment metadata capture.
