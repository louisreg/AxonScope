import numpy as np
import pytest

import axonscope as axs
from axonscope.solvers.observer_runtime import VM_RASTER_OBSERVATION_KEY


def test_public_unmyelinated_template_and_simulate():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon)
    sim.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.02 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )

    result = axs.simulate(sim, duration=0.1 * axs.ms, dt=0.05 * axs.ms)

    assert result.Vm.shape == (2, 11)
    assert np.asarray(result.t).shape == (2,)
    assert isinstance(axon, axs.axons.Unmyelinated)
    assert axon.resolved_formulation == "single-cable"
    assert result.axon is axon
    assert result.simulation is sim


def test_public_axon_is_descriptive_and_simulation_owns_protocol():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    assert not hasattr(axon, "add_current_clamp")
    assert not hasattr(axon, "add_extracellular_context")
    assert not hasattr(axon, "set_position")
    assert not hasattr(axon, "plot_geometry")
    assert callable(axon.layout.plot)

    sim = axs.AxonInstance(axon, y=20.0 * axs.um)
    clamp = axs.IntracellularCurrentClamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.02 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )
    sim.add_intracellular_context(context=clamp)

    assert sim.axon is axon
    assert sim.n_compartments == axon.n_compartments
    assert sim.y_um == pytest.approx(20.0)
    assert not hasattr(sim, "intracellular_clamps")
    assert len(sim.intracellular_contexts) == 1
    assert sim.intracellular_contexts[0] is clamp


def test_public_simulation_owns_one_extracellular_context():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon)
    electrode = axs.PointSourceElectrode(
        x=50.0 * axs.um,
        z=1000.0 * axs.um,
        stimulus=axs.Stimulus.constant(0.0 * axs.uA),
    )
    context = axs.AnalyticalExtracellularContext(electrodes=[electrode])

    sim.add_extracellular_context(context=context)

    assert sim.extracellular_context is context
    assert sim.extracellular_contexts == (context,)
    assert sim.use_extracellular
    assert not hasattr(sim, "clear_extracellular_contexts")

    sim.clear_extracellular_context()

    assert sim.extracellular_context is None
    assert sim.extracellular_contexts == ()
    assert not sim.use_extracellular


def test_public_simulate_rejects_partial_final_time_step():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(ValueError, match="integer multiple"):
        axs.simulate(axon, duration=0.1 * axs.ms, dt=0.03 * axs.ms)


def test_public_recording_full_requests_observables():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate(
        axon,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.full(),
    )

    assert result.recordings is not None
    assert "Vm" in result.recordings
    assert "gates" in result.recordings
    assert "currents" in result.recordings
    assert "conductances" in result.recordings


def test_scalar_observer_only_run_returns_compact_observations_without_vm():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon)
    sim.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    recorded = axs.simulate(sim, duration=0.1 * axs.ms, dt=0.05 * axs.ms)
    compact = axs.simulate(
        sim,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    assert compact.recordings is None
    with pytest.raises(AttributeError, match="Vm recording"):
        _ = compact.Vm
    assert compact.observations is not None
    raster = compact.observations[VM_RASTER_OBSERVATION_KEY]
    assert raster.names == ("activation",)
    assert raster.words.shape == (1, 1, 1, 1)
    probe_index = int(np.asarray(raster.original_indices)[0, 0])
    np.testing.assert_array_equal(
        raster.unpack()[0, 0, 0],
        np.asarray(recorded.Vm)[:, probe_index] >= -80.0,
    )


def test_solver_side_peak_voltage_observer_is_not_supported():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon)
    peak = axs.analysis.PeakVoltage(target=axs.positions.CENTER)

    with pytest.raises(NotImplementedError, match="threshold-style Vm"):
        axs.simulate(
            sim,
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.none(),
            observers=[peak],
        )


def test_pool_observer_only_run_returns_compact_observations_without_vm():
    axons = []
    for amplitude in (0.4, 0.5):
        axon = axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
        sim = axs.AxonInstance(axon)
        sim.add_current_clamp(
            position=50.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.02 * axs.ms,
                duration=0.04 * axs.ms,
                amplitude=amplitude * axs.nA,
            ),
        )
        axons.append(sim)

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    recorded = axs.simulate_pool(
        axons,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(),
    )
    compact = axs.simulate_pool(
        axons,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    assert compact.recording_manifest.available == ()
    assert len(compact.cohorts) == 1
    assert compact.observations is not None
    assert compact.cohorts[0].observations is compact.observations
    raster = compact.observations[VM_RASTER_OBSERVATION_KEY]
    assert raster.words.shape == (2, 1, 1, 1)
    expected = np.stack(
        [
            np.asarray(recorded[row].Vm)[:, 0] >= -80.0
            for row in range(len(axons))
        ],
        axis=0,
    )
    np.testing.assert_array_equal(
        raster.unpack()[:, 0, 0, :],
        expected,
    )
    with pytest.raises(ValueError, match="Vm recording"):
        _ = compact[0].Vm
    assert compact[0].observations[VM_RASTER_OBSERVATION_KEY].words.shape == (1, 1, 1, 1)


def test_pool_observer_only_zero_field_does_not_materialize_dense_vstim():
    axons = []
    for amplitude in (0.4, 0.5):
        axon = axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
        sim = axs.AxonInstance(axon)
        sim.add_current_clamp(
            position=50.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.02 * axs.ms,
                duration=0.04 * axs.ms,
                amplitude=amplitude * axs.nA,
            ),
        )
        axons.append(sim)

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    axs.enable_benchmark("/tmp/axonscope-zero-vstim-test", print_summary=False, save=False)
    try:
        compact = axs.simulate_pool(
            axons,
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.none(),
            observers=[activation],
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert compact.observations is not None
    assert report is not None
    extracellular_events = [
        event for event in report.events if event.name == "inputs.extracellular"
    ]
    assert len(extracellular_events) == 1
    metadata = extracellular_events[0].metadata
    assert metadata["input_format"] == "zero_no_context"
    assert "vstim_mid" not in metadata
    assert metadata["skipped_dense_vstim_shape"] == [2, 2, 11]


def test_pool_extracellular_only_retained_output_skips_dense_zero_iinj():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    stimulus = axs.Stimulus.pulse(
        start=0.02 * axs.ms,
        duration=0.04 * axs.ms,
        amplitude=20.0 * axs.uA,
    )
    electrode = axs.PointSourceElectrode(
        x=50.0 * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode],
        sigma=0.3 * axs.S_per_m,
    )
    axons = []
    for y_um in (-10.0, 10.0):
        instance = axs.AxonInstance(axon, y=y_um * axs.um)
        instance.add_extracellular_context(context=context)
        axons.append(instance)

    axs.enable_benchmark("/tmp/axonscope-zero-iinj-test", print_summary=False, save=False)
    try:
        result = axs.simulate_pool(
            axons,
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert result[0].Vm.shape == (2, 1)
    assert report is not None
    intracellular_events = [
        event for event in report.events if event.name == "inputs.intracellular"
    ]
    assert len(intracellular_events) == 1
    metadata = intracellular_events[0].metadata
    assert metadata["input_format"] == "zero_no_intracellular_context"
    assert "iinj_mid" not in metadata
    assert metadata["skipped_dense_iinj_shape"] == [2, 2, 11]
    group_events = [event for event in report.events if event.name == "dispatch.group.total"]
    assert len(group_events) == 1
    group_metadata = group_events[0].metadata
    assert group_metadata["memory_estimate_total_nbytes"] > 0
    assert group_metadata["memory_estimate_components_nbytes"]["vstim_mid"] > 0
    assert group_metadata["memory_estimate_components_nbytes"]["iinj_dense"] == 0


def test_public_recording_signals_filter_single_result():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate(
        axon,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording(signals=[axs.signals.Vm, axs.signals.GATES]),
    )

    assert result.recordings is not None
    assert set(result.recordings) == {"Vm", "gates"}


def test_public_signal_descriptors_are_extensible():
    custom = axs.Signal(
        id=axs.SignalId("teaching_custom_signal"),
        result_key="teaching_custom_signal",
        unit="arbitrary",
        description="User-defined teaching signal.",
        quantity_type=float,
    )

    recording = axs.Recording(signals=[custom])

    assert recording.signals == (custom,)
    assert recording.voltage is False
    assert custom.id == axs.SignalId("teaching_custom_signal")


def test_public_single_recording_requires_voltage_with_observables():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="observable-only recording"):
        axs.simulate(
            axon,
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.only(axs.signals.GATES),
        )


def test_public_single_recording_spatial_filter_is_explicitly_unwired():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="spatial single-axon"):
        axs.simulate(
            axon,
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        )


def test_public_generic_unmyelinated_from_membrane():
    membrane = axs.membranes.HodgkinHuxley(celsius=6.3)
    axon = axs.axons.Unmyelinated(
        membrane=membrane,
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
    )

    assert axon.n_compartments == 11
    assert axon.layout.sections[0].membrane is membrane


def test_public_composite_membrane_can_build_generic_unmyelinated():
    membrane = axs.membranes.Sundt()
    axon = axs.axons.Unmyelinated(
        membrane=membrane,
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
    )

    assert axon.n_compartments == 11
    assert axon.layout.sections[0].membrane is membrane


def test_public_pool_accepts_simulation_protocols():
    axon_a = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon_b = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon_a.set_position(y=20.0 * axs.um, z=30.0 * axs.um)
    axon_b.set_position(y=-40.0 * axs.um, z=10.0 * axs.um)

    result = axs.simulate_pool(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )

    assert len(result) == 2
    assert result[0].simulation.y_um == 20.0
    assert result[1].simulation.z_um == 10.0


def test_public_simulate_pool_accepts_unit_duration_and_dt():
    axon = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )

    result = axs.simulate_pool(
        [axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )

    assert len(result) == 1
    assert result[0].Vm.shape == (2, 11)


def test_public_simulate_pool_returns_canonical_cohort_result():
    axon_model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(axon_model, y=-20.0 * axs.um)
    axon_b = axs.AxonInstance(axon_model, y=20.0 * axs.um)
    recording = axs.Recording.center(axs.signals.Vm)

    result = axs.simulate_pool(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=recording,
    )

    assert isinstance(result, axs.AxonSimulationResult)
    assert not isinstance(result, list)
    assert len(result) == 2
    assert len(result.cohorts) == 1
    assert result.cohorts[0].Vm.shape == (2, 2, 1)
    assert result.axons == (axon_model, axon_model)
    assert result.simulations == (axon_a, axon_b)
    assert result.diagnostics[0]["pool_index"] == 0

    manifest = result.recording_manifest
    assert isinstance(manifest, axs.RecordingManifest)
    assert manifest.policy is recording
    assert manifest.requested_signals == (axs.signals.Vm,)
    assert manifest.available_signals == (axs.signals.MEMBRANE_VOLTAGE,)
    vm_manifest = manifest.signal(axs.signals.Vm)
    assert isinstance(vm_manifest, axs.RecordedSignal)
    assert vm_manifest.result_key == "Vm"
    assert vm_manifest.unit == "millivolt"
    assert vm_manifest.cohort_indices == (0,)
    assert vm_manifest.cohort_shapes == ((2, 2, 1),)
    assert vm_manifest.cohort_count == 1
    assert result[0].recording_manifest is manifest
    with pytest.raises(TypeError, match="signals values"):
        manifest.signal("Vm")

    first = result.axon(0)
    assert isinstance(first, axs.AxonResultView)
    assert first.index == 0
    assert first.simulation is axon_a
    assert first.record_indices == (5,)
    assert first.trace_values(index=0)[0].shape == (2,)

    dense_vm = result.signal(axs.signals.Vm)
    assert dense_vm.shape == (2, 2, 1)
    np.testing.assert_allclose(np.asarray(first.Vm), dense_vm[0])
    assert result.views[1].simulation is axon_b
    assert result[1].to_sim_result().simulation is axon_b

    with pytest.raises(TypeError, match="signals values"):
        result.signal("Vm")

    with pytest.raises(ValueError, match="exactly one"):
        _ = result.single


def test_public_simulate_pool_single_view_and_heterogeneous_cohorts():
    short_axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    long_axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.6 * axs.um,
        compartments=13,
        celsius=6.3 * axs.degC,
    )

    one = axs.simulate_pool(
        [short_axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )
    assert one.single.axon is short_axon
    assert one.single.Vm.shape == (2, 11)

    mixed = axs.simulate_pool(
        [short_axon, long_axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )
    assert [view.Vm.shape for view in mixed] == [(2, 11), (2, 13)]
    assert len(mixed.cohorts) == 2
    with pytest.raises(ValueError, match="heterogeneous cohorts"):
        mixed.signal(axs.signals.Vm)


def test_public_axon_population_normalizes_instances_and_axons():
    plain_axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    wrapped_axon = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=120.0 * axs.um,
            diameter=0.6 * axs.um,
            compartments=13,
            celsius=6.3 * axs.degC,
        ),
        y=20.0 * axs.um,
    )

    population = axs.AxonPopulation([plain_axon, wrapped_axon], name="demo")

    assert len(population) == 2
    assert population.name == "demo"
    assert not population.is_single
    assert population.axons == (plain_axon, wrapped_axon.axon)
    assert population.instances[0].axon is plain_axon
    assert population.instances[1] is wrapped_axon
    assert tuple(population) == population.instances
    assert repr(population) == "AxonPopulation(n=2, name='demo')"


def test_public_axon_population_rejects_empty_and_invalid_entries():
    with pytest.raises(ValueError, match="at least one"):
        axs.AxonPopulation([])

    with pytest.raises(TypeError, match="invalid entries"):
        axs.AxonPopulation([object()])


def test_public_simulate_pool_accepts_axon_population():
    population = axs.AxonPopulation.single(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )

    result = axs.simulate_pool(
        population,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )

    assert population.is_single
    assert len(result) == 1
    assert result[0].simulation is population[0]


def test_public_root_axon_simulation_runs_single_instance():
    instance = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    recording = axs.Recording.full()
    simulation = axs.AxonSimulation(
        instance,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=recording,
    )

    result = simulation.run()

    assert simulation.is_single
    assert not simulation.is_population
    assert result.simulation is instance
    assert result.recording is recording
    assert result.Vm.shape == (2, 11)


def test_public_root_axon_simulation_keeps_one_row_population_lifecycle():
    population = axs.AxonPopulation.single(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    simulation = axs.AxonSimulation(
        population,
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    )

    results = simulation.run()

    assert simulation.population is population
    assert simulation.is_population
    assert not simulation.is_single
    assert len(results) == 1
    assert results[0].simulation is population[0]
    assert results[0].diagnostics["dispatch_method"] == "scalar"


def test_public_root_axon_simulation_runs_population():
    first = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        ),
        y=10.0 * axs.um,
    )
    second = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        ),
        y=20.0 * axs.um,
    )
    simulation = axs.AxonSimulation(
        [first, second],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    results = simulation.run()

    assert simulation.is_population
    assert len(results) == 2
    assert results[0].simulation is first
    assert results[1].simulation is second
    assert results[0].record_indices == (5,)
    assert results[1].Vm.shape == (2, 1)


def test_public_root_axon_simulation_rejects_empty_population():
    with pytest.raises(ValueError, match="at least one"):
        axs.AxonSimulation([], duration=0.1 * axs.ms, dt=0.05 * axs.ms)


def test_public_root_axon_simulation_rejects_pool_solver_object():
    first = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    second = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    simulation = axs.AxonSimulation(
        [first, second],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        solver=axs.solvers.CrankNicholson(),
    )

    with pytest.raises(NotImplementedError, match="solver_options"):
        simulation.run()


def test_public_pool_recording_center_maps_to_batch_recording():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate_pool(
        [axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    fiber_result = result[0]
    assert fiber_result.Vm.shape == (2, 1)
    assert fiber_result.record_indices == (5,)


def test_public_pool_recording_rejects_observable_groups():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="pool recording currently supports Vm only"):
        axs.simulate_pool(
            [axon],
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.full(),
        )

    with pytest.raises(NotImplementedError, match="pool recording currently supports Vm only"):
        axs.simulate_pool(
            [axon],
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording.only(axs.signals.GATES),
        )


def test_public_pool_recording_rejects_unwired_filters():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="position-based batch recording"):
        axs.simulate_pool(
            [axon],
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording(signals=axs.signals.Vm, positions=[50.0 * axs.um]),
        )

    with pytest.raises(NotImplementedError, match="temporal recording subsampling"):
        axs.simulate_pool(
            [axon],
            duration=0.1 * axs.ms,
            dt=0.05 * axs.ms,
            recording=axs.Recording(signals=axs.signals.Vm, sample_dt=0.05 * axs.ms),
        )


def test_public_pool_recording_indices_maps_to_batch_recording():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate_pool(
        [axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.indices([0, 10], axs.signals.Vm),
    )

    fiber_result = result[0]
    assert fiber_result.Vm.shape == (2, 2)
    assert fiber_result.record_indices == (0, 10)
    expected_positions_um = axon.layout.position_values(unit=axs.um)[[0, 10]]
    np.testing.assert_allclose(
        fiber_result.position_values(unit=axs.um),
        expected_positions_um,
    )


def test_public_myelinated_mrg_template_and_section_layout():
    layout = axs.membranes.SectionLayout(
        node=axs.membranes.AxNode(),
        stin=axs.membranes.Passive(),
    )

    assert layout.membrane_for("NODE").kind == "axnode"
    assert layout.membrane_for("stin").kind == "passive"

    axon = axs.axons.MRG(diameter=5.7 * axs.um, nodes=3)
    assert axon.nodes == 3
    assert len(axon.node_indices) >= 2


def test_public_myelinated_constructor_accepts_mrg_like_layout():
    section_membranes = axs.membranes.SectionLayout(
        node=axs.membranes.AxNode(),
        mysa=axs.membranes.Passive(Rm=1e6, EL=-80.0),
        flut=axs.membranes.Passive(Rm=1e6, EL=-80.0),
        stin=axs.membranes.Passive(Rm=1e6, EL=-80.0),
    )
    layout = axs.axons.mrg_like_layout(
        diameter=5.7 * axs.um,
        nodes=3,
        membranes=section_membranes,
    )

    axon = axs.axons.Myelinated(
        layout=layout,
    )

    assert axon.nodes == 3
    flat = axs.axons.flatten_layout(axon.layout)
    assert set(kind.lower() for kind in flat.section_names) == {"node", "mysa", "flut", "stin"}
    assert len(flat.membrane_models) == axon.n_compartments
    assert np.asarray(axon.x_nodes_um).shape == np.asarray(axon.node_indices).shape
