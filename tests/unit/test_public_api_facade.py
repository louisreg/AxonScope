import numpy as np
import pytest

import axonscope as axs


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


def test_public_single_recording_requires_voltage_with_observables():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="single-axon simulation currently always returns Vm"):
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
